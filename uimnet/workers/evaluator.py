#! UY
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#

#/usr/bin/env python3
import pickle
import os
import time
import functools
import itertools
import pickle
import numpy as np  # BUG: Load numpy before multiprocessing
import multiprocessing as mp
from pathlib import Path
import pandas as pd
import torch
import torch.distributed as dist
import collections
import pickle
import copy
from torch.utils.data import dataloader
import shutil

from omegaconf import OmegaConf
import submitit

from uimnet import datasets
from uimnet import measures
from uimnet import workers
from uimnet import utils
from uimnet import datasets as uidatasets

from uimnet import __DEBUG__

from uimnet.metrics.auc import AUC
from uimnet.metrics.out_domain import InAsIn, InAsOut, OutAsIn, OutAsOut

METRICS = [AUC, InAsIn, InAsOut, OutAsIn, OutAsOut]
OUT = 'easy'

@utils.timeit
def get_loaders_datanodes(partitions, train_cfg, loaders_kwargs, seed):

  datanodes = {}
  loaders = {}

  key = ('train', 'in')
  datanodes[key] = datasets.SplitDataNode(
    partitions[key],
    datasets.TRANSFORMS,
    splits_props=train_cfg.dataset.splits_props,
    seed=train_cfg.dataset.seed
  )
  for split, loader in datanodes[key].get_loaders(**loaders_kwargs).items():
    loaders[(split, key[1])] = loader

  keys = [('val', 'in'), ('val', OUT)]
  for key in keys:
    datanodes[key] = datasets.SimpleDataNode(
      partitions[key],
      datasets.TRANSFORMS,
      seed= seed
    )
    loaders[key] = datanodes[key].get_loader(**loaders_kwargs)


  return loaders, datanodes

class Evaluator(workers.Worker):

  def __init__(self):
    super(Evaluator, self).__init__()

    self.model_dir = None
    self.eval_cfg = None
    self.train_cfg = None
    self.algorithm = None
    self.partitions = None
    self.trace = None


  def checkpoint(self, *args, **kwargs):
    if utils.is_not_distributed_or_is_rank0():
      new_callable = Evaluator()
      utils.write_trace(f'{self.trace}.interrupted', dir_=self.train_cfg.output_dir)
      return submitit.helpers.DelayedSubmission(new_callable,
                                                model_dir=self.model_dir,
                                                eval_cfg=self.eval_cfg,
                                                train_cfg=self.train_cfg,
                                                Algorithm=self.Algorithm,
                                                partitions=self.partitions
                                                )
  def __call__(self, model_dir, eval_cfg, train_cfg, Algorithm, Measure, partitions):

    self.model_dir = model_dir
    self.eval_cfg = copy.deepcopy(eval_cfg)
    self.train_cfg = copy.deepcopy(train_cfg)
    self.Algorithm = Algorithm
    self.datasets  = datasets

    self.setup(eval_cfg)
    eval_cfg, train_cfg = [utils.guard(el) for el in (eval_cfg, train_cfg)]

    self.trace = f'ood_{Measure.__name__}'

    # Check if training was completed
    path = Path(self.model_dir)
    if not utils.trace_exists('train.done', dir_=str(path)):
      utils.message(f'Train completion tracer missing!')
      return dict(status='missing', records=None)

    utils.message(f'Train completion tracer found.')
    utils.write_trace(f'{self.trace}.running', dir_=str(path))
    # utils.write_trace('ood_evaluation.running', dir_=str(path))

    ##############
    ## Datasets ##
    ##############

    loaders_kwargs = dict(
      batch_size=train_cfg.dataset.batch_size,
      shuffle=False,
      pin_memory=True if 'cuda' in eval_cfg.experiment.device else False,
      num_workers=eval_cfg.experiment.num_workers)
    loaders, datanodes = get_loaders_datanodes(partitions, train_cfg,
                                               loaders_kwargs=loaders_kwargs,
                                               seed=eval_cfg.dataset.seed)
    for datanode in datanodes.values():
      datanode.eval()

    num_classes = partitions[('train', 'in')].num_classes


    ###############
    ## Algorithm ##
    ###############
    self.algorithm = Algorithm(num_classes=num_classes,
                               arch=train_cfg.algorithm.arch,
                               device=eval_cfg.experiment.device,
                               use_mixed_precision=train_cfg.algorithm.use_mixed_precision,
                               seed=train_cfg.algorithm.seed,
                               sn=train_cfg.algorithm.sn,
                               sn_coef=train_cfg.algorithm.sn_coef,
                               sn_bn=train_cfg.algorithm.sn_bn
                               )


    utils.message(eval_cfg)
    self.algorithm.initialize()
    self.algorithm.load_state(train_cfg.output_dir,
                              map_location=eval_cfg.experiment.device)

    records = []
    self.algorithm.eval()
    with torch.no_grad():
      for temperature_mode in ['initial', 'learned']:

        self.algorithm.set_temperature(temperature_mode)
        measure = Measure(algorithm=self.algorithm)
        measure.estimate(loaders[('train', 'in')])

        measurements = collections.defaultdict(list)
        features = collections.defaultdict(list)
        labels = collections.defaultdict(list)
        for (partition, split) in [('eval', 'in'),
                                   ('val', 'in'), ('val', OUT)]:
          key = (partition, split)
          for i, batch in enumerate(loaders[key]):
            x, y = batch['x'].cuda(), batch['y'].cuda()
            # if key == ('val', OUT):
            #   x = torch.zeros_like(x).normal_()
            _measurement = measure(x)
            _labels = y.view(-1, 1)
            if utils.is_distributed():
              _measurement = torch.cat(utils.all_gather(_measurement), dim=0)
              _labels = torch.cat(utils.all_gather(_labels), dim=0)

            measurements[key] += [_measurement.detach().cpu()]
            labels[key] += [_labels.detach().cpu()]
            if __DEBUG__ and i > 1:
              break

        measurements = {k: torch.cat(l) for k, l in dict(measurements).items()}
        labels = {k: torch.cat(l) for k, l in dict(labels).items()}

        if utils.is_not_distributed_or_is_rank0():
          torch.save(measurements, path / f'{Measure.__name__}_measurements.pth')
          torch.save(labels, path / f'{Measure.__name__}_labels.pth')

        # Evaluating metrics
        for Metric in METRICS:

          metric = Metric(measurements[('eval', 'in')])
          value = metric(measurements[('val', 'in')],
                        measurements[('val', OUT)])

          record = dict(metric=metric.__class__.__name__,
                        measure=Measure.__name__,
                        value=value,
                        temperature_mode=temperature_mode)

          record.update(utils.flatten_nested_dicts(train_cfg))
          records += [record]

    # Saving records
    if utils.is_not_distributed_or_is_rank0():
      save_path = path / f'{Measure.__name__}_results.pkl'
      shutil.rmtree(save_path, ignore_errors=True)

      with open(save_path, 'wb') as fp:
        pickle.dump(records, fp, protocol=pickle.HIGHEST_PROTOCOL)

      utils.message(pd.DataFrame.from_records(utils.apply_fun(utils.to_scalar, records)).round(4))
      utils.write_trace(f'{self.trace}.done', dir_=str(path))

    return records

if __name__ == '__main__':
  pass
