#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import copy
import multiprocessing as mp
import argparse
import numpy as np
import random
import torch
import torch.distributed as dist
import sys
import os
import time
import re
import logging
import collections
import pandas as pd
from pathlib import Path
import functools

from uimnet.algorithms.base import Algorithm
import submitit

from uimnet import algorithms
from uimnet import datasets
from uimnet import metrics
from uimnet import utils
from uimnet import workers
from uimnet import __DEBUG__
from omegaconf import OmegaConf


class Trainer(workers.Worker):

  def __init__(self):

    super(Trainer, self).__init__()
    """
    Trainer
    """

    # INITIAL WORKER STATE
    #
    # Kwargs-vals. could use inspect but this saner.
    self.cfg = None
    self.Algorithm = None
    self.dataset = None

    self.algorithm = None
    self.datanode = None

    self.prediction_metrics = None
    self.train_iter = 0
    self.current_epoch = 0
    self.records = collections.defaultdict(list)

  @utils.timeit
  def save(self, cfg):
    output_dir = cfg.output_dir

    # 1 Serialize algorithm
    self.algorithm.save_state(output_dir)
    # 2 Checkpointing trainer
    output_path = Path(output_dir)
    if utils.is_not_distributed_or_is_rank0():
      checkpoint = dict(rng_state=torch.get_rng_state(),
                        train_iter=self.train_iter,
                        current_epoch=self.current_epoch)

      torch.save(checkpoint, f=output_path / 'trainer_state.pt')
      #3 saving records
      torch.save(self.records, output_path / 'trainer_records.pt')

    filepath = output_path.absolute() /  f'cfg_rank_{cfg.experiment.rank}.yaml'

    with open(filepath , 'w') as fp:
      OmegaConf.save(config=cfg, f=fp.name)

    return

  @utils.timeit
  def load(self, output_dir):

    output_path = Path(output_dir)
    # 1 Load algorithm state
    self.algorithm.load_state(output_dir, map_location=self.cfg.experiment.device)
    trainer_state = torch.load(output_path / 'trainer_state.pt')
    torch.set_rng_state(trainer_state["rng_state"])
    self.train_iter = trainer_state['train_iter'] #+ 1
    self.current_epoch = trainer_state['current_epoch'] #+ 1

    self.records = torch.load(output_path/ 'trainer_records.pt')
    utils.message(f'Continuing from checkpoint at epoch {self.current_epoch}')

    return

  def checkpoint(self, *args, **kwargs):

    if utils.is_not_distributed_or_is_rank0():
      new_callable = Trainer()
      utils.write_trace('train.interrupted', dir_=self._cfg.output_dir)
      return submitit.helpers.DelayedSubmission(new_callable,
                                                cfg=self._cfg,
                                                Algorithm=self.Algorithm,
                                                dataset=self.dataset
                                                )

  def maybe_load_checkpoint(self, cfg):
    # Maybe load from checkpoint if exists
    output_path = Path(cfg.output_dir).absolute()
    if (output_path / 'trainer_state.pt').exists():
      utils.message(f'Checkpoint found. Currrent epoch before loading {self.current_epoch}')
      self.load(cfg.output_dir)
      utils.message(f'Resuming training from epoch @ {self.current_epoch}')

    return

  @utils.timeit
  def train_epoch(self, cfg):

    _device = cfg.experiment.device
    train_loader = self.datanode.get_loader('train',
                                            batch_size=cfg.dataset.batch_size,
                                            shuffle=True,
                                            pin_memory=True if 'cuda' in _device else False,
                                            num_workers=cfg.experiment.num_workers)
    if utils.is_distributed():
      train_loader.sampler.set_epoch(self.current_epoch)


    self.algorithm.train()
    self.datanode.train()

    records = []
    for batch in train_loader:
      self.train_iter += 1
      batch = utils.apply_fun(functools.partial(utils.to_device, device=cfg.experiment.device), batch)
      record = self.algorithm.update(batch['x'], batch['y'], self.current_epoch)
      record.update(dict(epoch=self.current_epoch, train_iter=self.train_iter))
      records += [record]
      if __DEBUG__ and self.train_iter > 2:
        break

    records = utils.apply_fun(utils.to_numpy, records)
    self.records['mainloop'] += records

    df = pd.DataFrame.from_records(records, index=['epoch', 'train_iter'])
    utils.message('\n' + df.mean(axis=0).round(4).to_string())

    if utils.is_not_distributed_or_is_rank0():
      with open(os.path.join(cfg.output_dir, 'mainloop_data.csv'), 'a') as fp:
        df.to_csv(fp, mode='a', header=fp.tell() == 0)

    return

  def stamp_record(self, record):
    return record

  @utils.timeit
  def evaluate(self, cfg):
    _device = cfg.experiment.device

    self.algorithm.eval()
    self.datanode.eval()

    records = []
    dataloaders = self.datanode.get_loaders(batch_size=cfg.dataset.batch_size,
                                            shuffle=True,
                                            pin_memory=True if 'cuda' in _device else False,
                                            num_workers=cfg.experiment.num_workers)

    for sname, split_loader in dataloaders.items():
      with torch.no_grad():
        record = self.prediction_metrics(self.algorithm, split_loader)
        record.update(epoch=self.current_epoch, train_iter=self.train_iter, split=sname)
        records += [self.stamp_record(record)]

    records = utils.apply_fun(utils.to_numpy, records)

    self.records['evaluation'] += records

    df = pd.DataFrame.from_records(records, index=['epoch'])
    utils.message('\n' + df.round(4).to_string())

    if utils.is_not_distributed_or_is_rank0():
      with open(os.path.join(cfg.output_dir, 'evaluation_data.csv'), 'a') as fp:
        df.to_csv(fp, mode='a', header=fp.tell() == 0)

    return records

  def __call__(self, cfg, Algorithm, dataset):
    elapsed_time = time.time()

    # First store an immutable copy of the arguments. Immutability is
    # surprisingly a hard property to attribute in python.
    self._cfg = copy.deepcopy(cfg)
    self.Algorithm = Algorithm
    self.dataset = dataset
    ## The keywords above will be accessed in n the checkpoint callback, in
    ## order to serialize the instantiation of this class and its arguments.

    self.setup(cfg)  # Setup modifies cfg. It needs a state on the worker.
    self.cfg = cfg
    if utils.is_not_distributed_or_is_rank0():
      utils.write_trace('train.running', dir_=cfg.output_dir)

    utils.message(cfg)

    self.datanode = datasets.SplitDataNode(
      dataset=dataset,
      transforms=datasets.TRANSFORMS,
      splits_props=cfg.dataset.splits_props,
      seed=cfg.dataset.seed)

    num_classes = self.datanode.splits['train'].num_classes
    self.algorithm = Algorithm(num_classes=num_classes,
                               arch=cfg.algorithm.arch,
                               device=cfg.experiment.device,
                               use_mixed_precision=cfg.algorithm.use_mixed_precision,
                               seed=cfg.algorithm.seed,
                               sn=cfg.algorithm.sn,
                               sn_coef=cfg.algorithm.sn_coef,
                               sn_bn=cfg.algorithm.sn_bn)

    self.algorithm.initialize(self.datanode.dataset)
    utils.message(self.algorithm)

    if utils.is_distributed():
      self.prediction_metrics = metrics.FusedPredictionMetrics()
    else:
      self.prediction_metrics = metrics.PredictionMetrics()

    self.maybe_load_checkpoint(cfg)
    utils.maybe_synchronize()

    utils.message('Starting mainloop.')

    for epoch in range(self.current_epoch + 1, cfg.experiment.num_epochs + 1):
      utils.message(f'Starting epoch {epoch}')
      self.current_epoch = epoch
      self.train_epoch(cfg)

      is_last_epoch = epoch == cfg.experiment.num_epochs
      if epoch % cfg.experiment.evaluate_every == 0 or is_last_epoch:
        self.evaluate(cfg)

      if epoch % cfg.experiment.checkpoint_every == 0 or is_last_epoch:
        self.save(cfg)

      utils.maybe_synchronize()

    utils.message('Training complete. Finalizing...')
    self.finalize(cfg)

    return {'data': self.records,
            'cfg': cfg,
            'elapsed_time': time.time() - elapsed_time,
            'status': 'done'}


  def finalize(self, cfg):

    if utils.is_not_distributed_or_is_rank0():
      utils.write_trace( 'train.done', dir_=cfg.output_dir)
    utils.message(f'Training completed.')

    return


if __name__ == "__main__":
    pass
