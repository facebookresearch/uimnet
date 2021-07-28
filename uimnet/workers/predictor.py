#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import copy
import torch
import torch.distributed as dist
import time
from pathlib import Path
import pickle

from uimnet.algorithms.base import Algorithm
import submitit

from uimnet import algorithms
from uimnet import datasets
from uimnet import metrics
from uimnet import utils
from uimnet import workers
from uimnet import __DEBUG__
from omegaconf import OmegaConf


class Predictor(workers.Worker):

  def __init__(self):

    super(Predictor, self).__init__()
    """
    Trainer
    """

    # INITIAL WORKER STATE
    #
    # Kwargs-vals. could use inspect but this saner.
    self._prediction_cfg = None
    self._train_cfg = None
    self.Algorithm = None
    self.train_dataset = None
    self.val_dataset = None

    self.prediction_metrics = None


  def checkpoint(self, *args, **kwargs):

    if utils.is_not_distributed_or_is_rank0():
      new_callable = Predictor()
      utils.write_trace('prediction.interrupted', dir_=self._train_cfg.output_dir)
      return submitit.helpers.DelayedSubmission(new_callable,
                                                prediction_cfg=self._prediction_cfg,
                                                train_cfg=self._train_cfg,
                                                Algorithm=self.Algorithm,
                                                train_dataset=self.train_dataset,
                                                val_datase=self.val_dataset
                                                )

  def __call__(self, prediction_cfg, train_cfg, Algorithm, train_dataset, val_dataset):
    elapsed_time = time.time()

    self._prediction_cfg = copy.deepcopy(prediction_cfg)
    self._train_cfg = copy.deepcopy(train_cfg)
    self.Algorithm = Algorithm
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    self.setup(prediction_cfg)

    if utils.is_not_distributed_or_is_rank0():
      utils.write_trace('prediction.running', dir_=train_cfg.output_dir)

    utils.message(train_cfg)
    utils.message(prediction_cfg)

    train_datanode = datasets.SplitDataNode(
      dataset=train_dataset,
      transforms=datasets.TRANSFORMS,
      splits_props=train_cfg.dataset.splits_props,
      seed=train_cfg.dataset.seed).eval()

    val_datanode = datasets.SimpleDataNode(
      dataset=val_dataset,
      transforms=datasets.TRANSFORMS,
      seed=prediction_cfg.experiment.seed).eval()

    num_classes = train_datanode.splits['train'].num_classes
    assert num_classes == val_datanode.dataset.num_classes

    algorithm = Algorithm(num_classes=num_classes,
                          arch=train_cfg.algorithm.arch,
                          device=prediction_cfg.experiment.device,
                               use_mixed_precision=train_cfg.algorithm.use_mixed_precision,
                               seed=train_cfg.algorithm.seed,
                               sn=train_cfg.algorithm.sn,
                               sn_coef=train_cfg.algorithm.sn_coef,
                               sn_bn=train_cfg.algorithm.sn_bn)

    algorithm.initialize(train_datanode.dataset)
    utils.message(algorithm)

    if utils.is_distributed():
      prediction_metrics = metrics.FusedPredictionMetrics()
    else:
      prediction_metrics = metrics.PredictionMetrics()


    algorithm.load_state(train_cfg.output_dir, map_location=prediction_cfg.experiment.device)
    loaders_kwargs = dict(batch_size=prediction_cfg.dataset.batch_size,
                          shuffle=False,
                          pin_memory=True if 'cuda' in prediction_cfg.experiment.device else False,
                          num_workers=prediction_cfg.experiment.num_workers)

    loaders = train_datanode.get_loaders(**loaders_kwargs)
    loaders['val'] = val_datanode.get_loader(**loaders_kwargs)

    all_records = []
    algorithm.eval()
    train_keys = utils.flatten_nested_dicts(train_cfg)
    for temperature_mode in ['initial', 'learned']:
      algorithm.set_temperature(temperature_mode)
      utils.message(f'Algorithm temperature: {algorithm.temperature.tau}')
      for split_name, loader in loaders.items():
        with torch.no_grad():
          record = prediction_metrics(algorithm, loader)
        record.update(train_keys)
        record['split'] = split_name
        record['temperature_mode'] = temperature_mode
        all_records += [record]

    all_records = utils.apply_fun(utils.to_scalar, all_records)
    if utils.is_not_distributed_or_is_rank0():
      with open(Path(train_cfg.output_dir) / 'predictive_records.pkl', 'wb') as fp:
        pickle.dump(all_records, fp, protocol=pickle.HIGHEST_PROTOCOL)

    if utils.is_not_distributed_or_is_rank0():
      utils.write_trace(trace='prediction.done', dir_=train_cfg.output_dir)
    utils.message(f'Prediction completed.')

    import pandas as pd
    df = pd.DataFrame.from_records(all_records).round(4)
    keys = ['temperature_mode', 'split', 'ACC@1','ACC@5', 'NLL', 'ECE']
    print(df[keys])

    return {'data': all_records,
            'train_cfg': train_cfg,
            'prediction_cfg': prediction_cfg,
            'elapsed_time': time.time() - elapsed_time,
            'status': 'done'}




if __name__ == "__main__":
    pass
