#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import pickle
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

import torch.nn.functional as F
import torch.nn.parameter as P
import submitit

from uimnet.algorithms.base import Algorithm
from uimnet import algorithms
from uimnet import datasets
from uimnet import metrics
from uimnet import utils
from uimnet import workers
from uimnet import __DEBUG__
from omegaconf import OmegaConf

from uimnet import __DEBUG__

class Calibrator(workers.Worker):

  OPTIM_KWARGS=dict(lr=0.01, max_iter=50)
  def __init__(self):

    super(Calibrator, self).__init__()
    """
    Calibrator
    """
    # INITIAL WORKER STATE
    #
    # Kwargs-vals. could use inspect but this saner.
    self.cfg = None
    self.Algorithm = None
    self.dataset = None

    self.algorithm = None
    self.datanode = None


    return

  def checkpoint(self, *args, **kwargs):

    if utils.is_not_distributed_or_is_rank0():
      new_callable = Calibrator()
      utils.write_trace('calibration.interrupted', dir_=str(self.train_cfg.output_dir))
      return submitit.helpers.DelayedSubmission(new_callable,
                                                cfg=self.train_cfg,
                                                Algorithm=self.Algorithm,
                                                dataset=self.dataset
                                                )

  def __call__(self, calibration_cfg, train_cfg, Algorithm, dataset):

    elapsed_time = time.time()
    self.train_cfg = train_cfg
    self.calibration_cfg = calibration_cfg
    self.Algorithm = Algorithm
    self.dataset = dataset

    self.setup(calibration_cfg)  # Setup modifies cfg. It needs a state on the worker.
    utils.message(calibration_cfg)

    if utils.is_not_distributed_or_is_rank0():
      if not utils.trace_exists('train.done', dir_=train_cfg.output_dir):
        err_msg = f'Training not finished'
        raise RuntimeError(err_msg)
      utils.write_trace('calibration.running', dir_=train_cfg.output_dir)

    self.datanode = datasets.SplitDataNode(
      dataset=dataset,
      transforms=datasets.TRANSFORMS,
      splits_props=train_cfg.dataset.splits_props,
      seed=train_cfg.dataset.seed)
    num_classes = self.datanode.splits['train'].num_classes

    self.algorithm = Algorithm(num_classes=num_classes,
                               arch=train_cfg.algorithm.arch,
                               device=calibration_cfg.experiment.device,
                               use_mixed_precision=train_cfg.algorithm.use_mixed_precision,
                               seed=train_cfg.algorithm.seed,
                               sn=train_cfg.algorithm.sn,
                               sn_coef=train_cfg.algorithm.sn_coef,
                               sn_bn=train_cfg.algorithm.sn_bn
                               )

    self.algorithm.initialize()
    utils.message(self.algorithm)

    adapt_state = train_cfg.experiment.distributed and not utils.is_distributed()
    self.algorithm.load_state(train_cfg.output_dir, map_location=calibration_cfg.experiment.device,
                              adapt_state=adapt_state)


    # Preparing for calibration
    self.datanode.eval()
    self.algorithm.eval()
    eval_loader = self.datanode.get_loader('eval',
                                            batch_size=calibration_cfg.dataset.batch_size,
                                            shuffle=False,
                                            pin_memory=True if 'cuda' in calibration_cfg.experiment.device else False,
                                            num_workers=calibration_cfg.experiment.num_workers)

    collected = collections.defaultdict(list)
    utils.message('Collecting logits and targets')
    with torch.no_grad():
      for i, batch in enumerate(eval_loader):
        batch = utils.apply_fun(functools.partial(utils.to_device, device=calibration_cfg.experiment.device), batch)
        _logits = self.algorithm(batch['x'])
        _y = batch['y']
        if utils.is_distributed():
          _logits = torch.cat(utils.all_gather(_logits), dim=0)
          _y = torch.cat(utils.all_gather(_y), dim=0)
        collected['logits'] += [_logits.cpu()]
        collected['y'] += [_y.cpu()]
        if __DEBUG__ and i > 2:
          break
      collected = dict(collected)
      collected = {k: torch.cat(v, dim=0) for k, v in collected.items()}
      #utils.message(collected['y'].unique())
      #collected = {k: utils.all_cat(v, dim=0) for k, v in collected.items()}
    utils.message(f'logits.shape: {collected["logits"].shape}, y.shape: {collected["y"].shape}')
    self.algorithm.cpu()


    utils.message(f'Temperature before reinitialization {self.algorithm.temperature.tau}')
    utils.message(f'Reinitializing temperature')
    self.algorithm.temperature.reinitialize_temperature()
    utils.message(f"Temperature after reinitialization {self.algorithm.temperature.tau}")
    with torch.no_grad():
      self.algorithm.temperature.tau.fill_(1.5)
    utils.message(f'Tau before calibration: {self.algorithm.temperature.tau}')
    self.algorithm.temperature.tau.requires_grad = True

    tau_initial = self.algorithm.temperature.tau.data.clone()
    utils.message(f'Temperature before calibration {tau_initial}')
    optimizer = torch.optim.LBFGS([self.algorithm.temperature.tau], **self.OPTIM_KWARGS)

    def _closure():
      loss_value = F.cross_entropy(collected['logits'] / self.algorithm.temperature.tau, collected['y'])
      utils.message(f'Calibration loss value {loss_value}')
      loss_value.backward()
      utils.message(f'Temperature gradient after backward: {self.algorithm.temperature.tau.grad}')
      return loss_value

    optimizer.step(_closure)
    self.algorithm.temperature.tau.requires_grad = False
    utils.message(f'Temperature after calibration {self.algorithm.temperature.tau}')

    utils.message('Finalizing calibration')
    # self.algorithm.cuda()
    utils.message('serializing model.')
    if utils.is_not_distributed_or_is_rank0():
      self.algorithm.save_state(train_cfg.output_dir)
    self.finalize(train_cfg)

    return {'data': dict(tau_star=self.algorithm.temperature.tau.detach().cpu(), tau_initial=tau_initial.detach().cpu()),
            'calibration_cfg': calibration_cfg,
            'train_cfg': train_cfg,
            'elapsed_time': time.time() - elapsed_time,
            'status': 'done'}

  def finalize(self, cfg):

    if utils.is_not_distributed_or_is_rank0():
      utils.write_trace('calibration.done', dir_=cfg.output_dir)
    utils.message(f'Calibration completed.')

    return


if __name__ == "__main__":
    pass
