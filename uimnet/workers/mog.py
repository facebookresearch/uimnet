#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
from ast import dump
import copy
import multiprocessing as mp
import argparse
import pickle
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
import torch.nn as nn
import submitit
from omegaconf import OmegaConf

from uimnet.algorithms.base import Algorithm
from uimnet import algorithms
from uimnet import datasets
from uimnet import metrics
from uimnet import utils
from uimnet import workers
from uimnet import __DEBUG__


from uimnet.modules.gaussian_mixture import GaussianMixture

class MOG(workers.Worker):
  def __init__(self):
    super(MOG, self).__init__()
    self.mog_cfg = None
    self.train_cfg = None
    self.Algorithm = None
    self.dataset = None

  def __call__(self, mog_cfg, train_cfg, Algorithm, dataset):
    elapsed_time = time.time()
    self.mog_cfg = mog_cfg
    self.train_cfg = train_cfg
    self.Algorithm = Algorithm
    self.dataset = dataset

    self.setup(mog_cfg)
    utils.message(mog_cfg)

    if utils.is_not_distributed_or_is_rank0():
      if not utils.trace_exists('train.done', dir_=train_cfg.output_dir):
        err_msg = f'Training not finished'
        raise RuntimeError(err_msg)
      utils.write_trace('mog.running', dir_=train_cfg.output_dir)

    utils.message('Instantiating data node')
    # Training will be done either on the validation set or the training set
    self.datanode = datasets.SplitDataNode(
      dataset=dataset,
      transforms=datasets.TRANSFORMS,
      splits_props=train_cfg.dataset.splits_props,
      seed=train_cfg.dataset.seed)
    num_classes = self.datanode.splits['train'].num_classes

    utils.message('Instantiating algorithm')

    self.algorithm = Algorithm(num_classes=num_classes,
                               arch=train_cfg.algorithm.arch,
                               device=mog_cfg.experiment.device,
                               use_mixed_precision=train_cfg.algorithm.use_mixed_precision,
                               seed=train_cfg.algorithm.seed,
                               sn=train_cfg.algorithm.sn,
                               sn_coef=train_cfg.algorithm.sn_coef,
                               sn_bn=train_cfg.algorithm.sn_bn)

    self.algorithm.initialize()
    utils.message(self.algorithm)
    self.algorithm.load_state(train_cfg.output_dir, map_location=mog_cfg.experiment.device)
    utils.message(f'Algorithm gaussian mixture attribute {self.algorithm.gaussian_mixture}')

    # Preparing for mixture of gaussian estimation
    self.datanode.eval()
    self.algorithm.eval()
    eval_loader = self.datanode.get_loader('train',
                                            batch_size=mog_cfg.dataset.batch_size,
                                            shuffle=False,
                                            pin_memory=True if 'cuda' in mog_cfg.experiment.device else False,
                                            num_workers=mog_cfg.experiment.num_workers)

    collected = collections.defaultdict(list)
    utils.message('Collecting logits and features')
    with torch.no_grad():
      for i, batch in enumerate(eval_loader):
        batch = utils.apply_fun(functools.partial(utils.to_device, device=mog_cfg.experiment.device), batch)
        x, y = batch['x'], batch['y']
        collected['features'] += [self.algorithm.get_features(x).cpu()]
        collected['y'] += [y]
        #if __DEBUG__ and i > 2:
          #break
    collected = dict(collected)

    # Concatenating locally and acorss workers
    utils.message('Concatenating accross workers')
    #collected = {k: utils.all_cat(v, dim=0) for k, v in collected.items()}
    collected = {k: torch.cat(v, dim=0) for k, v in collected.items()}

    all_classes = collected['y'].unique()
    utils.message(f'{type(self.algorithm)}:{len(all_classes)}, {num_classes}')
    assert len(all_classes) == num_classes

    #gaussian_mixture = GaussianMixture(K=num_classes, D=collected['features'].shape[1])
    for y in all_classes:
      mask = torch.where(y == collected['y'])
      self.algorithm.gaussian_mixture.add_gaussian_from_data(collected['features'][mask], y,eps=mog_cfg.eps)

    utils.message('Serialiazing estimated mixture of gaussians.')
    if utils.is_not_distributed_or_is_rank0():
      #del self.algorithm.gaussian_mixture
      #self.algorithm.add_module('gaussian_mixture', gaussian_mixture)
      self.algorithm.save_state(train_cfg.output_dir)
      # mog_path = Path(train_cfg.output_dir) / 'mog.pkl'
      # with open(mog_path, 'wb') as fp:
      #   pickle.dump(gaussian_mixture, fp, protocol=pickle.HIGHEST_PROTOCOL)
      utils.write_trace('mog.done', dir_=train_cfg.output_dir)

    utils.message('Mixture of gaussians estimation completed.')

    return {'data': None, #self.algorithm.gaussian_mixture,
            'mog_cfg': mog_cfg,
            'train_cfg': train_cfg,
            'elapsed_time': time.time() - elapsed_time,
            'status': 'done'}



if __name__ == '__main__':
  pass
