#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#

from pathlib import Path
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp

import numpy as np

from uimnet import utils
from uimnet import algorithms
from uimnet.modules.temperature import Temperature
from omegaconf import OmegaConf
import torch.distributed as dist


class BaseEnsemble(torch.nn.Module):

  HPARAMS = {}

  def __init__(self, paths, num_classes, arch, device, seed, use_mixed_precision=False, **kwargs):
    super(BaseEnsemble, self).__init__()

    self.paths = paths
    self.num_classes = num_classes
    self.arch = arch
    self.device = device
    self.use_mixed_precision = use_mixed_precision
    self.grad_scaler = amp.grad_scaler.GradScaler(enabled=use_mixed_precision)
    self.temperature = Temperature()

    utils.set_random_seed(seed)
    self.seed = seed

    self.algorithms = torch.nn.ModuleDict({})
    self.optimizers = dict()

    self.gaussian_mixture = None

  def setup_optimizers(self):
    raise NotImplementedError

  def initialize(self, *args, **kwargs):

    self.to(self.device)
    for name, algorithm in self.load_algorithms().items():
      self.algorithms[name] = algorithm

    self.load_algorithms_state()
    self.setup_optimizers()

  def load_algorithms(self):

    # First load model on cpu
    _algorithms = {}
    for path in map(Path, self.paths):
      path = Path(path)
      if not utils.trace_exists('train.done', dir_=str(path)):
        err_msg = f'Model @ {path} didn\'t complete training'
        raise RuntimeError(err_msg)

      filename = 'cfg_rank_0.yaml'
      with open(path/ filename, 'r') as fp:
        train_cfg = OmegaConf.load(fp.name)

      Algorithm = algorithms.__dict__[train_cfg.algorithm.name]
      algorithm = Algorithm(num_classes=self.num_classes,
                            arch=train_cfg.algorithm.arch,
                            device=self.device,
                            use_mixed_precision=train_cfg.algorithm.use_mixed_precision,
                            # Keep data seed fixed
                            seed=train_cfg.algorithm.seed,
                            sn=train_cfg.algorithm.sn,
                            sn_coef=train_cfg.algorithm.sn_coef,
                            sn_bn=train_cfg.algorithm.sn_bn
                            )
      algorithm.initialize()
      _algorithms[str(path)] = algorithm.cpu()

    return _algorithms


  def load_algorithms_state(self, *args, **kwargs):

    map_location = 'cpu'
    for _dir, algorithm in self.algorithms.items():
      filename = 'cfg_rank_0.yaml'
      # with open(Path(_dir)/ filename, 'r') as fp:
      #   train_cfg = OmegaConf.load(fp.name)
      # adapt_state = train_cfg.experiment.distributed and not utils.is_distributed()
      algorithm.load_state(_dir, map_location=map_location)
    return


  def load_state(self, dir_, map_location, adapt_state=False):

    with open(Path(dir_) / 'algorithm_state.pkl', 'rb') as fp:
      states = torch.load(fp, map_location=map_location)
    algorithm_state = states['algorithm']
    adapted_algorithm_state = self.state_dict()
    for k, kk in zip(adapted_algorithm_state, algorithm_state):
      adapted_algorithm_state[k] = algorithm_state[kk]

    # if adapt_state:
      # msg = 'Adapting state dict from distributed training to non-distributed evaluation'
      # utils.message(msg)
      # algorithm_state = utils.adapt_state_dict(algorithm_state, self.state_dict())
    # self.load_state_dict(algorithm_state)
    self.load_state_dict(adapted_algorithm_state)
    return states

  def save_state(self, dir_):
    states = dict(algorithm=self.state_dict())

    if utils.is_not_distributed_or_is_rank0():
      with open(Path(dir_) / 'algorithm_state.pkl', 'wb') as fp:
        torch.save(states, fp)
    return








  # def save_state(self, dir_):
  #   states = dict(algorithm=self.state_dict(),
  #                 optimizers={k: v.state_dict() for k, v in self.optimizers.items()})

  #   if utils.is_not_distributed_or_is_rank0():
  #     with open(Path(dir_) / 'algorithm_state.pkl', 'wb') as fp:
  #       torch.save(states, fp)
  #   return

  # def load_state(self, dir_, map_location, adapt_state=False):

  #   with open(Path(dir_) / 'algorithm_state.pkl', 'rb') as fp:
  #     states = torch.load(fp, map_location=map_location)
  #   algorithm_state = states['algorithm']

  #   adapted_algorithm_state = self.state_dict()
  #   for k, kk in zip(adapted_algorithm_state, algorithm_state):
  #     adapted_algorithm_state[k] = algorithm_state[kk]

  #   # if adapt_state:
  #     # msg = 'Adapting state dict from distributed training to non-distributed evaluation'
  #     # utils.message(msg)
  #     # algorithm_state = utils.adapt_state_dict(algorithm_state, self.state_dict())
  #   # self.load_state_dict(algorithm_state)
  #   self.load_state_dict(adapted_algorithm_state)
  #   for optimizer_name in self.optimizers:
  #     self.optimizers[optimizer_name].load_state_dict(
  #         states["optimizers"][optimizer_name]
  #     )
  #   return states




  def process_minibatch(self, x, y):
    x = x.to(self.device, non_blocking=True)
    y = y.to(self.device, non_blocking=True)
    return x, y

  def forward_members(self, x):

    output = []
    for name, algorithm in self.algorithms.items():
      algorithm = algorithm.cuda()
      output.append(algorithm(x).cpu())
      # algorithm = algorithm.cpu()
    return output

  def forward(self, x):
    return self.temperature(self._forward(x))

  def update(self, x, y):
    raise NotImplementedError

  def set_temperature(self, mode):
    if not mode in ['initial', 'learned']:
      err_msg = f'Unrecognized temperature mode: {mode}'
      raise ValueError(err_msg)
    self.temperature.mode = mode
    return

if __name__ == '__main__':
  pass
