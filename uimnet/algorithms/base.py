#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
"""

A catalog of algorithms for image classificaton, each of them providing:
    * A constructor that receives the number of classes, training device,
      and random seed to initialize the model, loss, and optimizer.
    * A method `.update(x, y)` that takes a gradient step to minimize the
      prediction error at the labeled minibatch (x, y)
    * A method `.forward(x)` that returns predictions for the unlabeled
      minibatch (x)
    * An optional method `.uncertainty(x)` that provides uncertainty scores
      for the unlabeled minibatch (x)

A class `Ensemble` is provided at the bottom of this file, in order to evaluate
ensembles of trained models in evaluate.py.

"""
import functools
from pathlib import Path
from uimnet.modules.temperature import Temperature
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
import torchvision

import random
import numpy as np

from uimnet import utils
import uimnet.modules.spectral_normalization
import uimnet.modules.gaussian_mixture

import torch.distributed as dist


class Algorithm(torch.nn.Module):
  HPARAMS = {}

  def __init__(self, num_classes, arch, device, seed, use_mixed_precision=False, sn=False, sn_coef=1, sn_bn=False):
    super(Algorithm, self).__init__()
    self.num_classes = num_classes
    self.arch = arch
    self.device = device
    self.use_mixed_precision = use_mixed_precision
    self.grad_scaler = amp.grad_scaler.GradScaler(enabled=use_mixed_precision)

    self.sn = sn
    self.sn_bn = sn_bn
    self.sn_coef = sn_coef

    self.has_native_measure = False

    utils.set_random_seed(seed)
    self.seed = seed

    self.hparams = {}
    if seed == 0:
      self.set_default_hparams_()
    else:
      self.set_random_hparams_()

    self.temperature = Temperature()
    self.networks = torch.nn.ModuleDict()
    self.optimizers = dict()

    # HACK(Ishmael): Get dimension of the feature space form the featurizer
    D = dict(resnet18=512,
             resnet50=2048,
             resnet152=2048)[arch]
    self.gaussian_mixture = uimnet.modules.gaussian_mixture.GaussianMixture(K=num_classes, D=D).cpu()

  def set_temperature(self, mode):
    if not mode in ['initial', 'learned']:
      err_msg = f'Unrecognized temperature mode: {mode}'
      raise ValueError(err_msg)
    self.temperature.mode = mode
    return



  def set_default_hparams_(self):
    for key, val in self.HPARAMS.items():
      self.hparams[key] = val[0]

  def set_random_hparams_(self):
    for key, val in self.HPARAMS.items():
      self.hparams[key] = val[1]()

  def maybe_distribute(self):

    if utils.is_distributed():
      for net_name, net in self.networks.items():
        CLS = torch.nn.parallel.DistributedDataParallel
        self.networks[net_name] = CLS(
            net, device_ids=[self.device], find_unused_parameters=True)

      # Ensuring that all the weights are initialized as in rank 0
        with torch.no_grad():
          for p in self.networks[net_name].parameters():

            p0 = torch.zeros_like(p)
            if dist.get_rank() == 0:
              p0 = p.clone()

            dist.broadcast(p0, src=0)
            p.data.copy_(p0.data)

  def get_l2_reg(self):
    l2_reg = 0.
    if self.hparams["weight_decay"] > 0.:
      for net_name, net in self.networks.items():
        for module in net.modules():
          if hasattr(module, 'weight') and module.weight is not None:
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
              continue
            l2_reg += module.weight.norm(p='fro').pow(2)
    return l2_reg

  def setup_optimizers(self):
    self.lr = self.hparams["lr"]
    for net_name in self.networks.keys():
      self.optimizers[net_name] = torch.optim.SGD(
          self.networks[net_name].parameters(),
          lr=self.lr,
          momentum=self.hparams["momentum"],
          weight_decay=0.)

  def maybe_apply_sn(self):
    if self.sn:
      patch = uimnet.modules.spectral_normalization.utils.monkey_patch_layers
      self.networks['featurizer'].apply(functools.partial(
          patch, sn_coef=self.sn_coef, sn_bn=self.sn_bn))
    return

  def initialize(self, dataset=None):
    # TODO: cleaner way
    if self.__class__.__name__ == "DUE":
        self.networks = torch.nn.ModuleDict(self.construct_networks(dataset))
    else:
        self.networks = torch.nn.ModuleDict(self.construct_networks())
    self.maybe_apply_sn()
    self.to(self.device)
    self.gaussian_mixture = self.gaussian_mixture.cpu()
    self.maybe_distribute()
    self.setup_optimizers()

  def adjust_learning_rate_(self, epoch):
    if len(self.optimizers) == 0:
      err_msg = f'Optimizers not instantiated'
      raise RuntimeError(err_msg)

    lr = self.lr * (0.1**(epoch // 30))
    for net_name, optimizer in self.optimizers.items():
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr

  def construct_networks(self):
    raise NotImplementedError

  def get_state(self):
    return dict(algorithm=self.state_dict(),
                  optimizers={k: v.state_dict() for k, v in self.optimizers.items()})

  def save_state(self, dir_):
    state = self.get_state()

    if utils.is_not_distributed_or_is_rank0():
      with open(Path(dir_) / 'algorithm_state.pkl', 'wb') as fp:
        torch.save(state, fp)
    return

  def load_state(self, dir_, map_location=None, adapt_state=False):

    if map_location is None:
      map_location = 'cpu' #self.device

    with open(Path(dir_) / 'algorithm_state.pkl', 'rb') as fp:
      state = torch.load(fp, map_location=map_location)
    algorithm_state = state['algorithm']

    adapted_algorithm_state = self.state_dict()
    for k, kk in zip(adapted_algorithm_state, algorithm_state):
      adapted_algorithm_state[k] = algorithm_state[kk]

    # if adapt_state:
      # msg = 'Adapting state dict from distributed training to non-distributed evaluation'
      # utils.message(msg)
      # algorithm_state = utils.adapt_state_dict(algorithm_state, self.state_dict())
    # self.load_state_dict(algorithm_state)
    self.load_state_dict(adapted_algorithm_state)
    self.gaussian_mixture = self.gaussian_mixture.cpu()
    for optimizer_name in self.optimizers:
      self.optimizers[optimizer_name].load_state_dict(
          state["optimizers"][optimizer_name]
      )


    state_keys = list(state['algorithm'].keys())
    for i, (key, value) in enumerate(self.state_dict().items()):
      _key = state_keys[i]
      state['algorithm'][_key] = state['algorithm'][_key].to(value.device)

    return state

  def process_minibatch(self, x, y):
    raise NotImplementedError

  def forward(self, x):
    # with amp.autocast(enabled=self.use_mixed_precision):
    return self.temperature(self._forward(x))

  def _forward(self, x):
    raise NotImplementedError

  def get_features(self, x):
    return self.networks['featurizer'](x)



if __name__ == '__main__':
  pass
