#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
"""
Spectral Normalization from https://arxiv.org/abs/1802.05957
with additional variable `coeff` or max spectral norm.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from uimnet.modules.spectral_normalization.base import SN


# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
  def __init__(self, in_features, out_features, bias=True,
               num_svs=1, num_itrs=1, eps=1e-12, sn_coef=1):
    nn.Linear.__init__(self, in_features, out_features, bias)
    SN.__init__(self, num_svs, num_itrs, out_features, eps=eps, sn_coef=sn_coef)
  def forward(self, x):
    return F.linear(x, self.W_(), self.bias)
if __name__ == '__main__':

  pass
