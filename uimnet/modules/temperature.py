#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#

#
import collections
import torch
import torch.nn as nn

class Temperature(nn.Module):
  def __init__(self, tau_0=1.):
    super(Temperature, self).__init__()
    self.register_buffer('tau0', torch.as_tensor([tau_0]))
    self.tau = nn.parameter.Parameter(torch.as_tensor([tau_0]), requires_grad=False)
    self.mode = 'initial'

  def reinitialize_temperature(self):
    with torch.no_grad():
      self.tau.copy_(self.tau0)

  def forward(self, l):
    if not self.mode in ['initial', 'learned']:
      err_msg = f'Unrecognized temperature mode: {self.mode}'
      raise ValueError(err_msg)

    if self.mode == 'initial':
      return l / self.tau0
    else:
      return l / self.tau


if __name__ == '__main__':
  pass
