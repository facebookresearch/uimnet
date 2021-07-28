#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch

class Measure(torch.nn.Module):
  def __init__(self, algorithm):
    super(Measure, self).__init__()
    self.algorithm = algorithm

  def estimate(self, *args, **kwargs):
    pass

  def forward(self, x):
    raise NotImplementedError
if __name__ == '__main__':
  pass
