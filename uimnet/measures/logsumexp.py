#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch
from uimnet.measures.measure import Measure


class LogSumExp(Measure):
  """
  Soft maximum
  """
  def __init__(self, algorithm):
    super(LogSumExp, self).__init__(algorithm=algorithm)

  def forward(self, x):
    return torch.logsumexp(self.algorithm(x), dim=1).mul(-1)


if __name__ == '__main__':
  pass
