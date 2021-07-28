#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch
import uimnet
from uimnet.measures.measure import Measure


class Entropy(Measure):
  """
    Entropy
    """

  def __init__(self, algorithm):
    super(Entropy, self).__init__(algorithm=algorithm)

  def forward(self, x):
    sm = self.algorithm(x).softmax(1)
    zeros = torch.zeros(()).to(sm.device)
    h = sm * torch.where(sm == 0, zeros, torch.log(sm))
    return h.sum(1).mul(-1)


if __name__ == '__main__':
  pass
