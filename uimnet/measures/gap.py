#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch
import uimnet
from uimnet.measures.measure import Measure


class Gap(Measure):
  """
  Minus the gap between two largest softmax scores
  """

  def __init__(self, algorithm):
    super(Gap, self).__init__(algorithm=algorithm)

  def forward(self, x):
    top2 = self.algorithm(x).softmax(1).topk(2, dim=1).values
    return (top2[:, 0] - top2[:, 1]).mul(-1)


if __name__ == '__main__':
  pass
