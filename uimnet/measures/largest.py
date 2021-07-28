#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch
import uimnet
from uimnet.measures.measure import Measure


class Largest(Measure):
  """
  Minus the largest softmax score
  """

  def __init__(self, algorithm):
    super(Largest, self).__init__(algorithm=algorithm)

  def forward(self, x):
    return self.algorithm(x).softmax(1).max(1).values.mul(-1)


if __name__ == '__main__':
  pass
