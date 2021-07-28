#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch
import numpy as np
import uimnet
from uimnet.measures.measure import Measure

class Native(Measure):
  """
  Native uncertanity measure
  """

  def __init__(self, algorithm):
    super(Native, self).__init__(algorithm=algorithm)

  def forward(self, x):
    if hasattr(self.algorithm, 'uncertainty'):
      return self.algorithm.uncertainty(x)
    else:
      return x.new_zeros((x.shape[0], )).fill_(np.nan)


if __name__ == '__main__':
  pass
