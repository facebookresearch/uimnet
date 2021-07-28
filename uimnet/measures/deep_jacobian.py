#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import numpy as np
import torch
from uimnet import algorithms
from uimnet.measures.measure import Measure


class DeepJacobian(Measure):
  """
    Gradient norm (norm of Jacobian)
    """

  def __init__(self, algorithm):
    super(DeepJacobian, self).__init__(algorithm=algorithm)

    name = algorithm.__class__.__name__
    self.invalid = False
    if name in ["MCDropout", "Ensemble"]:
      self.invalid = True

  def forward(self, x):
    if self.invalid:
      return x.new_zeros((x.shape[0], )).fill_(np.nan)

    x = x.clone().detach()
    featurizer = self.algorithm.networks['featurizer']
    classifier = self.algorithm.networks['classifier']
    z = featurizer(x).detach()

    with torch.enable_grad():
      z.requires_grad = True
      g = torch.autograd.grad(classifier(z).sum(), z, create_graph=True)[0]

    return g.view(len(x), -1).pow(2).mean(1).detach()
if __name__ == '__main__':
  pass
