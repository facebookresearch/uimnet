#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch
import torchvision

from uimnet import utils
from uimnet.measures.measure import Measure


class Augmentations(Measure):
  """
  Minus the largest softmax score
  """

  def __init__(self, algorithm):
    super(Augmentations, self).__init__(algorithm=algorithm)
    self.transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        torchvision.transforms.RandomGrayscale()])
    self.m = 10

  def forward(self, x):
      x = x.to(self.algorithm.device)

      predictions = 0.
      for _ in range(self.m):
        xtilde = self.transform(x)
        predictions = predictions + self.algorithm(xtilde).softmax(1)
      predictions = predictions / self.m

      return predictions.max(1).values.mul(-1)

if __name__ == '__main__':
  pass
