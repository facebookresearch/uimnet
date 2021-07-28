#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
from typing import Dict
import gc
import unittest
import torch
import torch.multiprocessing as tmp
from uimnet import utils

from uimnet.measures import __MEASURES__
from uimnet.modules.gaussian_mixture import GaussianMixture

import torch.nn as nn

def sample_batch(batch_size, num_classes):
  batch = dict(
    index=torch.arange(batch_size),
    x=torch.ones(batch_size, 3, 224, 224).normal_(),
    y=torch.randint(num_classes, size=(batch_size, )).long())
  return batch

class DummyAlgorithm(nn.Module):
  def __init__(self, num_classes, device):
    super(DummyAlgorithm, self).__init__()
    self.num_classes =num_classes
    self.num_features = 512
    self.device = device
    # Keep the gaussian mixture on the cpu
    self.gaussian_mixture = GaussianMixture(self.num_classes, self.num_features).cpu()

  def get_features(self, x):
    return x.new_zeros(size=(x.shape[0], self.num_features)).normal_()

  def forward(self, x):
    return x.contiguous().view(x.shape[0], -1)[:, :self.num_classes].normal_()

  def initialize(self):
    self.to(self.device)
    self.gaussian_mixture.cpu()



class _TestMeasure(object):

  batch_size = 32
  device = 'cuda:0'
  num_classes = 266

  @classmethod
  def setUpClass(cls):

    algorithm = DummyAlgorithm(cls.num_classes, cls.device)
    algorithm.initialize()

    cls.measure = __MEASURES__[cls.measure_name](algorithm)
    batch = sample_batch(cls.batch_size, cls.num_classes)
    cls.batch = utils.map_device(batch, cls.device)

  def test_measurement_shape(self):

    with torch.no_grad():
      measurement = self.measure(self.batch['x'])

    actual = measurement.shape
    expected = (self.batch_size, )
    self.assertEqual(actual, expected)


  @classmethod
  def tearDownClass(cls):
    del cls.measure
    del cls.batch

    if torch.cuda.is_available() and torch.cuda.is_initialized():
      torch.cuda.reset_max_memory_allocated()
    gc.collect()

parameters = [
  {'measure_name': name} for name in __MEASURES__
]

for Test in utils.parametrize(_TestMeasure, parameters):
  globals()[Test.__name__] = Test
  del Test







if __name__ == '__main__':
  unittest.main()
