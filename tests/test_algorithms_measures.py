#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
from typing import Dict
import gc
from uimnet.algorithms.base import Algorithm
import unittest
from numpy.testing._private.utils import measure
import torch
import torch.multiprocessing as tmp
from parameterized import parameterized_class, parameterized
from uimnet import utils
import itertools

from uimnet.algorithms import __ALGORITHMS__
from uimnet.measures import __MEASURES__

def sample_batch(batch_size, num_classes):
  batch = dict(
    index=torch.arange(batch_size),
    x=torch.ones(batch_size, 3, 224, 224).normal_(),
    y=torch.randint(num_classes, size=(batch_size, )).long())
  return batch

class _TestAlgorithmMeasure(object):

  batch_size = 32
  device = 'cuda:0'
  num_classes = 266
  arch='resnet50'

  @classmethod
  def setUpClass(self):

    algorithm_kwargs = dict(num_classes=self.num_classes,
                            arch=self.arch,
                            device=self.device,
                            seed=0,
                            use_mixed_precision=False,
                            sn=False,
                            sn_coef=1.,
                            sn_bn=False
                            )

    self.algorithm = __ALGORITHMS__[self.algorithm_name](**algorithm_kwargs)
    self.algorithm.initialize()

    batch = sample_batch(self.batch_size, self.num_classes)
    self.batch = utils.map_device(batch, self.device)


  def test_measurement_shape(self):
    for name, Measure in __MEASURES__.items():
      with self.subTest(msg=f'{name}'):
        measure = Measure(self.algorithm)
        with torch.no_grad():
          measurement = measure(self.batch['x'])
        actual = measurement.shape
        expected = (self.batch_size, )
        self.assertEqual(actual, expected)
        del measure


  @classmethod
  def tearDownClass(cls):
    del cls.algorithm
    del cls.batch

    if torch.cuda.is_available() and torch.cuda.is_initialized():
      torch.cuda.reset_max_memory_allocated()
    gc.collect()



parameters = [dict(algorithm_name=algorithm_name) for algorithm_name in __ALGORITHMS__]
for Test in utils.parametrize(_TestAlgorithmMeasure, parameters):
  globals()[Test.__name__] = Test
  del Test

if __name__ == '__main__':
  unittest.main()
