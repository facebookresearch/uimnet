#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import sys
import shutil
from pathlib import Path
from typing import Dict
import gc
from uimnet.algorithms.base import Algorithm
import unittest
import torch
import torch.multiprocessing as tmp
from parameterized import parameterized_class
import uimnet
import uuid
import tempfile
from uimnet import utils
from uimnet.algorithms import __ALGORITHMS__
import pytest



def sample_batch(batch_size, num_classes):
  batch = dict(
    index=torch.arange(batch_size),
    x=torch.ones(batch_size, 3, 224, 224).normal_(),
    y=torch.randint(num_classes, size=(batch_size, )).long())
  return batch

@pytest.fixture(params=list(__ALGORITHMS__.keys()))
def algorithm_batch(request):

  batch_size = 32
  device = 'cuda:0'
  num_classes = 266
  arch='resnet50'

  algorithm_kwargs = dict(num_classes=num_classes,
                          arch=arch,
                          device=device,
                          seed=0,
                          use_mixed_precision=False,
                          sn=False,
                          sn_coef=1.,
                          sn_bn=False
                          )
  algorithm_name = request.param
  algorithm = __ALGORITHMS__[algorithm_name](**algorithm_kwargs)
  algorithm.initialize()
  dir_name =  f'{algorithm.__class__.__name__}_{str(uuid.uuid4().hex)}'
  output_path = Path(tempfile.gettempdir()).absolute() / dir_name
  output_path.mkdir(parents=True, exist_ok=True)
  batch = sample_batch(batch_size, num_classes)
  yield algorithm, batch
  # cleanup
  batch = sample_batch(batch_size, num_classes)
  batch = utils.map_device(batch, device)
  shutil.rmtree(output_path)
  return


def test_forward_shape(algorithm_batch):
  algorithm, batch = algorithm_batch
  with torch.no_grad():
    logits = algorithm(batch['x'])
  actual = logits.shape
  expected = (len(batch['x']), algorithm.num_classes)
  assert actual == expected

def test_update(algorithm_batch):
  algorithm, batch = algorithm_batch

  out = algorithm.update(batch['x'], batch['y'])
  # assert torch.allclose(out.sum(dim=1), torch.ones_like(out[:, 0]))
  assert (type(out['loss']) == type(out['cost']) == float)

if __name__ == '__main__':
  pass
