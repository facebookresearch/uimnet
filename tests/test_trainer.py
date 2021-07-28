#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
from uimnet.algorithms.base import Algorithm
import uuid
from pathlib import Path
import shutil
import tempfile
import gc
import torch

import unittest
from uimnet import workers
from uimnet import utils
from uimnet.algorithms import __ALGORITHMS__
from omegaconf import OmegaConf
from uimnet.datasets.debug import DebugDataset

class _TestTrainer(object):
  arch = 'resnet50'
  device = 'cuda:0'
  nsamples = 256 * 2
  num_classes = 266
  data_shape = (3, 224, 224)

  cfg = """
  sweep_dir: null
  output_dir: null  # subfolder. Mutable at dispatch

  algorithm:
    name: null
    arch: null
    use_mixed_precision: False
    seed: 0 # Mutable at dispatch
    sn: False
    sn_coef: 1.0
    sn_bn: True

  dataset:
    seed: 0
    batch_size: 256
    splits_props:
      train: 0.9
      eval: 0.1

  slurm:
    preset: 'distributed_8'  # Key-value pairs below override presets.
    time: 60
    mem_per_gpu: '32G'
    cpus_per_task: 5
    partition: learnfair
    array_parallelism: 512

  experiment:
    distributed: False
    platform: slurm
    evaluate_every: 1
    checkpoint_every: 1
    num_epochs: 1
    ## ----- Mutable on the worker during distributed/device setup.
    output_dir: null
    seed: 42  # Workers seed
    device: 'cuda:0'
    rank: null
    local_rank: null
    world_size: null
    dist_protocol: null
    dist_url: null
    num_workers: 5
    # ------

  """
  @classmethod
  def setUpClass(cls):
    Algorithm = __ALGORITHMS__[cls.algorithm_name]

    cls.device = 'cuda:0'
    cls.dataset = DebugDataset(nsamples=cls.nsamples,
                                num_classes=cls.num_classes,
                                data_shape=cls.data_shape,
                                )
    dir_name =  f'{cls.__name__}_{str(uuid.uuid4().hex)}'
    cls.output_path = Path(tempfile.gettempdir()).absolute() / dir_name
    cls.output_path.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.create(cls.cfg)
    OmegaConf.set_struct(cfg, True)
    cfg.output_dir = str(cls.output_path)
    cfg.experiment.device = cls.device
    cfg.algorithm.name = cls.algorithm_name
    cfg.algorithm.arch = cls.arch

    cls.trainer = workers.Trainer()
    cls.trainer_output = cls.trainer(cfg, Algorithm, cls.dataset)

  def test_done(self):

    actual = self.trainer_output['status']
    expected = 'done'
    self.assertEqual(actual, expected)

  def test_trace(self):
    trace_exists = utils.trace_exists('train.done', str(self.output_path))
    self.assertTrue(trace_exists)

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls.output_path)
    del cls.trainer
    del cls.dataset
    if torch.cuda.is_available() and torch.cuda.is_initialized():
      torch.cuda.reset_max_memory_allocated()
    gc.collect()





parameters = [dict(algorithm_name=algorithm_name) for algorithm_name in __ALGORITHMS__]
for Test in utils.parametrize(_TestTrainer, parameters):
  globals()[Test.__name__] = Test
  del Test





if __name__ == '__main__':
  unittest.main()
