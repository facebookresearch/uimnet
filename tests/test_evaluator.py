#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
from uimnet.workers import Calibrator
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

class _TestEvaluator(object):

  arch = 'resnet50'
  device = 'cuda:0'
  nsamples = 256 * 2
  num_classes = 266
  data_shape = (3, 224, 224)

  train_cfg = """
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
  evaluation_cfg = """

  output_dir: null  # subfolder. Mutable at dispatch
  temperature_mode: 'learned'

  measures:
    - Augmentations
    - Gap
    - Jacobian
    - MixtureOfGaussians
    - Entropy
    - Largest
    - Native

  dataset:
    equalize_partitions:  True
    batch_size: 32
    split: 'val'
    seed: 42
    partition: 'in'

  slurm:
    preset: 'distributed_8'  # Key-value pairs below override presets.
    time: 10
    mem_per_gpu: '32G'
    cpus_per_task: 5
    partition: devlab
    array_parallelism: 512
    comment: NeurIPS2021


  experiment:
    distributed: False
    platform: 'slurm'
    #platform: 'debug'
    ## ----- Mutable on the worker during distributed/device setup.
    output_dir: null
    seed: 42  # Workers seed
    device: 'cuda:0'
    rank: null
    local_rank: null
    world_size: null
    dist_protocol: null
    dist_url: null
    # ------
    num_workers: 5

  """
  @classmethod
  def setUpClass(cls):
    Algorithm = __ALGORITHMS__[cls.algorithm_name]

    cls.device = 'cuda:0'
    cls.train_dataset = DebugDataset(nsamples=cls.nsamples,
                                num_classes=cls.num_classes,
                                data_shape=cls.data_shape,
                                )
    dir_name =  f'{cls.__name__}_{str(uuid.uuid4().hex)}'
    cls.output_path = Path(tempfile.gettempdir()).absolute() / dir_name
    cls.output_path.mkdir(parents=True, exist_ok=True)

    train_cfg = OmegaConf.create(cls.train_cfg)
    OmegaConf.set_struct(train_cfg, True)
    train_cfg.output_dir = str(cls.output_path)
    train_cfg.experiment.device = cls.device
    train_cfg.algorithm.name = cls.algorithm_name
    train_cfg.algorithm.arch = cls.arch

    with utils.suppress_stdout():
      cls.trainer = workers.Trainer()
      cls.trainer_output = cls.trainer(train_cfg, Algorithm, cls.train_dataset)

    evaluation_cfg = OmegaConf.create(cls.evaluation_cfg)
    OmegaConf.set_struct(evaluation_cfg, True)
    evaluation_cfg.output_dir = str(cls.output_path)
    evaluation_cfg.experiment.device = cls.device

    with utils.suppress_stdout():
      cls.evaluator = workers.Evaluator()
      cls.eval_dataset = DebugDataset(nsamples=cls.nsamples,
                                  num_classes=cls.num_classes,
                                  data_shape=cls.data_shape,
                                  )
      with torch.no_grad():
        cls.evaluator_output = cls.evaluator(evaluation_cfg, train_cfg, Algorithm=Algorithm,
                                            train_dataset=cls.train_dataset,
                                            val_dataset=cls.eval_dataset)


  def test_trace(self):
    trace_exists = utils.trace_exists('ood_evaluation.done', str(self.output_path))
    self.assertTrue(trace_exists)

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls.output_path)
    del cls.trainer
    del cls.evaluator
    del cls.train_dataset
    del cls.eval_dataset
    if torch.cuda.is_available() and torch.cuda.is_initialized():
      torch.cuda.reset_max_memory_allocated()
    gc.collect()




parameters = [dict(algorithm_name=algorithm_name) for algorithm_name in __ALGORITHMS__]
for Test in utils.parametrize(_TestEvaluator, parameters):
  globals()[Test.__name__] = Test
  del Test





if __name__ == '__main__':
  unittest.main()
