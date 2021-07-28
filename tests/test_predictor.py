#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import uuid
from pathlib import Path
import shutil
import tempfile
import torch
from uimnet import workers
from uimnet import utils
from uimnet.algorithms import __ALGORITHMS__
from omegaconf import OmegaConf
from uimnet.datasets.debug import DebugDataset
import pytest

@pytest.fixture(params=list(__ALGORITHMS__.keys()))
def test_trainer(request):

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
    batch_size: 32
    splits_props:
      train: 0.9
      eval: 0.1

  experiment:
    evaluate_every: 1
    checkpoint_every: 1
    num_epochs: 1
    seed: 42  # Workers seed
    distributed: False
    platform: slurm
    ## ----- Mutable on the worker during distributed/device setup.
    output_dir: null
    device: 'cuda:0'
    rank: null
    local_rank: null
    world_size: null
    dist_protocol: null
    dist_url: null
    num_workers: 5
    # ------

  """

  train_cfg = OmegaConf.create(train_cfg)
  OmegaConf.set_struct(train_cfg, True)
  device = 'cuda:0'
  arch = 'resnet18'
  # algorithm_name = 'ERM'
  algorithm_name = request.param

  nsamples = 32 * 2
  num_classes = 266
  data_shape = (3,  224, 224)

  train_dataset = DebugDataset(nsamples=nsamples,
                              num_classes=num_classes,
                              data_shape=data_shape,
                              )


  dir_name =  f'{str(uuid.uuid4().hex)}'
  output_path = Path(tempfile.gettempdir()).absolute() / dir_name
  output_path.mkdir(parents=True, exist_ok=True)

  train_cfg.experiment.device = device
  train_cfg.algorithm.name = algorithm_name
  train_cfg.algorithm.arch = arch
  train_cfg.output_dir = str(output_path)

  Algorithm = __ALGORITHMS__[algorithm_name]

  with utils.suppress_stdout():
    trainer = workers.Trainer()
    trainer_output = trainer(train_cfg, Algorithm, train_dataset)

  assert trainer_output['status'] == 'done'
  assert utils.trace_exists('train.done', str(output_path))

  yield train_cfg, Algorithm, train_dataset

  shutil.rmtree(output_path)

def test_predictor(test_trainer):

  prediction_cfg = """
  output_dir: null  # subfolder. Mutable at dispatch

  dataset:
    seed: 0
    batch_size: 32

  experiment:
    distributed: False
    platform: slurm
    evaluate_every: 1
    checkpoint_every: 1
    num_epochs: 1

    ## ----- Mutable on the worker during distributed/device setup.
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

  prediction_cfg = OmegaConf.create(prediction_cfg)
  OmegaConf.set_struct(prediction_cfg, True)
  prediction_cfg.experiment.device = 'cuda:0'

  train_cfg, Algorithm, train_dataset = test_trainer
  nsamples = 32 * 2
  num_classes = 266
  data_shape = (3,  24, 224)

  val_dataset = DebugDataset(nsamples=nsamples,
                             num_classes=num_classes,
                             data_shape=data_shape)

  with utils.suppress_stdout():
    predictor = workers.Predictor()
    predictor_output = predictor(prediction_cfg, train_cfg, Algorithm, train_dataset, val_dataset)

  assert predictor_output['status'] == 'done'
  assert utils.trace_exists('prediction.done', train_cfg.output_dir)
