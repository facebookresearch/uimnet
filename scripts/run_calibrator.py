#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
from uimnet import utils
from uimnet import algorithms
from uimnet import workers

from omegaconf import OmegaConf
from pathlib import Path

import torch
import numpy as np
import os
import argparse
import pickle
import filelock

CALIBRATION_CFG = """
output_dir: null  # subfolder. Mutable at dispatch

dataset:
  name: ImageNat
  root: /checkpoint/ishmaelb/data/datasets/ILSVRC2012
  equalize_partitions:  True
  batch_size: 256
  seed: 42

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
  dist_protocol: 'env'
  dist_url: null
  # ------
  num_workers: 5


"""
def parse_arguments():

  parser = argparse.ArgumentParser(description='runs prediction state on the model director')
  parser.add_argument('-m', '--model_dir', type=str, required=True)
  parser.add_argument('-c', '--clustering_file', type=str, required=True)
  parser.add_argument('--local_rank', type=int, default=None)
  parser.add_argument('-d', '--distributed', action='store_true')
  parser.add_argument('--dist_protocol', type=str, default='env')

  return parser.parse_args()

def partition_datasets(calibration_cfg, partitions):

  all_datasets = {}
  for split_name in ['train', 'val']:
    all_datasets[split_name] = utils.partition_dataset(name=calibration_cfg.dataset.name,
                                        root=calibration_cfg.dataset.root,
                                        split=split_name,
                                        partitions=partitions,
                                        equalize_partitions=calibration_cfg.dataset.equalize_partitions)

  return all_datasets

@utils.timeit
def run_calibrator(model_dir, clustering_file, args):

  model_path = Path(model_dir)

  # train_cfg = utils.load_cfg(model_path / 'train_cfg.yaml')


  if (model_path / 'train_cfg.yaml').is_file():
    train_cfg = utils.load_cfg(model_path / 'train_cfg.yaml')
  elif (model_path / 'cfg_rank_0.yaml').is_file():
    train_cfg = utils.load_cfg(model_path / 'cfg_rank_0.yaml')
  else:
    err_msg = 'train onfig not found'
    raise ValueError(err_msg)

  OmegaConf.set_struct(train_cfg, True)
  calibration_cfg = OmegaConf.create(CALIBRATION_CFG)
  OmegaConf.set_struct(calibration_cfg, True)
  calibration_cfg.experiment.distributed = args.distributed
  calibration_cfg.experiment.dist_protocol = args.dist_protocol
  if utils.is_distributed():
    os.environ['OMP_NUM_THREADS'] = calibration_cfg.experiment.num_workers

  with filelock.FileLock(clustering_file + '.lock'):
    with open(clustering_file, 'rb') as fp:
      clustering = pickle.load(fp)
  all_datasets = partition_datasets(calibration_cfg, partitions=clustering['partitions'])

  Algorithm = utils.load_model_cls(train_cfg)

  calibrator = workers.Calibrator()
  output = calibrator(calibration_cfg, train_cfg, Algorithm=Algorithm, dataset=all_datasets['train']['in'])
  return output

if __name__ == '__main__':
  args = parse_arguments()
  calibrator_output = run_calibrator(args.model_dir, args.clustering_file, args)
