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
import torch.distributed as dist
import torch.multiprocessing as tmp
import numpy as np
import os
import argparse
import pickle
import filelock

TRAIN_CFG = """

  sweep_dir: null
  output_dir: null  # subfolder. Mutable at dispatch

  dataset:
    name: ImageNat
    root: /checkpoint/ishmaelb/data/datasets/ILSVRC2012
    equalize_partitions: True
    seed: 0
    batch_size: 256
    splits_props:
      train: 0.9
      eval: 0.1

  algorithm:
    name: null
    arch: null
    use_mixed_precision: True
    seed: 0 # Mutable at dispatch
    sn: False
    sn_coef: 1.0
    sn_bn: True

  experiment:
    distributed: False
    platform: local
    evaluate_every: 10
    checkpoint_every: 10
    num_epochs: 100

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



def parse_arguments():

  parser = argparse.ArgumentParser(description='Trains model')
  parser.add_argument('-a', '--algorithm', type=str, required=True)
  parser.add_argument('--arch', type=str, default='resnet18')
  parser.add_argument('-m', '--model_dir', type=str, required=True)
  parser.add_argument('-c', '--clustering_file', type=str, required=True)
  parser.add_argument('--local_rank', type=int, default=None)
  parser.add_argument('-d', '--distributed', action='store_true')
  parser.add_argument('--dist_protocol', type=str, default='env')

  return parser.parse_args()

def partition_datasets(train_cfg, partitions):

  all_datasets = {}
  for split_name in ['train', 'val']:
    all_datasets[split_name] = utils.partition_dataset(name=train_cfg.dataset.name,
                                        root=train_cfg.dataset.root,
                                        split=split_name,
                                        partitions=partitions,
                                        equalize_partitions=train_cfg.dataset.equalize_partitions)

  return all_datasets

def train_algorithm(train_cfg, Algorithm, dataset):

  if utils.is_distributed():
    os.environ['OMP_NUM_THREADS'] = train_cfg.experiment.num_workers

  trainer = workers.Trainer()
  output = trainer(train_cfg, Algorithm, dataset=dataset)
  return output

@utils.timeit
def run_trainer(model_dir, algorithm_name, arch, clustering_file, distributed, dist_protocol):

  model_path = Path(model_dir)
  model_path.mkdir(parents=True, exist_ok=True)

  train_cfg = OmegaConf.create(TRAIN_CFG)
  OmegaConf.set_struct(train_cfg, True)

  train_cfg.output_dir =model_dir
  train_cfg.algorithm.name = algorithm_name
  train_cfg.algorithm.arch = arch
  train_cfg.experiment.distributed = distributed
  train_cfg.experiment.dist_protocol = dist_protocol

  with open(model_path / 'train_cfg.yaml', 'w') as fp:
    OmegaConf.save(train_cfg, fp.name)

  with filelock.FileLock(clustering_file + '.lock'):
    with open(clustering_file, 'rb') as fp:
      clustering = pickle.load(fp)

  datasets = partition_datasets(train_cfg, partitions=clustering['partitions'])
  Algorithm = utils.load_model_cls(train_cfg)

  trainer_args = (train_cfg, Algorithm, datasets['train']['in'])
  output = train_algorithm(*trainer_args)
  return utils.pack(output)

if __name__ == '__main__':
  args = parse_arguments()
  trainer_output = run_trainer(args.model_dir, args.algorithm, arch=args.arch, clustering_file=args.clustering_file,
                               distributed=args.distributed, dist_protocol=args.dist_protocol)
