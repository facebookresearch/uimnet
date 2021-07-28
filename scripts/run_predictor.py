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

PREDICTION_CFG = """
output_dir: null  # subfolder. Mutable at dispatch

dataset:
  name: ImageNat
  root: /checkpoint/ishmaelb/data/datasets/ILSVRC2012
  equalize_partitions:  True
  batch_size: 256
  seed: 42

experiment:
  distributed: True
  platform: slurm

  ## ----- Mutable on the worker during distributed/device setup.
  seed: 42  # Workers seed
  device: 'cuda:0'
  rank: null
  local_rank: null
  world_size: null
  dist_protocol: env
  dist_url: null
  num_workers: 5
  # ------

"""



def parse_arguments():

  parser = argparse.ArgumentParser(description='runs prediction state on the model director')
  parser.add_argument('-m', '--model_dir', type=str, required=True)
  parser.add_argument('-c', '--clustering_file', type=str, required=True)
  parser.add_argument('--local_rank', type=int, default=None)
  parser.add_argument('-d', '--distributed', action='store_true')
  parser.add_argument('--dist_protocol', type=str, default='env')

  return parser.parse_args()

def partition_datasets(prediction_cfg, partitions):

  all_datasets = {}
  for split_name in ['train', 'val']:
    all_datasets[split_name] = utils.partition_dataset(name=prediction_cfg.dataset.name,
                                        root=prediction_cfg.dataset.root,
                                        split=split_name,
                                        partitions=partitions,
                                        equalize_partitions=prediction_cfg.dataset.equalize_partitions)

  return all_datasets

# def run_predictor(prediction_cfg, validation_cfg, Algorithm, train_dataset, val_dataset):
#   pass

# def run(model_dir, clustering_file, distributed, ranks=None):
#   if ranks is None:
#     ranks = list(range(dist.get_world_size()))

#   if not distributed:
#     pass

#   processes = []
#   if distributed:
#     for rank in ranks:
#       p = tmp.Process(target=run_predictor, predictor_args)
#       p.start()
#       processes.append(p)
#     for p in processes:
#       p.join()








@utils.timeit
def run_predictor(model_dir, clustering_file, args):

  model_path = Path(model_dir)

  train_cfg = utils.load_cfg(model_path / 'train_cfg.yaml')
  OmegaConf.set_struct(train_cfg, True)
  prediction_cfg = OmegaConf.create(PREDICTION_CFG)
  OmegaConf.set_struct(prediction_cfg, True)
  prediction_cfg.experiment.distributed = args.distributed
  prediction_cfg.experiment.dist_protocol = args.dist_protocol

  if utils.is_distributed():
    os.environ['OMP_NUM_THREADS'] = prediction_cfg.experiment.num_workers

  with filelock.FileLock(clustering_file + '.lock'):
    with open(clustering_file, 'rb') as fp:
      clustering = pickle.load(fp)
  all_datasets = partition_datasets(prediction_cfg, partitions=clustering['partitions'])

  Algorithm = utils.load_model_cls(train_cfg)

  predictor = workers.Predictor()
  output = predictor(prediction_cfg, train_cfg, Algorithm=Algorithm,
                     train_dataset=all_datasets['train']['in'],
                     val_dataset=all_datasets['val']['in'])

  return output

if __name__ == '__main__':
  args = parse_arguments()
  predictor_output = run_predictor(args.model_dir, args.clustering_file, args)
