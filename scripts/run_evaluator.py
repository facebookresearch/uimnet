#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
from uimnet import utils
from uimnet import algorithms
from uimnet import workers
from uimnet import measures

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

EVALUATION_CFG = """

  dataset:
    name: ImageNat
    root: /checkpoint/ishmaelb/data/datasets/ILSVRC2012
    equalize_partitions:  True
    batch_size: 32
    seed: 42

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


def parse_arguments():

  parser = argparse.ArgumentParser(description='runs evaluation state on the model director')
  parser.add_argument('-m', '--model_dir', type=str, required=True)
  parser.add_argument('-c', '--clustering_file', type=str, required=True)
  parser.add_argument('--local_rank', type=int, default=None)
  parser.add_argument('-d', '--distributed', action='store_true')
  #parser.add_argument('-m', '--measures', type=str, nargs='+', default=['Largest'])
  parser.add_argument('--measure', type=str, default='Largest')
  parser.add_argument('--dist_protocol', type=str, default='env')

  return parser.parse_args()

def partition_datasets(evaluation_cfg, partitions):

  all_partitions = {}
  for split_name in ['train', 'val']:
    split_partition  = utils.partition_dataset(name=evaluation_cfg.dataset.name,
                                               root=evaluation_cfg.dataset.root,
                                               split=split_name,
                                               partitions=partitions,
                                               equalize_partitions=evaluation_cfg.dataset.equalize_partitions)

    for partition_name, partition in split_partition.items():
      all_partitions[(split_name, partition_name)]  = partition

  return all_partitions

def load_clustering(clustering_file):
  clustering_path = Path(clustering_file)
  with filelock.FileLock(str(clustering_path)  + '.lock'):
    with open(clustering_path, 'rb') as fp:
      return pickle.load(fp)

@utils.timeit
def run_evaluator(model_dir, clustering_file, measure_name, distributed, dist_protocol):

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
  evaluation_cfg = OmegaConf.create(EVALUATION_CFG)
  OmegaConf.set_struct(evaluation_cfg, True)
  evaluation_cfg.experiment.distributed = distributed
  evaluation_cfg.experiment.dist_protocol = dist_protocol

  if utils.is_distributed():
    os.environ['OMP_NUM_THREADS'] = evaluation_cfg.experiment.num_workers

  clustering = load_clustering(clustering_file)
  partitions = partition_datasets(evaluation_cfg, partitions=clustering['partitions'])
  Algorithm = utils.load_model_cls(train_cfg)
  Measure = measures.__MEASURES__[measure_name]
  worker =  workers.Evaluator()
  worker_args = (model_dir, evaluation_cfg, train_cfg, Algorithm, Measure, partitions)

  output = worker(*worker_args)
  return output

if __name__ == '__main__':
  args = parse_arguments()
  evaluator_output = run_evaluator(args.model_dir, args.clustering_file,
                                   measure_name=args.measure,
                                   distributed=args.distributed,
                                   dist_protocol=args.dist_protocol,
                                   )
