#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
"""
Evaluate in-domain metrics on the sweep directory
"""
import copy
import argparse
import os
import submitit
import pickle
import itertools
import concurrent.futures
import torch

import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
import submitit
import torch.multiprocessing as tmp
import filelock

from uimnet import utils
from uimnet import workers
from uimnet import __SLURM_CONFIGS__
import uimnet.measures
from uimnet import __DEBUG__

EVALUATION_CFG = """
  dataset:
    name: ImageNat
    root: /checkpoint/ishmaelb/data/datasets/ILSVRC2012
    equalize_partitions:  True
    batch_size: 256
    seed: 42

  slurm:
    preset: 'distributed_8'  # Key-value pairs below override presets.
    time: 720
    mem_per_gpu: '32G'
    cpus_per_task: 5
    partition: learnlab
    array_parallelism: 512
    comment: NeurIPS2021

  experiment:
    distributed: True
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
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-s', '--sweep_dir', type=str, required=True)
  parser.add_argument('-f', '--force', action='store_true')
  parser.add_argument('-m', '--measures', type=str, nargs='+', default=['Largest'])
  parser.add_argument('-o', '--output', type=str, default='out_of_domain_table.tex')
  parser.add_argument('--ensembles_only', action='store_true')
  parser.add_argument('--algorithms_only', action='store_true')
  return parser.parse_args()



class ExtractRecords(object):
  def __call__(self, records_filepath):
    with open(records_filepath, 'rb') as fp:
      record = pickle.load(fp)
      return utils.apply_fun(utils.to_scalar, record)

@utils.timeit
def collect_ood_records(models_paths, Measures, max_workers=10):

  records_filepaths = []
  for model_path, Measure in itertools.product(models_paths, Measures):
    trace = f'ood_{Measure.__name__}'
    if utils.trace_exists(f'{trace}.done', dir_=str(model_path)):
      records_filepaths += [model_path / f'{Measure.__name__}_results.pkl']

  with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
     records = list(executor.map(ExtractRecords(), records_filepaths))

  return sum(records, [])

@utils.timeit
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

@utils.timeit
def load_partitions(evaluation_cfg, clustering_file):
    clustering = load_clustering(clustering_file=clustering_file)
    partitions = partition_datasets(evaluation_cfg, partitions=clustering['partitions'])
    return partitions

def maybe_remove_done(models_paths, Measures, force):
  filtered = []
  for model_path, Measure in itertools.product(models_paths, Measures):
    trace = f'ood_{Measure.__name__}'
    if not utils.trace_exists(f'{trace}.done', dir_=str(model_path)) or force:
      filtered += [(model_path, Measure)]
  return filtered


@utils.timeit
def load_clustering(clustering_file):

  clustering_path = Path(clustering_file)
  with filelock.FileLock(str(clustering_path)  + '.lock'):
    with open(clustering_path, 'rb') as fp:
      return pickle.load(fp)

@utils.timeit
def run_evaluation(sweep_dir, force, Measures, ensembles_only, algorithms_only):

  sweep_path = Path(sweep_dir)
  clustering_path = sweep_path / 'clustering.pkl'

  models_paths = [el for el in sweep_path.iterdir() if el.is_dir()]
  models_paths = filter(utils.is_model, models_paths)
  models_paths = filter(utils.train_done, models_paths)
  models_paths = filter(utils.calibration_done, models_paths)

  evaluation_cfg = OmegaConf.create(EVALUATION_CFG)

  if ensembles_only and algorithms_only:
    err_msg = 'only one of ensemble_only or algorithm_only can be selected at a time'
    raise ValueError(err_msg)

  if algorithms_only:
    models_paths = list(filter(utils.is_algorithm, models_paths))

  if ensembles_only:
    models_paths = list(filter(utils.is_ensemble, models_paths))
    evaluation_cfg.slurm.constraint = 'volta32gb'

  executor = utils.get_slurm_executor(copy.deepcopy(evaluation_cfg).slurm,
                                      log_folder=str(sweep_path / 'logs' / 'run_evaluation'))

  # Constructing jobs
  jobs, paths= [], []
  partitions = load_partitions(evaluation_cfg, clustering_file=str(clustering_path))

  with executor.batch():
    for model_path, Measure in itertools.product(models_paths, Measures):
      trace = f'ood_{Measure.__name__}'
      is_done = utils.trace_exists(f'{trace}.done', dir_=str(model_path))
      if is_done and not force:
        print(f'{Measure.__name__} is done. Skipping')
        continue

      if (model_path / 'train_cfg.yaml').is_file():
        train_cfg = utils.load_cfg(model_path / 'train_cfg.yaml')
      elif (model_path / 'cfg_rank_0.yaml').is_file():
        train_cfg = utils.load_cfg(model_path / 'cfg_rank_0.yaml')
      else:
        err_msg = 'train config not found'
        raise ValueError(err_msg)

      Algorithm = utils.load_model_cls(train_cfg)
      worker_args = (
        str(model_path),
        evaluation_cfg,
        train_cfg,
        Algorithm,
        Measure,
        partitions)
      worker = workers.Evaluator()
      job = executor.submit(worker, *worker_args)
      jobs += [job]
      paths += [model_path]
      utils.write_trace(f'{trace}.pending', dir_=str(model_path))

  # Waiting for jobs to finish
  finished_jobs, jobs = utils.handle_jobs(jobs)

  return finished_jobs, jobs



if __name__ == '__main__':
  args = parse_arguments()
  Measures = [uimnet.measures.__MEASURES__[name] for name in args.measures]
  output = run_evaluation(args.sweep_dir, force=args.force, Measures=Measures,
                                    algorithms_only=args.algorithms_only,
                                    ensembles_only=args.ensembles_only
                                    )
