#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
"""
Evaluate in-domain metrics on the sweep directory
"""
import argparse
import os
import submitit
import pickle
import concurrent.futures
import torch
import copy
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
import submitit
import torch.multiprocessing as tmp

from uimnet import utils
from uimnet import workers
from uimnet import __SLURM_CONFIGS__


PREDICTION_CFG = """
output_dir: null  # subfolder. Mutable at dispatch

dataset:
  name: ImageNat
  root: /checkpoint/ishmaelb/data/datasets/ILSVRC2012
  equalize_partitions:  True
  batch_size: 256
  seed: 42


slurm:
  preset: 'distributed_8'  # Key-value pairs below override presets.
  time: 180
  mem_per_gpu: '32G'
  cpus_per_task: 5
  partition: learnfair
  array_parallelism: 512
  constraint: volta32gb

experiment:
  distributed: True
  platform: slurm

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

def parse_arguments():
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-s', '--sweep_dir', type=str, required=True)
  parser.add_argument('-f', '--force', action='store_true')
  parser.add_argument('-o', '--output', type=str, default='indomain_table.tex')
  return parser.parse_args()

@utils.timeit
def load_datasets(root, name, clustering_path):

  with open(Path(clustering_path), 'rb') as fp:
    clustering = pickle.load(fp)

  datasets = {}
  for split in ['train', 'val']:
    datasets[split] = utils.partition_dataset(name=name,
                                      root=root,
                                      split=split,
                                      partitions=clustering['partitions'],
                                      equalize_partitions=True)
  return datasets

class ExtractRecords(object):
  def __call__(self, model_path):
    with open(model_path / 'predictive_records.pkl', 'rb') as fp:
      record = pickle.load(fp)
      return utils.apply_fun(utils.to_scalar, record)

@utils.timeit
def get_indomain_records(models_paths):
  models_paths = list(models_paths)
  #max_workers = max(2, tmp.cpu_count() - 2)
  max_workers = 10
  # test = ExtractRecords()(model_path=models_paths[0])
  with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    all_records = list(executor.map(ExtractRecords(), models_paths))
  return sum(all_records, [])


def make_indomain_dataframe(evaluation_records: list):
  """
  Generates In-Domain table
  -------------------------
  """
  # Generating the predictive results dataframe
  df = pd.DataFrame.from_records(evaluation_records).round(4)

  group_keys = ['algorithm.arch', 'dataset.name', 'algorithm.name', 'algorithm.sn','temperature_mode', 'split']
  val_keys = ['ACC@1', 'ACC@5', 'NLL', 'ECE']

  group_df = df.groupby(group_keys)[val_keys]
  mean_df, std_df, count_df = [el.round(4) for el in [group_df.mean(), group_df.std(), group_df.count()]]

  final_df = utils.make_error_bars_df_with_count(mean_df, std_df, count_df)

  return final_df


def run_prediction(sweep_dir, force):

  sweep_path = Path(sweep_dir)
  clustering_path = sweep_path / 'clustering.pkl'

  models_paths = filter(utils.is_model, sweep_path.iterdir())
  models_paths = list(filter(utils.train_done, models_paths))

  prediction_cfg = OmegaConf.create(PREDICTION_CFG)

  root = os.getenv('DATASETS_ROOT')
  name = 'ImageNat'

  datasets = load_datasets(root=root,
                          name=name,
                          clustering_path=clustering_path
                          )
  executor = utils.get_slurm_executor(copy.deepcopy(prediction_cfg.slurm),
                                      log_folder=str(sweep_path / 'logs' / 'run_prediction'))
  # Constructing jobs
  jobs, paths = [], []
  with executor.batch():
    # Construcing jobs
    for model_path in models_paths:
      if utils.prediction_done(model_path) and not force:
        print(f'{model_path} is done. Skipping.')
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
        prediction_cfg,
        train_cfg,
        Algorithm,
        datasets['train']['in'],
        datasets['val']['in'])
      worker = workers.Predictor()
      job = executor.submit(worker, *worker_args)
      jobs += [job]
      paths += [model_path]
      utils.write_trace('prediction.pending', dir_=str(model_path))

  beholder = utils.Beholder(list(zip(jobs, paths)), stem='prediction')
  beholder.start()
  finished_jobs, jobs = utils.handle_jobs(jobs)
  # Collecting results
  jobs_results = [job.results() for job in finished_jobs]
  return jobs_results



if __name__ == '__main__':
  args = parse_arguments()
  output = run_prediction(args.sweep_dir, force=args.force)
