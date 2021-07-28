#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
from typing import List, Dict
import os
import copy
from numpy.core.multiarray import concatenate
from pandas.io.parsers import _evaluate_usecols
import submitit
import collections
import pickle
import itertools
import functools
from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf
import torch
from uimnet import utils
from uimnet import metrics
from uimnet import algorithms
from uimnet import ensembles
from uimnet import workers
from uimnet import __SLURM_CONFIGS__


@utils.timeit
def stack_tables_measurements(ood_results:List[List[Dict]]):
  # Collect records
  stacked = []
  for workers_results in ood_results:

    # Getting train and eval_cfg
    train_cfg = [el.pop('train_cfg') for el in workers_results][0]
    eval_cfg = [el.pop('eval_cfg') for el in workers_results][0]

    # measurements are dict of tensors
    op = functools.partial(torch.cat, dim=0)
    valid_tables = utils.map_dict([el['valid_tables'] for el in workers_results], op=op)
    test_tables = utils.map_dict([el['test_tables'] for el in workers_results], op=op)
    valid_measurements = utils.map_dict([el['valid_measurements'] for el in workers_results], op=op)
    test_measurements = utils.map_dict([el['test_measurements'] for el in workers_results], op=op)

    stacked += [dict(
      train_cfg=train_cfg,
      eval_cfg=eval_cfg,
      valid_tables=valid_tables,
      test_tables=test_tables,
      valid_measurements=valid_measurements,
      test_measurements=test_measurements,
    )]

  return stacked

def _collect_oodomain_records(cfg, all_results):
  results_by_train_cfg = collections.defaultdict(list)
  for result in all_results:
    key = result['train_cfg']
    results_by_train_cfg[key] += [result]
  results_by_train_cfg = dict(results_by_train_cfg)
  all_records = []
  Metrics = [metrics.__dict__[el] for el in cfg.metrics]
  for train_cfg, results in results_by_train_cfg.items():

    ids = dict()
    for i, result in enumerate(results):
      #utils.message(f"{result['eval_cfg'].dataset.partition}")
      ids[result['eval_cfg'].dataset.partition] = i
    #utils.message(ids)
    # Computing in vs easy
    for Metric in Metrics:
      in_results = results[ids['in']]
      # TODO: replace by results[ids['in_eval']]
      # for ood_partition in ['easy', 'difficult']:
      for ood_partition in ['easy']:
        oo_results = results[ids[ood_partition]]
        for measure_name in in_results['test_measurements']:
          metric = Metric(measurement_in_val=in_results['valid_measurements'][measure_name])
          value = metric(in_results['test_measurements'][measure_name],
                        oo_results['test_measurements'][measure_name],
                        in_results['test_tables'],
                        oo_results['test_tables'])

          record = dict(metric=metric.__class__.__name__,
                        value=value,
                        measurement=measure_name,
                        ood_partition=ood_partition,
                        temperature_mode=in_results['eval_cfg'].temperature_mode
                        )
          record.update(utils.flatten_nested_dicts(train_cfg))
          all_records += [record]

  return all_records

@utils.timeit
def collect_oodomain_records(cfg: OmegaConf, ood_results:List[List[Dict]]):
  """
  cfg: Config file
  ood_results: List of workers outputs.
  """

  all_results = stack_tables_measurements(ood_results)

  initial_temp_results = [el for el in all_results if el['eval_cfg'].temperature_mode == 'initial']
  learned_temp_results = [el for el in all_results if el['eval_cfg'].temperature_mode == 'learned']

  all_records = []
  all_records += _collect_oodomain_records(cfg, initial_temp_results)
  all_records += _collect_oodomain_records(cfg, learned_temp_results)

  return dict(ood=all_records)



def make_oodomain_dataframe(records):

  df = pd.DataFrame.from_records(records).round(4)
  df.value = df.value.apply(float)
  group_keys = ['measurement', 'metric', 'ood_partition', 'algorithm.name', 'algorithm.arch']
  val_keys = ['value']

  group_df = df.groupby(group_keys)[val_keys]
  mean_df, std_df, count_df = [el.round(4) for el in [group_df.mean(), group_df.std(), group_df.count()]]

  final_df = utils.make_error_bars_df_with_count(mean_df, std_df, count_df)

  return final_df

if __name__ == '__main__':
  pass
