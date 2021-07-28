#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import collections
from pathlib import Path
import functools
import concurrent
import concurrent.futures

import pandas as pd
from omegaconf import OmegaConf
from torch import multiprocessing as tmp
from uimnet import utils


class ExtractRecords(object):
  def __init__(self, _tables):
    self._tables = _tables

  def __call__(self, subdir):
    _tables = self._tables
    records = collections.defaultdict(list)
    train_cfg_fmt = 'cfg_rank_0.yaml'  # TODO(Ishmael): handle non distributed case

    with open(subdir / train_cfg_fmt, 'r') as fp:
      train_cfg = OmegaConf.load(fp.name)

    # load the csv files
    for i, filename in enumerate(map(lambda el: f'{el}_data.csv', _tables)):

      df = pd.read_csv(subdir / filename)

      for key, item in utils.flatten_nested_dicts(train_cfg).items():
        df[key] = item

      records[_tables[i]] += df.to_dict('records')
    return dict(records)


@utils.timeit
def collect_indomain_records(sweep_dir, _tables=None):

  if _tables is None:
    _tables = ['mainloop', 'evaluation']  # Data collected on the trainer

  def _is_valid_subpath(subpath):

    if not subpath.exists():
      return False

    if utils.trace_exists('train.done', dir_=str(subpath)):
      return True

    return False

  sweep_path = Path(sweep_dir)
  subdirs = filter(_is_valid_subpath, sweep_path.iterdir())
  _extract_records = ExtractRecords(_tables=_tables)

  max_workers = max(2, tmp.cpu_count() - 2)
  with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
     all_records = list(executor.map(_extract_records, subdirs))

  all_records = utils.map_dict(all_records)

  return dict(all_records)



def make_indomain_dataframe(evaluation_records: list):
  """
  Generates In-Domain table
  -------------------------
  """
  # Generating the predictive results dataframe
  df = pd.DataFrame.from_records(evaluation_records).round(4)

  final_epoch = df.epoch.max()
  epoch_df = df[df.epoch == final_epoch]

  group_keys = ['algorithm.arch', 'dataset.name', 'algorithm.name', 'algorithm.sn','temperature_mode', 'split']
  val_keys = ['ACC@1', 'ACC@5', 'NLL', 'ECE']

  group_df = epoch_df.groupby(group_keys)[val_keys]
  mean_df, std_df, count_df = [el.round(4) for el in [group_df.mean(), group_df.std(), group_df.count()]]

  final_df = utils.make_error_bars_df_with_count(mean_df, std_df, count_df)

  return final_df

if __name__ == '__main__':
  pass
