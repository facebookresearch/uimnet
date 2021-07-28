#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import collections
import pickle
import pandas as pd
import torch
from pathlib import Path
from uimnet import utils
import argparse
def parse_arguments():
  parser = argparse.ArgumentParser(description='converts evaluation records into paper records')
  parser.add_argument('-i', '--input', type=str, help='Path to evaluation records pickle in sweep folder')
  parser.add_argument('-O', '--output', type=str,  help='savepath')

  return parser.parse_args()

PREDICTIVE_METRICS = ['ACC@1', 'ACC@5', 'NLL', 'ECE']

def check_metric(predictive_metric):
  if not (predictive_metric in PREDICTIVE_METRICS):
    err_msg = f'Unrecognized predictive metric: {predictive_metric}.'
    raise ValueError(err_msg)
  return

def marginalize_data_seeds(df, predictive_metric, reset_index=True):
  check_metric(predictive_metric)

  group_keys = ['algorithm.name', 'algorithm.arch', 'algorithm.sn', 'algorithm.seed']
  # Marginalizing over data seeds
  grouped = df.groupby(group_keys)
  #ops = ['mean', 'std', 'count']
  marginalized_df= grouped[PREDICTIVE_METRICS].mean()
  if reset_index:
    marginalized_df = marginalized_df.reset_index()
  return marginalized_df


def rank_algorithm_seeds(df, predictive_metric, reset_index=True):
  check_metric(predictive_metric)

  marginalized_df = marginalize_data_seeds(df, predictive_metric)
  # Ranking algorithm seeds according to the mean performance accross dataseeds
  ascending = True
  if predictive_metric in ['NLL', 'ECE']:
    ascending = False

  ranked_df = marginalized_df.groupby(['algorithm.name', 'algorithm.arch', 'algorithm.sn']) \
                             .apply(lambda el: el.sort_values([predictive_metric], ascending=ascending))
  if reset_index:
    ranked_df = ranked_df.reset_index(drop=True)
  return ranked_df


def select_topk_algorithm_seeds(df, predictive_metric, k=1, reset_index=True):
  check_metric(predictive_metric)
  marginalized_df = marginalize_data_seeds(df, predictive_metric)
  ascending = True
  if predictive_metric in ['NLL', 'ECE']:
    ascending = False
  best_df = marginalized_df.groupby(['algorithm.name', 'algorithm.arch', 'algorithm.sn']) \
                           .apply(lambda el: el.sort_values([predictive_metric], ascending=ascending).nsmallest(k, predictive_metric))

  if reset_index:
    best_df = best_df.reset_index(drop=True)

  return best_df

def select_models(all_records, predictive_metric, verbose=True):
  """
  Select top-K algorithm seeds corresponding to the best predictive metrics.
  """
  evaluation_records = all_records['evaluation']
  # Selection based on evaluation records
  df = pd.DataFrame.from_records(evaluation_records)
  # HACK(Ishmael): Train epoch is awfully slow. So ensembles are "trained" for
  # only one epoch. We set the epoch counter for ensemble models to that of non
  # ensembles models
  # --- Hack starts
  if 'ensemble.name' in df.columns:
    ensembles_idx = df['ensemble.name'].isin(['Bagging'])
    max_epoch = df[~ensembles_idx]['epoch'].max()
    df.epoch[ensembles_idx] = max_epoch
    assert max_epoch == df.epoch.max()
  # --- Hack ends.
  # Selecting best seeds
  cond = (df.epoch == df.epoch.max()) & (df.split == 'eval')
  df = df[cond]
  best_df = select_topk_algorithm_seeds(df, predictive_metric, k=1)
  if verbose:
    utils.message(f'Selected models based on in-domain {predictive_metric}')
    utils.message(best_df.to_string())

  model_keys = ['algorithm.name', 'algorithm.arch', 'algorithm.seed']
  best_models= best_df[model_keys].to_dict('records')
  #  Selecting algorithm seeds
  filtered_records = collections.defaultdict(list)
  for record_name, records in all_records.items():

    if records is None:
      continue

    for record in records:
      model = {k: v for k,v in record.items() if k in model_keys}
      if model in best_models:
        ## HACK(Ishmael): Fuggly hack... don't live in a house with broken windows.
        ## --- Hack starts.
        if 'Bagging' in record['algorithm.name']:
          record['epoch'] = max_epoch
        ## -- Hack ends.
        filtered_records[record_name] += [record]

  filtered_records = dict(filtered_records)
  return filtered_records

def make_indomain_dataframes(evaluation_records: list):
  """
  Generates In-Domain table
  -------------------------
  """
  # Generating the predictive results dataframe
  df = pd.DataFrame.from_records(evaluation_records).round(4)
  df['algorithm.temperature_mode'] = 'initial'
  df['metric_in'] = True

  final_epoch = df.epoch.max()
  epoch_df = df[(df.epoch == final_epoch) & (df.split == 'eval')]

  group_keys = ['algorithm.name', 'algorithm.sn', 'algorithm.temperature_mode']
  val_keys = ['ACC@1', 'ACC@5', 'NLL', 'ECE']

  group_df = epoch_df.groupby(group_keys)[val_keys]
  mean_df, std_df, count_df = [el.round(4) for el in [group_df.mean(), group_df.std(), group_df.count()]]

  mean_df = mean_df.melt(value_vars=val_keys, var_name='metric', value_name='std_value', ignore_index=False).reset_index()
  mean_df.columns = ['algorithm', 'spectral_on', 'calibration_on', 'metric', 'value_mean']
  std_df = std_df.melt(value_vars=val_keys, var_name='metric', value_name='mean_value', ignore_index=False).reset_index()
  std_df.columns = ['algorithm', 'spectral_on', 'calibration_on', 'metric', 'value_std']

  mean_df['value_std'] = std_df['value_std']
  mean_df['measure'] = 'in'

  mean_df['k'] = 1
  return mean_df

def make_oodomain_dataframes(records, selected_partition='easy'):

  df = pd.DataFrame.from_records(records).round(4)
  df = df[df.ood_partition == selected_partition]
  df['metric_in'] = False
  df.value = df.value.apply(float)
  #group_keys = ['algorithm.name', 'algorithm.arch', 'algorithm.sn', 'algorithm.temperature_mode', 'measurement', 'ood_partition', 'metric']
  group_keys = ['algorithm.name', 'algorithm.sn', 'temperature_mode', 'measurement', 'metric']
  val_keys = ['value']

  group_df = df.groupby(group_keys)[val_keys]
  mean_df, std_df, count_df = [el.round(4).reset_index() for el in [group_df.mean(), group_df.std(), group_df.count()]]

  mean_df.columns= ['algorithm', 'spectral_on', 'calibration_on', 'measure', 'metric', 'value_mean']
  std_df.columns = ['algorithm', 'spectral_on', 'calibration_on', 'measure', 'metric', 'value_std']

  mean_df['value_std'] = std_df['value_std']
  #mean_df.columns = mean_df.columns.droplevel()
  #mean_df.columns.name = ''
  #std_df.columns = std_df.columns.droplevel()
  #std_df.columns.name = ''

  mean_df['k'] = 1
  return mean_df

def prepare_table_records(all_records, predictive_metric='NLL', selected_partition='easy'):

    filtered_records = select_models(all_records, predictive_metric=predictive_metric, verbose=True)
    # First perform model selelection
    idf_mean = make_indomain_dataframes(filtered_records['evaluation'])
    odf_mean = make_oodomain_dataframes(filtered_records['ood'])


    return idf_mean.to_dict('records') + odf_mean.to_dict('records')

if __name__ == '__main__':
  args = parse_arguments()
  with open(args.input, 'rb') as fp:
    all_records = pickle.load(fp)

  table_records = prepare_table_records(all_records)
  with open(args.output, 'wb') as fp:
    pickle.dump(table_records, fp)
