#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import pandas as od
from torch.cuda.memory import reset_accumulated_memory_stats

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
  ascending = False
  if predictive_metric in ['NLL', 'ECE']:
    ascending = True

  ranked_df = marginalized_df.groupby(['algorithm.name', 'algorithm.arch', 'algorithm.sn']) \
                             .apply(lambda el: el.sort_values([predictive_metric], ascending=ascending))
  if reset_index:
    ranked_df = ranked_df.reset_index(drop=True)
  return ranked_df


def select_topk_algorithm_seeds(df, predictive_metric, k=1, reset_index=True):
  check_metric(predictive_metric)
  marginalized_df = marginalize_data_seeds(df, predictive_metric)

  best_df = marginalized_df.groupby(['algorithm.name', 'algorithm.arch', 'algorithm.sn'])
  if predictive_metric in ['NLL', 'ECE']:
    ascending = False
    best_df = best_df.apply(lambda el: el.sort_values([predictive_metric], ascending=ascending).nsmallest(k, predictive_metric))
  else:
    ascending = True
    best_df = best_df.apply(lambda el: el.sort_values([predictive_metric], ascending=ascending).nlargest(k, predictive_metric))

  if reset_index:
    best_df = best_df.reset_index(drop=True)

  return best_df

if __name__ == '__main__':
  pass
