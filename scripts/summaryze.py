#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import argparse
import prompt_toolkit
from prompt_toolkit import prompt
from pathlib import Path
from uimnet import utils
from uimnet import measures
import concurrent.futures
import pickle
import pandas as pd
import tabulate

MAX_WORKERS = 10

STAGES = ['train', 'calibration', 'prediction'] + \
  [f'ood_{el}' for el in measures.__MEASURES__.keys()]

def parse_arguments():
  parser = argparse.ArgumentParser(description='Check completions of sweeps stages')
  parser.add_argument('-s', '--sweep_dir', type=str, required=True)

  return parser.parse_args()

def main(args):
  sweep_path = Path(args.sweep_dir)
  loop(sweep_path)
  return

def _help():
  return """
  help: prints help.
  q: quits.
  summarize: prints summary.

  command should be of the form (stage, state, query)
  """

def loop(sweep_path):
  while True:
    prompt_msg = '>'
    command = prompt(prompt_msg)
    records = check_sweep_stages(sweep_path)
    df = get_df(records)
    print(command)
    try:
      if command == 'help':
        print(_help())
      elif command == 'summarize':
        print(df)
      elif command == 'q':
        break
      else:
        stage, state, query = command.split(' ')
        idx = (stage, )
        if state != 'all':
          idx += (state, )
        print(df.loc[idx][query])
    except Exception as e:
      print(f'Invalid command {command}')
      print(e)
      continue
  return


def get_df(records):
  df = pd.DataFrame.from_records(records)
  grouped = df.groupby(['stage', 'state'])['model'].agg(count='count', paths=lambda el: list(el))
  return grouped

def check_model_stages(model_path):

  records = []
  for stage in STAGES:
    trace_path = model_path / f'{stage}.state'
    if trace_path.exists():
      with open(trace_path, 'r') as fp:
        state = fp.read()
    else:
      state = 'missing'
    record = dict(model=str(model_path),
                  stage=stage,
                  state=state)
    records += [record]
  return records

def check_sweep_stages(sweep_path):

  subpaths = [el for el in sweep_path.iterdir() if el.is_dir()]
  def _is_model_path(path):
    return (path /  '.algorithm').exists() or (path / '.ensemble').exists()
  models_paths = filter(_is_model_path, subpaths)

  with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    all_records = sum(list(executor.map(check_model_stages, models_paths)), [])

  return all_records


if __name__ == '__main__':
  args = parse_arguments()
  all_records = main(args)
