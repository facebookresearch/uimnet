#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
from uimnet import utils
import torch
import time
import itertools

from uimnet.algorithms import __ALGORITHMS__
from uimnet.measures import __MEASURES__
import pandas as pd
from tqdm import tqdm




def benchmark(nbatches, arch, device):
  records = []
  use_mixed_precisions = [False]
  iterator = list(itertools.product(use_mixed_precisions, __ALGORITHMS__.values()))
  for use_mixed_precision,  Algorithm in tqdm(iterator, desc='Benchmarking', total=len(iterator), leave=True):
    algorithm = Algorithm(num_classes=266,
                          arch=arch,
                          device=device,
                          seed=0,
                          use_mixed_precision=use_mixed_precision,

                          )
    algorithm.initialize()

    Measures = list(__MEASURES__.values())
    desc = f'Algorithm: {algorithm.__class__.__name__}, mixed_precision: {use_mixed_precision}'
    for Measure in tqdm(Measures, desc=desc, total=len(Measures), leave=False):
      algorithm.eval()
      measure = Measure(algorithm)

      eval_time = time.time()
      with torch.no_grad():
        for iter_ in range(nbatches):
          x = torch.zeros(32, 3, 224, 224).to(device)
          out = measure(x)
      eval_time = time.time() - eval_time
      records += [dict(algorithm=algorithm.__class__.__name__,
                        measure=measure.__class__.__name__,
                        eval_time=eval_time,
                        use_mixed_precision=use_mixed_precision,
                      )]
  return records

def main():
  device = 'cuda:0'
  records = benchmark(2, 'resnet18', device=device)
  df = pd.DataFrame.from_records(records).round(4)
  df = df.groupby(['measure', 'algorithm', 'use_mixed_precision'])['eval_time'].last()
  print(df.to_string() + '\n')

  return df


if __name__ == '__main__':
  main()
