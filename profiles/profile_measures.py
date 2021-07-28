#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
from uimnet import utils
import time
import itertools

import torch
import torch.autograd.profiler as profiler

from uimnet.algorithms import __ALGORITHMS__
from uimnet.measures import __MEASURES__
import pandas as pd
from tqdm import tqdm




def profile_measures(arch, device):
  records = []
  use_mixed_precisions = [False, True]
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
      with torch.no_grad():
        x = torch.zeros(32, 3, 224, 224).to(device)
        with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
          msg = f'{algorithm.__class__.__name__} and {measure.__class__.__name__} (mixed precision: {use_mixed_precision})'
          with profiler.record_function(msg):
            out = measure(x)
          records += [dict(algorithm=algorithm.__class__.__name__,
                            measure=measure.__class__.__name__,
                            use_mixed_precision=use_mixed_precision,
                            prof=prof,                          )]

  return records

def main():
  device = 'cuda:0'
  records = profile_measures('resnet18', device)

  for record in records:
    print(f"{record['algorithm']}/{record['measure']} (mixed precision: {record['use_mixed_precision']})\n")
    print(record['prof'].key_averages().table(sort_by='cuda_memory_usage', row_limit=10))
    print("\n")


  return records


if __name__ == '__main__':
  main()
