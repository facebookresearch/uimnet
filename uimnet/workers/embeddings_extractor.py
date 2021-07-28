#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import collections
import functools
import os
from pathlib import Path

import time
import torch
import torch.distributed as dist

from omegaconf import OmegaConf

from uimnet import utils
from uimnet import workers
from uimnet import __DEBUG__




class EmbeddingsExtractor(workers.Worker):

  def __init__(self):
    # Contains workers state
    super(EmbeddingsExtractor, self).__init__()

    self.cfg = None
    self.encoder = None
    self.datanode = None


  def __call__(self, cfg, encoder, datanode):

    self.setup(cfg)

    # Dump configuration file
    output_path = Path(cfg.sweep_dir)
    output_path.mkdir(parents=True, exist_ok=True)


    with open(output_path / f'cfg_{cfg.experiment.rank}', 'w') as fp:
     OmegaConf.save(cfg, f=fp.name)

    self.encoder = encoder.to(cfg.experiment.device)
    if utils.is_distributed():
      encoder = torch.nn.parallel.DistributedDataParallel(
        encoder, device_ids=[cfg.experiment.device])
    self.datanode = datanode

    _to_device = functools.partial(utils.to_device,
                                   device=cfg.experiment.device)

    self.encoder.eval()
    self.datanode.eval()

    _pin_memory = True if 'cuda' in cfg.experiment.device else False,
    loader = self.datanode.get_loader(batch_size=cfg.dataset.batch_size,
                                      shuffle=False,
                                      pin_memory=_pin_memory,
                                      num_workers=cfg.experiment.num_workers)


    out = collections.defaultdict(list)
    with torch.no_grad():
      for i, batch in enumerate(loader):
        batch = utils.apply_fun(_to_device, batch)
        out['embeddings'] += [self.encoder(batch['x'])]
        out['targets'] += [batch['y']]
        out['indices'] += [batch['index']]
        if i > 2 and __DEBUG__:
          break

      out = {k: torch.cat(v, dim=0).detach().cpu() for k, v in out.items()}

    return out

if __name__ == '__main__':
  pass
