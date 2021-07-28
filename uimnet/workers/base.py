#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import os
import socket

from omegaconf.omegaconf import OmegaConf
import torch
import torch.distributed as dist

import numpy as np
from uimnet import utils
import submitit
from uimnet import __DEBUG__

class Worker(object):

  def __init__(self, *args, **kwargs):
    super(Worker, self).__init__()


  def checkpoint(self, *args, **kwargs):

    if utils.is_not_distributed_or_is_rank0():
      new_callable = Worker()
      return submitit.helpers.DelayedSubmission(new_callable, *args, **kwargs)


  def setup(self, cfg):

    self.maybe_setup_distributed(cfg)
    utils.set_random_seed(cfg.experiment.seed)
    torch.cuda.set_device(cfg.experiment.device)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    return

  def maybe_setup_distributed(self, cfg):

    # Turning off the read only flags on the config file
    OmegaConf.set_readonly(cfg, False)

    if cfg.experiment.distributed:
      utils.message('Distributed job detected')

      cfg.experiment.world_size = os.environ.get('WORLD_SIZE', cfg.experiment.world_size)
      cfg.experiment.rank = os.environ.get('RANK', cfg.experiment.world_size)

      # Scenario 1: Initializing from environment
      if cfg.experiment.dist_protocol == 'env':
        cfg.experiment.dist_url = f'{cfg.experiment.dist_protocol}://'

      # Scenario 2: Check if using SLURM on the FAIR cluster
      elif 'SLURM_JOB_ID' in os.environ and cfg.experiment.platform == 'slurm':
        # TODO Improve SLURM Support:
        # - n different tasks on the same node will require n different IPs
        # - Specifying local rank for jobs multi-nodes jobs with unequal
        # numbers of workers per node.
        utils.message('Slurm job detected')
        _world_size = int(os.environ['SLURM_NNODES']
                         ) * int(os.environ["SLURM_TASKS_PER_NODE"][0])
        utils.message(f'Slurm world size={_world_size}')
        cfg.experiment.world_size = int(os.environ['SLURM_NTASKS'])
        assert _world_size == cfg.experiment.world_size

        procid = int(os.environ['SLURM_PROCID'])
        cfg.experiment.rank = procid
        # Using TCP/IP by default on SLURM
        cfg.experiment.dist_protocol = 'tcp'
        # Using first node as master node
        master_addr = os.getenv("SLURM_JOB_NODELIST").split(',')[0].replace(
            '[', ''
        )
        master_port = f'4000'
        cfg.experiment.dist_url = f'{cfg.experiment.dist_protocol}://{master_addr}:{master_port}'
        cfg.experiment.num_workers = max(2, (torch.multiprocessing.cpu_count() // torch.cuda.device_count()) - 2)

      # Scenario 3: TCP/IP
      elif cfg.experiment.dist_protocol == 'tcp':
        # Each script should be called with
        master_addr = str(os.getenv('MASTER_ADDR', socket.gethostname()))
        master_port = str(os.getenv('MASTER_PORT', 4000))
        cfg.experiment.dist_url = f'{cfg.experiment.dist_protocol}://{master_addr}:{master_port}'

      # Scenario 4: File initialization
      elif cfg.experiment.dist_protocol == 'file':
        connection_file = os.environ.get('CONNECTION_FILE')
        cfg.experiment.dist_url = f'{cfg.experiment.dist_protocol}://{connection_file}'

      # Scenario 5: explicitely provided connection address
      else:
        if cfg.experiment.dist_protocol is None and cfg.experiment.dist_url is None:
          err_msg = f'Specify dist_url or valid dist_protocol'
          raise ValueError(err_msg)

      assert cfg.experiment.dist_url is not None
      assert cfg.experiment.rank is not None
      assert cfg.experiment.world_size is not None

      cfg.experiment.rank = int(cfg.experiment.rank)
      cfg.experiment.world_size = int(cfg.experiment.world_size)
      dist.init_process_group(
          backend='nccl',
          init_method=cfg.experiment.dist_url,
          rank=cfg.experiment.rank,
          world_size=cfg.experiment.world_size
      )

      local_rank = cfg.experiment.rank % torch.cuda.device_count()
      cfg.experiment.local_rank = cfg.experiment.local_rank
      cfg.experiment.device = f"cuda:{local_rank}"

    if cfg.experiment.device is None:
      err_msg = 'Please specify device'
      raise ValueError(err_msg)

    OmegaConf.set_readonly(cfg, True)
    return


  def save(self, cfg):
    return

if __name__ == '__main__':
  pass
