# -*- coding: utf-8 -*-
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
"""
Package wide configuration
"""
import os
import getpass
from pathlib import Path

os.environ['MKL_THREADING_LAYER'] = 'gnu'   # Weird bug.
__DEBUG__ = int(os.getenv('DEBUG', 0))
__USE_CHECKPOINTS__ = int(os.getenv('USE_CHECKPOINTS', 0))
__USERNAME__ = getpass.getuser()
__PROJECT_ROOT__ = os.path.realpath(
    os.path.join(
        os.path.dirname(os.path.join(os.path.abspath(__file__))),
        '..'))  # Project Root

# SLURM configurations
__SLURM_CONFIGS__ = {
    'one_gpu': dict(nodes=1, gpus_per_node=1, ntasks_per_node=1),
    'distributed_4': dict(nodes=1, gpus_per_node=4, ntasks_per_node=4),
    'distributed_8': dict(nodes=1, gpus_per_node=8, ntasks_per_node=8),
    'distributed_16': dict(nodes=2, gpus_per_node=8, ntasks_per_node=8),
    'distributed_32': dict(nodes=4, gpus_per_node=8, ntasks_per_node=8)
}

if __name__ == 'main':
  pass
