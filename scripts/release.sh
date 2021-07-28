#!/usr/bin/env bash
##
## # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
##
##
set -euo pipefail

cd $UI_ROOT || exit
export DEBUG=0
export USE_CHECKPOINTS=0

SWEEP_DIR="$1"
./scripts/main.py -c ./configs/release -s $SWEEP_DIR -m NLL
