#!/usr/bin/env bash
##
## # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
##
##
set -euo pipefail

cd $UI_ROOT || exit
export DEBUG=1
export USE_CHECKPOINTS=1

SWEEP_DIR="$1"
./scripts/main.py -c ./configs/dev -s $SWEEP_DIR -m NLL
