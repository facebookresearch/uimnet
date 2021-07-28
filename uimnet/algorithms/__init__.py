#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
from uimnet.algorithms.base import Algorithm
from uimnet.algorithms.distillation import RND, OC
from uimnet.algorithms.erm import ERM
from uimnet.algorithms.mcdropout import MCDropout
from uimnet.algorithms.mimo import MIMO
from uimnet.algorithms.mixup import Mixup
from uimnet.algorithms.rbf import RBF
from uimnet.algorithms.soft_labeler import SoftLabeler
from uimnet.algorithms.due import DUE

__algorithms__ = [RND, OC, ERM, MCDropout, MIMO, Mixup, RBF, SoftLabeler, DUE]
__ALGORITHMS__ = {el.__name__: el for el in __algorithms__}
