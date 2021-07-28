#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
from uimnet.measures.entropy import Entropy
from uimnet.measures.largest import Largest
from uimnet.measures.gap import Gap
from uimnet.measures.jacobian import Jacobian
from uimnet.measures.deep_jacobian import DeepJacobian
from uimnet.measures.logsumexp import LogSumExp
from uimnet.measures.native import Native
from uimnet.measures.mog import MixtureOfGaussians
from uimnet.measures.augmentation import Augmentations

__measures__ = [Entropy, Largest, Gap, Jacobian, DeepJacobian, LogSumExp, Native,
                MixtureOfGaussians,
                Augmentations
                ]
__MEASURES__ = {el.__name__: el for el in __measures__}
