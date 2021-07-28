#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import unittest
from parametrized import parametrized
from uimnet.algorithms.erm import ERM

class TestERM(unittest.TestCase):
  @parametrized.expand([
    ERM.__class__,
  ])



if __name__ == '__main__':
  pass
