#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch


class InAsIn(object):
  def __init__(self, measurement_in_val, alpha=0.05):
    self.alpha = alpha
    self.threshold = measurement_in_val.quantile(1 - alpha).item()
    self.true_class = 0
    self.predicted_class = 0

  def __call__(self, measurement_in, measurement_out):
    domain_predictions = torch.cat((
      measurement_in, measurement_out)).gt(self.threshold).float()
    # in-domain is labeled as 0, out-domain is labeled as 1
    domain_targets = torch.cat((torch.zeros_like(measurement_in),
                                torch.ones_like(measurement_out)))

    true_vec = domain_targets.eq(self.true_class)
    predicted_vec = domain_predictions.eq(self.predicted_class)

    return (true_vec * predicted_vec).sum() / true_vec.sum()


class InAsOut(InAsIn):
  def __init__(self, measurement_in_val, alpha=0.05):
      super().__init__(measurement_in_val, alpha)
      self.true_class = 0
      self.predicted_class = 1


class OutAsIn(InAsIn):
  def __init__(self, measurement_in_val, alpha=0.05):
      super().__init__(measurement_in_val, alpha)
      self.true_class = 1
      self.predicted_class = 0


class OutAsOut(InAsIn):
  def __init__(self, measurement_in_val, alpha=0.05):
      super().__init__(measurement_in_val, alpha)
      self.true_class = 1
      self.predicted_class = 1


if __name__ == '__main__':
  pass
