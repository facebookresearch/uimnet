#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import collections
import functools
import torch
from uimnet import numerics
from uimnet import utils
from uimnet.ensembles import Bagging
from uimnet.measures.measure import Measure
from uimnet import __DEBUG__

class MixtureOfGaussians(Measure):
  def __init__(self, algorithm):
    super(MixtureOfGaussians, self).__init__(algorithm=algorithm)
    self.mus = {}
    self.covs = {}
    self.counts = {}
    self.N = None
    self.cov_estimator = 'full'


  def forward(self, x):

    _device = x.device
    # Two cases: Ensembles or not
    if isinstance(self.algorithm, Bagging):
      members = self.algorithm.algorithms
      N = x.shape
      K = self.algorithm.num_classes
      M = len(members)
      all_logprobs = torch.zeros(N, M, K)
      for m, member in enumerate(members):
        feat = member.get_features(x).cpu()
        all_logprobs[:, m, :] = member.gaussian_mixture.log_prob(feat)
      return numerics.log_marginalization_from_logits(all_logprobs).to(x.device)
    else:
      feat = self.algorithm.get_features(x).cpu()
      # import ipdb; ipdb.set_trace()
      return self.algorithm.gaussian_mixture.log_prob(feat).view(-1, ).to(_device)

  def estimate(self, train_loader):
    _to_device = functools.partial(utils.to_device, device=self.algorithm.device)

    collected = collections.defaultdict(list)
    utils.message('Collecting logits and features')
    with torch.no_grad():
      for i, batch in enumerate(train_loader):
        batch = utils.apply_fun(_to_device, batch)
        x, y = batch['x'], batch['y']
        _feats = self.algorithm.get_features(x).detach()
        _y = y.detach()
        if utils.is_distributed():
          _feats = torch.cat(utils.all_gather(_feats), dim=0)
          _y = torch.cat(utils.all_gather(_feats), dim=0)
        collected['features'] += [_feats.cpu()]
        collected['y'] += [_y.cpu()]
        if __DEBUG__ and i > 2:
          break

      collected = dict(collected)
      collected = {k: torch.cat(v, dim=0) for k, v in collected.items()}

      num_classes = self.algorithm.num_classes
      all_classes = collected['y'].unique()
      utils.message(f'{type(self.algorithm)}:{len(all_classes)}, {num_classes}')
      assert len(all_classes) == num_classes

      for y in all_classes:
        mask = torch.where(y == collected['y'])
        X = collected['features'][mask]
        mu_hat = X.mean(dim=0)  # D
        cov_hat = self.estimate_cov(X, mu_hat, cov_estimator=self.cov_estimator)
        self.mus[int(y)] = mu_hat
        self.covs[int(y)] = cov_hat
        self.counts[int(y)] = len(y)
      self.N = sum(self.counts.values())

  def log_prob(self, x):
    with torch.no_grad():
      K = len(self.counts)
      D = x.shape[1]



    # GMM estimation
if __name__ == '__main__':
  pass
