#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import scipy
import scipy.stats
from scipy.stats import multivariate_normal
import numpy as np
from uimnet import utils


class GaussianMixture(nn.Module):

    def __init__(self, K, D):
      super(GaussianMixture, self).__init__()

      self.D = D
      self.K = K
      self.register_buffer('means',  torch.zeros(K, D))
      self.register_buffer('covs', torch.cat([torch.eye(D).unsqueeze(0) for _ in range(K)], dim=0))
      self.register_buffer('counts', torch.ones(K).long())


    def logsumexp(self, x):
      x_max = x.max(1, keepdim=True).values
      return x_max + (x - x_max).exp().sum(1, keepdim=True).log()

    def add_gaussian_from_data(self, x, y, eps=0):
      with torch.no_grad():
        idx = int(y)
        x = x.detach()

        x_num = torch.as_tensor(len(x))
        self.counts[idx] = x_num

        x_mean = x.mean(0)
        self.means[idx] = x_mean
        x_centered = x - x_mean
        x_covariance = (1 / (x_num - 1)) * (x_centered.t() @ x_centered)
        x_covariance += (torch.eye(x_covariance.size(1)).to(x.device) * eps)

        self.covs[idx] = x_covariance

    def log_prob(self, x):
      x = x.detach()
      device = x.device

      weights = self.counts / self.counts.sum()

      K = len(weights)
      # log_weights = torch.log(weights).view(1, -1).cpu().numpy()
      # log_probs = torch.zeros(x.shape[0], K).cpu().numpy()
      log_weights = torch.log(weights).view(1, -1)
      log_probs = torch.zeros(x.shape[0], K)

      for k, (mean, cov) in enumerate(zip(self.means, self.covs)):
        gaussian = MultivariateNormal(mean, cov)
        log_probs[:, k] = gaussian.log_prob(x).cpu()
        # gaussian = multivariate_normal(mean.cpu().detach().numpy(),
        #                                cov.cpu().detach().numpy())
        # log_prob = gaussian.logpdf(x.detach().cpu().numpy())
        # log_probs[:, k] = log_prob
      # log_prob = scipy.special.logsumexp(log_weights + log_probs, axis=1)
      # return log_prob
      return torch.logsumexp(log_weights + log_probs, dim=1).to(device)

if __name__ == '__main__':
  for D in [2**d for d in range(1, 10)]:
    gm = GaussianMixture(266, D)
    x = torch.zeros(10, D).normal_()
    print(f'Value at D={D}, {gm.log_prob(x)}')
