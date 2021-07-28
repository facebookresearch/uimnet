#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch


def log_marginalization_from_logits(logits):

  l = logits
  N, M, K = l.shape
  log_Znm = torch.logsumexp(l, dim=2, keepdim=True) # N X M X 1
  ltilde = l - log_Znm # N X M X K
  logM = torch.log(torch.as_tensor([[M]]).float()).to(l.device)

  log_py_given_x = -logM + torch.logsumexp(ltilde, dim=1)  # N X K
  return log_py_given_x

def test_log_marginalisation_from_logits():

  N = 1000
  K = 266
  M = 10
  logits = torch.zeros(N, M, K).normal_()
  log_py_given_x = log_marginalization_from_logits(logits)
  py_given_x = log_py_given_x.exp()

  greater_than_zero = torch.ge(py_given_x, 0)
  assert greater_than_zero.long().sum().bool().item()
  smaller_than_one = torch.ge(-py_given_x, -1)
  assert smaller_than_one.long().sum().bool().item()
  sum_to_one = py_given_x.sum(dim=1)
  assert torch.allclose(sum_to_one, torch.ones_like(sum_to_one))

if __name__ == '__main__':
  pass
