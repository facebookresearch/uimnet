#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from uimnet.modules.spectral_normalization.base import SN

# 2D Conv layer with spectral norm
class SNConv1d(nn.Conv2d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True,
               num_svs=1, num_itrs=1, eps=1e-12, sn_coef=1):
    nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, stride,
                     padding, dilation, groups, bias)
    SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps, sn_coef=sn_coef)

  def forward(self, x):
    return F.conv1d(x, self.W_(), self.bias, self.stride,
                    self.padding, self.dilation, self.groups)

# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True,
               num_svs=1, num_itrs=1, eps=1e-12, sn_coef=1):
    nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                     padding, dilation, groups, bias)
    SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps, sn_coef=sn_coef)
  def forward(self, x):
    return F.conv2d(x, self.W_(), self.bias, self.stride,
                    self.padding, self.dilation, self.groups)

if __name__ == '__main__':
  pass
