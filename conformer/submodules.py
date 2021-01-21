# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class FastGLU(nn.Module):
    def __init__(self, in_features: int) -> None:
        super(FastGLU, self).__init__()

        self.in_features = in_features
        self.linear = Linear(in_features, in_features << 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: B x C x D x T
        :return:
        """
        output = self.linear(x)
        return output[:, :self.in_features, :] * output[:, self.in_features:, :].sigmoid()


class DepthwiseConv1d(nn.Module):
    def __init__(self, num_channels: int, kernel_size: int) -> None:
        super(DepthwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, groups=num_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class PointwiseConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class LayerNorm(nn.Module):
    """ Wrapper class of torch.nn.LayerNorm """
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mean = z.mean(dim=-1, keepdim=True)
        std = z.std(dim=-1, keepdim=True)
        output = (z - mean) / (std + self.eps)
        output = self.gamma * output + self.beta
        return output


class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class View(nn.Module):
    """ Wrapper class of torch.view() for Sequential module. """
    def __init__(self, shape: tuple, contiguous: bool = False):
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous

    def forward(self, inputs):
        if self.contiguous:
            inputs = inputs.contiguous()

        return inputs.view(*self.shape)


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, inputs: Tensor):
        return inputs.transpose(*self.shape)