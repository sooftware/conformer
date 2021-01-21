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
from torch import Tensor
from conformer.submodules import LayerNorm
from conformer.modules import (
    FeedForwardNet,
    MultiHeadAttention,
    ConformerConvModule,
)


class ConformerBlock(nn.Module):
    def __init__(self, encoder_dim: int = 512):
        super(ConformerBlock, self).__init__()
        self.feed_forward1 = FeedForwardNet()
        self.attention = MultiHeadAttention()
        self.conv = ConformerConvModule()
        self.feed_forward2 = FeedForwardNet()
        self.layer_norm = LayerNorm(encoder_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = 0.5 * self.feed_forward1(x) + x
        x = self.attention(x) + x
        x = self.conv(x) + x
        x = 0.5 * self.feed_forward2(x) + x
        return self.layer_norm(x)
