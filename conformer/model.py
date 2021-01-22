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

import torch.nn as nn
from torch import Tensor

from conformer.conv import Conv2dSubampling
from conformer.decoder import ConformerDecoder
from conformer.encoder import ConformerEncoder
from conformer.wrapper import Linear


class Conformer(nn.Module):
    """
    Conformer: Convolution-augmented Transformer for Speech Recognition
    - https://arxiv.org/pdf/2005.08100.pdf
    """
    def __init__(
            self,
            input_dim: int = 80,
            encoder_dim: int = 512,
            num_layers: int = 17,
            dropout_p: float = 0.1,
    ) -> None:
        super(Conformer, self).__init__()

        self.conv_subsample = Conv2dSubampling(in_channels=1, out_channels=encoder_dim)
        self.input_projection = Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim)
        self.input_dropout = nn.Dropout(p=dropout_p)

        self.encoder = ConformerEncoder(encoder_dim, num_layers)
        self.decoder = ConformerDecoder()

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        """
        x: B x T x D
        """
        inputs, _ = self.extractor(inputs.transpose(1, 2), input_lengths)
        inputs = self.input_dropout(self.input_projection(inputs.transpose(1, 2)))

        encoder_outputs = self.encoder(inputs)
