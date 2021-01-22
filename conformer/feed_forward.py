import torch.nn as nn
from torch import Tensor

from conformer.activation import Swish
from conformer.wrapper import LayerNorm, Linear


class FeedForwardNet(nn.Module):
    """
    Conformer Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
    and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
    regularizing the network.

    Args:
        encoder_dim: Dimension of conformer encoder
        dropout_p: Ratio of dropout

    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by feed forward module.
    """
    def __init__(self, encoder_dim: int = 512, dropout_p: float = 0.1):
        super(FeedForwardNet, self).__init__()
        self.sequential = nn.Sequential(
            LayerNorm(encoder_dim),
            Linear(encoder_dim, encoder_dim << 2, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim << 2, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)
