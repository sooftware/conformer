import torch.nn as nn

from conformer import Conformer


def test_update_dropout_reaches_every_dropout_layer():
    model = Conformer(num_classes=10, input_dim=80, encoder_dim=32, num_encoder_layers=2, num_attention_heads=4)

    model.update_dropout(0.5)

    dropouts = [module for module in model.modules() if isinstance(module, nn.Dropout)]
    assert dropouts and all(dropout.p == 0.5 for dropout in dropouts)
