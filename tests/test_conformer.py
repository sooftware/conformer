import pytest
import torch

from conformer import Conformer
from conformer.convolution import Conv2dSubampling


def test_subsampling_output_lengths_match_output_size():
    subsample = Conv2dSubampling(in_channels=1, out_channels=4)

    for length in range(7, 100):
        outputs, output_lengths = subsample(torch.rand(1, length, 80), torch.LongTensor([length]))
        assert output_lengths.item() == outputs.size(1)


@pytest.mark.parametrize("length", [7, 120, 121, 123])
def test_padding_does_not_change_outputs(length):
    torch.manual_seed(0)
    model = Conformer(num_classes=10, input_dim=80, encoder_dim=32, num_encoder_layers=2, num_attention_heads=4)
    model.eval()

    short = torch.rand(1, length, 80)
    batch = torch.rand(2, 200, 80)
    batch[0, :length] = short[0]

    with torch.no_grad():
        expected, expected_lengths = model(short, torch.LongTensor([length]))
        outputs, output_lengths = model(batch, torch.LongTensor([length, 200]))

    assert output_lengths[0] == expected_lengths[0] == expected.size(1)
    assert torch.allclose(outputs[0, :expected.size(1)], expected[0], atol=1e-5)


def test_half_precision_forward():
    model = Conformer(num_classes=10, input_dim=80, encoder_dim=32, num_encoder_layers=1, num_attention_heads=4)
    model.eval().half()

    with torch.no_grad():
        outputs, _ = model(torch.rand(2, 100, 80).half(), torch.LongTensor([100, 60]))

    assert torch.isfinite(outputs).all()
