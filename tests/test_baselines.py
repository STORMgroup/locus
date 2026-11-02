"""locus/finetune/baselines.py: the encoder reads every base, averages the forward half and the channel-flipped
reverse-complement half of a reverse-complement-equivariant model's output, and pools over the real bases only."""
from types import SimpleNamespace

import torch
import torch.nn as nn

from locus.finetune.baselines import BaselineEncoder


class Recorder(nn.Module):
    """Stand-in backbone: remembers the input length and returns the first `length` positions of fixed hidden states."""

    def __init__(self, hidden: torch.Tensor):
        super().__init__()
        self.hidden = hidden

    def forward(self, tokens):
        self.length = tokens.shape[1]
        return (self.hidden[:, :tokens.shape[1]],)


def batch(length: int, valid: int) -> SimpleNamespace:
    ids = torch.full((1, length), 1, dtype=torch.long)
    mask = torch.zeros(1, length, dtype=torch.bool)
    mask[0, :valid] = True
    return SimpleNamespace(base_ids=ids, valid_mask=mask)


def test_reads_every_base():
    backbone = Recorder(torch.zeros(1, 32, 4))
    BaselineEncoder(backbone, 4, reverse_complement_equivariant=False)(batch(12, 12))
    assert backbone.length == 12


def test_equivariant_halves_are_averaged_with_channel_flip():
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12).expand(1, 5, 12).clone()  # d = 6: two halves of 6 channels
    encoder = BaselineEncoder(Recorder(hidden), 6, reverse_complement_equivariant=True)
    per_base = encoder.per_base(batch(5, 5))
    expected = 0.5 * (hidden[..., :6] + hidden[..., 6:].flip(dims=[-1]))
    assert per_base.shape == (1, 5, 6) and torch.equal(per_base, expected)


def test_mean_over_real_bases_only():
    hidden = torch.zeros(1, 6, 2)
    hidden[0, :4] = 1.0  # real bases
    hidden[0, 4:] = 100.0  # padding
    pooled = BaselineEncoder(Recorder(hidden), 2, reverse_complement_equivariant=False)(batch(6, 4))
    assert torch.allclose(pooled, torch.ones(1, 2))
