"""locus/finetune/baselines.py: an input longer than the baseline's maximum length keeps its first max_length bases;
shorter inputs are read unchanged."""
from types import SimpleNamespace

import torch
import torch.nn as nn

from locus.finetune.baselines import BaselineEncoder


class Recorder(nn.Module):
    """Stand-in backbone: returns one-hot-like hidden states and remembers the input length."""

    def forward(self, tokens):
        self.length = tokens.shape[1]
        return (tokens.unsqueeze(-1).float().expand(-1, -1, 4),)


def batch(length: int, valid: int) -> SimpleNamespace:
    ids = torch.full((1, length), 1, dtype=torch.long)
    mask = torch.zeros(1, length, dtype=torch.bool)
    mask[0, :valid] = True
    return SimpleNamespace(base_ids=ids, valid_mask=mask)


def test_truncation_to_max_length():
    backbone = Recorder()
    encoder = BaselineEncoder(backbone, 4, reverse_complement_equivariant=False, max_length=8)
    short = encoder(batch(6, 6))
    assert backbone.length == 6
    long = encoder(batch(12, 11))
    assert backbone.length == 8 and long.shape == short.shape
    unlimited = BaselineEncoder(Recorder(), 4, reverse_complement_equivariant=False)
    unlimited(batch(12, 11))
    assert unlimited.backbone.length == 12
