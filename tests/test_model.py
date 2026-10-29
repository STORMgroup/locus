"""Model sizes and forward-pass invariants."""
import random
from pathlib import Path

import torch

from locus.alphabet import MASK_ID
from locus.config import load_config
from locus.data.batching import build_model_batch
from locus.data.examples import SegmentedExample
from locus.models.locus_model import LoCUSModel, parameter_count
from locus.segmentation import Segment, fixed_bounds

ROOT = Path(__file__).resolve().parents[1]


def test_parameter_counts():
    expected = {"locus-0.5m": 471_556, "locus-2m": 1_737_604}
    for name, count in expected.items():
        assert parameter_count(LoCUSModel(load_config(ROOT / f"configs/pretrain/{name}.yaml").model)) == count


def _example(length, seed):
    rng = random.Random(seed)
    sequence = "".join(rng.choice("ACGT") for _ in range(length))
    segments = [Segment(sequence[a:b], a, b) for a, b in fixed_bounds(sequence, 10, 5)]
    targets = list(range(40, 50))
    return SegmentedExample(sequence, segments, targets, [MASK_ID] * len(targets), ["mask"] * len(targets), length - 64)


def test_padding_and_batch_invariance():
    torch.manual_seed(0)
    model = LoCUSModel(load_config(ROOT / "configs/pretrain/locus-2m.yaml").model).eval()
    examples = [_example(300, 1), _example(180, 2), _example(257, 3)]
    with torch.no_grad():
        batch = build_model_batch(examples)
        together = model(batch.base_ids, batch.valid_mask, batch.segment_starts, batch.nucleotide_mask, batch.segment_mask, batch.target_positions)
        for index, example in enumerate(examples):
            alone = build_model_batch([example])
            logits = model(alone.base_ids, alone.valid_mask, alone.segment_starts, alone.nucleotide_mask, alone.segment_mask, alone.target_positions)
            assert torch.allclose(logits[0], together[index, :logits.shape[1]], atol=1e-5)
