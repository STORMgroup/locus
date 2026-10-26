"""Collates pretraining examples into padded model inputs."""
from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np
import torch

from locus.alphabet import BASE_TO_ID, MASK_ID, OVERLAP_MASK_ID, PAD_ID

# ASCII byte -> base id (-1 for anything that is not A/C/G/T/N).
_BASE_LOOKUP = np.full(256, -1, dtype=np.int64)
for _base, _base_id in BASE_TO_ID.items():
    _BASE_LOOKUP[ord(_base)] = _base_id


@dataclass(frozen=True)
class ModelBatch:
    base_ids: torch.Tensor          # [batch, length] input bases; PAD where no segment covers a base
    valid_mask: torch.Tensor        # [batch, length]
    segment_starts: torch.Tensor    # [batch, segments], -1 at padding
    nucleotide_mask: torch.Tensor   # [batch, segments, longest segment]
    segment_mask: torch.Tensor      # [batch, segments]
    target_positions: torch.Tensor  # [batch, targets]
    target_mask: torch.Tensor       # [batch, targets]
    labels: torch.Tensor            # [batch, targets], -100 at padding

    def to(self, device) -> "ModelBatch":
        return ModelBatch(**{item.name: getattr(self, item.name).to(device) for item in fields(self)})


def build_model_batch(examples: Sequence) -> ModelBatch:
    if not examples:
        raise ValueError("Cannot build a batch without examples")
    if any(len(example.target_positions) == 0 for example in examples):
        raise ValueError("Every example must contain at least one target")
    batch_size = len(examples)
    target_count = max(len(example.target_positions) for example in examples)
    max_segments = max(len(example.segments) for example in examples)
    longest = max(len(item.dna) for example in examples for item in example.segments)
    max_length = max(len(example.sequence) for example in examples)

    nucleotide_mask = torch.zeros((batch_size, max_segments, longest), dtype=torch.bool)
    segment_starts = torch.full((batch_size, max_segments), -1, dtype=torch.long)
    segment_mask = torch.zeros((batch_size, max_segments), dtype=torch.bool)
    target_positions = torch.zeros((batch_size, target_count), dtype=torch.long)
    target_mask = torch.zeros((batch_size, target_count), dtype=torch.bool)
    labels = torch.full((batch_size, target_count), -100, dtype=torch.long)
    base_ids = torch.full((batch_size, max_length), PAD_ID, dtype=torch.long)
    valid_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)

    offsets = np.arange(longest)
    for index, example in enumerate(examples):
        sequence = example.sequence.upper()
        codes = _BASE_LOOKUP[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
        if (codes < 0).any():
            raise ValueError("Invalid DNA base in example")
        positions = np.asarray(example.target_positions, dtype=np.int64)
        if len(set(example.target_positions)) != len(positions):
            raise ValueError("Target positions must be unique within an example")
        if ((positions < 0) | (positions >= len(sequence))).any():
            raise ValueError("Target position outside the example")
        target_codes = codes[positions]
        if ((target_codes < 0) | (target_codes > 3)).any():
            raise ValueError("Target bases must be A/C/G/T")
        corrupted = np.asarray(example.corrupted_target_ids, dtype=np.int64)
        if len(corrupted) != len(positions) or not np.isin(corrupted, [0, 1, 2, 3, MASK_ID, OVERLAP_MASK_ID]).all():
            raise ValueError("Corrupted target ids must align with the targets and be A/C/G/T, MASK or OVERLAP_MASK")

        starts = np.asarray([item.start for item in example.segments], dtype=np.int64)
        ends = np.asarray([item.end for item in example.segments], dtype=np.int64)
        if not all(item.is_valid(sequence) for item in example.segments):
            raise ValueError("Invalid segment in example")
        valid = offsets[None, :] < (ends - starts)[:, None]
        covered = np.zeros(len(sequence), dtype=bool)
        for start, end in zip(starts, ends):
            covered[start:end] = True
        if not covered[positions].all():
            raise ValueError("A target position is not covered by any segment")
        flat_codes = np.where(covered, codes, PAD_ID)
        flat_codes[positions] = corrupted

        count = len(example.segments)
        nucleotide_mask[index, :count] = torch.from_numpy(valid)
        segment_starts[index, :count] = torch.from_numpy(starts)
        segment_mask[index, :count] = True
        target_positions[index, :len(positions)] = torch.from_numpy(positions)
        target_mask[index, :len(positions)] = True
        labels[index, :len(positions)] = torch.from_numpy(target_codes)
        base_ids[index, :len(sequence)] = torch.from_numpy(flat_codes)
        valid_mask[index, :len(sequence)] = True

    return ModelBatch(base_ids=base_ids, valid_mask=valid_mask, segment_starts=segment_starts, nucleotide_mask=nucleotide_mask,
                      segment_mask=segment_mask, target_positions=target_positions, target_mask=target_mask, labels=labels)
