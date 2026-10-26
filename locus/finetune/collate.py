"""Fine-tuning batches: segment boundaries per record and strand, padded model inputs, and the epoch order."""
import random
from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np
import torch

from locus.alphabet import BASE_TO_ID, PAD_ID, reverse_complement
from locus.segmentation import Segmenter

# byte -> base id for upper- and lower-case bases; 255 marks anything else
_BASE_LUT = np.full(256, 255, dtype=np.int64)
for _base, _id in BASE_TO_ID.items():
    _BASE_LUT[ord(_base)] = _BASE_LUT[ord(_base.lower())] = _id


def encode_bases(sequence: str) -> np.ndarray:
    encoded = _BASE_LUT[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
    if (encoded == 255).any():
        raise ValueError(f"Invalid DNA base in {sequence[:50]}...")
    return encoded


class SegmentCache:
    """Segment boundaries of every record on each strand, computed once. The reverse strand is segmented on its own
    reverse-complemented sequence (LCP is not strand-symmetric)."""

    def __init__(self, records: Sequence, segmenter: Segmenter):
        self.records = records
        self.segmenter = segmenter
        self._bounds = {}

    def sequence(self, index: int, orientation: str) -> str:
        sequence = self.records[index].sequence
        return sequence if orientation == "forward" else reverse_complement(sequence)

    def bounds(self, index: int, orientation: str) -> np.ndarray:
        """[segments, 2] start / end array (int32, so long inputs keep a small cache: about 25 kB per 16 kb record and strand)."""
        key = (index, orientation)
        if key not in self._bounds:
            self._bounds[key] = np.asarray(self.segmenter.bounds(self.sequence(index, orientation)), dtype=np.int32).reshape(-1, 2)
        return self._bounds[key]


@dataclass(frozen=True)
class SequenceBatch:
    base_ids: torch.Tensor         # [batch, length]
    valid_mask: torch.Tensor       # [batch, length]
    segment_starts: torch.Tensor   # [batch, segments], 0 at padding
    nucleotide_mask: torch.Tensor  # [batch, segments, longest segment]
    segment_mask: torch.Tensor     # [batch, segments]
    labels: torch.Tensor           # [batch]
    indices: torch.Tensor          # [batch] record indices

    def to(self, device) -> "SequenceBatch":
        return SequenceBatch(**{item.name: getattr(self, item.name).to(device) for item in fields(self)})

    def split(self, parts: int) -> list["SequenceBatch"]:
        """`parts` consecutive slices along the batch axis (padding kept), for gradient accumulation."""
        size = len(self.labels)
        bounds = [round(size * k / parts) for k in range(parts + 1)]
        return [SequenceBatch(**{item.name: getattr(self, item.name)[a:b] for item in fields(self)})
                for a, b in zip(bounds, bounds[1:]) if b > a]


class Collator:
    """Builds batches; orientation "forward", "reverse", or "random" (one strand drawn per window, from an RNG reseeded
    by (seed, epoch) so that an epoch's strands do not depend on earlier epochs)."""

    def __init__(self, cache: SegmentCache, orientation: str, seed: int = 0):
        if orientation not in {"forward", "reverse", "random"}:
            raise ValueError(f"Unknown orientation: {orientation}")
        self.cache = cache
        self.orientation = orientation
        self.seed = seed
        self.random = random.Random(seed)
        self.last_orientations = ""

    def set_epoch(self, epoch: int) -> None:
        self.random = random.Random(self.seed * 1_000_003 + epoch)

    def __call__(self, indices: Sequence[int]) -> SequenceBatch:
        sequences, spans, chosen = [], [], []
        for index in indices:
            orientation = self.random.choice(("forward", "reverse")) if self.orientation == "random" else self.orientation
            chosen.append("F" if orientation == "forward" else "R")
            sequences.append(self.cache.sequence(index, orientation))
            spans.append(np.asarray(self.cache.bounds(index, orientation), dtype=np.int64).reshape(-1, 2))
        self.last_orientations = "".join(chosen)
        batch_size = len(indices)
        max_length = max(len(sequence) for sequence in sequences)
        max_segments = max(len(span) for span in spans)
        longest = max(int((span[:, 1] - span[:, 0]).max()) for span in spans if len(span))
        base_ids = np.full((batch_size, max_length), PAD_ID, dtype=np.int64)
        valid_mask = np.zeros((batch_size, max_length), dtype=bool)
        segment_starts = np.zeros((batch_size, max_segments), dtype=np.int64)
        nucleotide_mask = np.zeros((batch_size, max_segments, longest), dtype=bool)
        segment_mask = np.zeros((batch_size, max_segments), dtype=bool)
        offsets = np.arange(longest)
        for row, (sequence, span) in enumerate(zip(sequences, spans)):
            base_ids[row, :len(sequence)] = encode_bases(sequence)
            valid_mask[row, :len(sequence)] = True
            if not len(span):
                continue
            lengths = np.clip(np.minimum(span[:, 1], len(sequence)) - span[:, 0], 0, None)
            nucleotide_mask[row, :len(span)] = offsets[None, :] < lengths[:, None]
            segment_starts[row, :len(span)] = span[:, 0]
            segment_mask[row, :len(span)] = True
        labels = torch.tensor([self.cache.records[index].label for index in indices], dtype=torch.long)
        return SequenceBatch(base_ids=torch.from_numpy(base_ids), valid_mask=torch.from_numpy(valid_mask),
                             segment_starts=torch.from_numpy(segment_starts), nucleotide_mask=torch.from_numpy(nucleotide_mask),
                             segment_mask=torch.from_numpy(segment_mask), labels=labels,
                             indices=torch.tensor(list(indices), dtype=torch.long))


def epoch_permutation(size: int, seed: int, epoch: int) -> list[int]:
    """Training order of an epoch, a function of (seed, epoch) only."""
    generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
    return torch.randperm(size, generator=generator).tolist()
