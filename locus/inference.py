"""Embeddings from a pretrained LoCUS model: each base's vector is the mean of the contextual vectors of the segments
covering it; a sequence's vector is the mean over its bases."""
from pathlib import Path

import numpy as np
import torch

from locus.config import with_paths_relative_to
from locus.finetune.collate import Collator, SegmentCache
from locus.finetune.data import SequenceRecord
from locus.models.locus_model import covering_segment_mean
from locus.segmentation import Segmenter

ROOT = Path(__file__).resolve().parents[1]


class Pretrained:
    def __init__(self, model, config, step: int, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.config = config
        self.step = step
        self.device = torch.device(device)
        self.segmenter = Segmenter(config.data)

    @torch.no_grad()
    def _base_vectors(self, sequences, orientation: str, batch_size: int) -> list:
        records = [SequenceRecord(name=str(i), sequence=s.upper(), label=0, task="", chromosome="") for i, s in enumerate(sequences)]
        collate = Collator(SegmentCache(records, self.segmenter), orientation)
        out = []
        for start in range(0, len(records), batch_size):
            batch = collate(list(range(start, min(start + batch_size, len(records))))).to(self.device)
            segments = self.model.encode_segments(batch.base_ids, batch.valid_mask, batch.segment_starts, batch.nucleotide_mask, batch.segment_mask)
            vectors = covering_segment_mean(segments, batch.segment_starts, batch.nucleotide_mask, batch.base_ids.shape[1]).float().cpu().numpy()
            lengths = batch.valid_mask.sum(dim=1).tolist()
            out.extend(vectors[row, :length] for row, length in enumerate(lengths))
        return out

    def embed(self, sequences, per_base: bool = False, both_strands: bool = True, batch_size: int = 32):
        """Sequence vectors [sequences, width], or with per_base=True a list of [length, width] arrays. With both_strands
        the reverse strand is encoded as well (segmented on its own) and its per-base vectors, mapped back to forward
        positions, are averaged with the forward ones."""
        sequences = list(sequences)
        forward = self._base_vectors(sequences, "forward", batch_size)
        if both_strands:
            reverse = self._base_vectors(sequences, "reverse", batch_size)
            forward = [0.5 * (f + r[::-1]) for f, r in zip(forward, reverse)]
        if per_base:
            return forward
        return np.stack([vectors.mean(axis=0) for vectors in forward])


def load(source: str = "locus-2m", device: str = "cpu") -> Pretrained:
    """A released model by name (locus.hub.MODELS), a Hub-layout directory, or a checkpoint of scripts/pretrain.py."""
    from locus.finetune.model import load_pretrained

    model, config, step = load_pretrained(source)
    return Pretrained(model, with_paths_relative_to(config, ROOT), step, device=device)
