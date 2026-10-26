"""LoCUS: base-level local attention, pooling into segments, a transformer over segments, and per-base read-out as the
mean over the segments covering each base.

Inputs of every forward pass (see locus.data.batching):
    base_ids        [batch, length]  input bases (masked bases carry MASK_ID or OVERLAP_MASK_ID)
    valid_mask      [batch, length]  real bases (False at padding)
    segment_starts  [batch, segments]  start of each segment (-1 at padding)
    nucleotide_mask [batch, segments, longest segment]  which offsets of each segment are real bases
    segment_mask    [batch, segments]  real segments
"""
import math

import torch
import torch.nn as nn

from locus.alphabet import OUTPUT_CLASSES
from locus.models.local_attention import BaseLocalEncoder
from locus.models.segment_transformer import SegmentTransformer

# Above this many [batch x segments x longest segment] entries, segments are pooled in length buckets instead of one
# padded gather (same result, bounded memory for inputs that contain a very long segment).
POOL_DENSE_LIMIT = 8_000_000
POOL_BUCKET_EDGES = (16, 32, 64, 128, 256, 512)


class SegmentPooler(nn.Module):
    """Attention pooling of a segment's base vectors with one learned query, plus an embedding of the segment length."""

    def __init__(self, config):
        super().__init__()
        self.model_dim = config.model_dim
        self.max_segment_length = config.max_segment_length
        self.length_embedding = nn.Embedding(config.max_segment_length + 1, config.model_dim)
        self.pool_query = nn.Parameter(torch.zeros(config.model_dim))
        nn.init.normal_(self.pool_query, std=0.02)

    def forward(self, base_vectors: torch.Tensor, nucleotide_mask: torch.Tensor) -> torch.Tensor:
        """base_vectors [segments, length, model_dim], nucleotide_mask [segments, length] -> [segments, model_dim]."""
        if base_vectors.dim() != 3:
            raise ValueError("base_vectors must have shape [segments, length, model_dim]")
        with torch.autocast(device_type=base_vectors.device.type, enabled=False):
            features = base_vectors.float()
            query = self.pool_query.float()
            scores = torch.einsum("sld,d->sl", features, query) * (1.0 / math.sqrt(self.model_dim))
            scores = scores.masked_fill(~nucleotide_mask, float("-inf"))
            weights = torch.softmax(scores, dim=1)
            pooled = torch.einsum("sl,sld->sd", weights, features)
        lengths = nucleotide_mask.sum(dim=1).clamp(max=self.max_segment_length)
        return pooled + self.length_embedding(lengths)


class SegmentNeighborMix(nn.Module):
    """Residual depthwise convolution (kernel 3) over the segment axis: each segment sees its two neighbours before the
    segment transformer."""

    def __init__(self, model_dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(model_dim, model_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=model_dim)
        self.activation = nn.GELU()

    def forward(self, segment_vectors: torch.Tensor, segment_mask: torch.Tensor) -> torch.Tensor:
        real_mask = segment_mask.unsqueeze(-1).to(dtype=segment_vectors.dtype)
        segment_vectors = segment_vectors * real_mask
        x = self.activation(self.conv(segment_vectors.transpose(1, 2))).transpose(1, 2) * real_mask
        return (segment_vectors + x) * real_mask


def covering_segment_mean(segment_vectors: torch.Tensor, segment_starts: torch.Tensor, nucleotide_mask: torch.Tensor,
                          length: int) -> torch.Tensor:
    """Per-base vectors [batch, length, model_dim]: each base gets the mean of every segment that covers it (zero when
    no segment does). The sums are taken in float32."""
    batch_size, _, model_dim = segment_vectors.shape
    b, s, offset = nucleotide_mask.bool().nonzero(as_tuple=True)
    position = segment_starts[b, s] + offset
    keep = (position >= 0) & (position < length)
    b, s, position = b[keep], s[keep], position[keep]
    flat = b * length + position
    vectors = segment_vectors[b, s].float()
    summed = torch.zeros(batch_size * length, model_dim, device=vectors.device)
    summed.index_add_(0, flat, vectors)
    count = torch.zeros(batch_size * length, device=vectors.device)
    count.index_add_(0, flat, torch.ones_like(vectors[:, 0]))
    mean = summed / count.clamp(min=1).unsqueeze(-1)
    return mean.to(segment_vectors.dtype).reshape(batch_size, length, model_dim)


class SegmentDecoder(nn.Module):
    """Pretraining head: predicts each target base from the segments covering it. For every covering segment,
    LayerNorm(GELU(Linear(segment vector) + embedding of the base's offset in the segment)) -> 4-way logits; the logits
    are averaged over the covering segments. The offset is used only here, never in a representation."""

    def __init__(self, config):
        super().__init__()
        self.max_segment_length = config.max_segment_length
        self.position = nn.Embedding(config.max_segment_length, config.model_dim)
        self.project = nn.Linear(config.model_dim, config.model_dim)
        self.norm = nn.LayerNorm(config.model_dim)
        self.classifier = nn.Linear(config.model_dim, OUTPUT_CLASSES)
        self.activation = nn.GELU()

    def forward(self, segment_vectors, segment_starts, nucleotide_mask, target_positions, length) -> torch.Tensor:
        """Logits [batch, targets, 4] (zero for a target that no segment covers)."""
        batch_size, target_count = target_positions.shape
        device = segment_vectors.device
        slot = torch.full((batch_size, length), -1, dtype=torch.long, device=device)
        valid = (target_positions >= 0) & (target_positions < length)
        tb, tj = valid.nonzero(as_tuple=True)
        slot[tb, target_positions[tb, tj]] = tj
        b, s, offset = nucleotide_mask.bool().nonzero(as_tuple=True)
        position = segment_starts[b, s] + offset
        keep = (position >= 0) & (position < length)
        b, s, offset, position = b[keep], s[keep], offset[keep], position[keep]
        j = slot[b, position]
        keep = j >= 0
        b, s, offset, j = b[keep], s[keep], offset[keep], j[keep]
        hidden = self.activation(self.project(segment_vectors[b, s]) + self.position(offset.clamp(max=self.max_segment_length - 1)))
        logits = self.classifier(self.norm(hidden)).float()
        flat = b * target_count + j
        summed = torch.zeros(batch_size * target_count, logits.shape[-1], device=device)
        summed.index_add_(0, flat, logits)
        count = torch.zeros(batch_size * target_count, device=device)
        count.index_add_(0, flat, torch.ones_like(logits[:, 0]))
        return (summed / count.clamp(min=1).unsqueeze(-1)).reshape(batch_size, target_count, -1)


class LoCUSModel(nn.Module):
    def __init__(self, config):
        """config: a ModelConfig (locus.config)."""
        super().__init__()
        self.config = config
        self.base_local_mixer = BaseLocalEncoder(config)
        self.segment_pooler = SegmentPooler(config)
        self.segment_neighbor_mix = SegmentNeighborMix(config.model_dim)
        self.backbone = SegmentTransformer(config)
        self.segment_decoder = SegmentDecoder(config)

    def encode_segments(self, base_ids, valid_mask, segment_starts, nucleotide_mask, segment_mask) -> torch.Tensor:
        """Contextual segment vectors [batch, segments, model_dim]."""
        if base_ids.dim() != 2:
            raise ValueError("base_ids must have shape [batch, length]")
        batch_size, length = base_ids.shape
        _, segment_count, longest = nucleotide_mask.shape
        model_dim = self.config.model_dim
        base_vectors = self.base_local_mixer(base_ids, valid_mask)

        segment_mask = segment_mask.bool()
        if batch_size * segment_count * longest <= POOL_DENSE_LIMIT:
            offsets = torch.arange(longest, device=base_ids.device)
            absolute = (segment_starts.clamp(min=0).unsqueeze(-1) + offsets).clamp(max=length - 1)
            gathered = torch.gather(base_vectors, 1, absolute.reshape(batch_size, -1, 1).expand(-1, -1, model_dim))
            gathered = gathered.reshape(batch_size, segment_count, longest, model_dim)
            segment_vectors = torch.zeros(batch_size, segment_count, model_dim, device=base_ids.device, dtype=gathered.dtype)
            if torch.any(segment_mask):
                segment_vectors[segment_mask] = self.segment_pooler(gathered[segment_mask], nucleotide_mask[segment_mask])
        else:
            segment_vectors = self._pool_segments_bucketed(base_vectors, segment_starts, nucleotide_mask, segment_mask, length)
        segment_vectors = self.segment_neighbor_mix(segment_vectors, segment_mask)

        segment_ends = segment_starts + nucleotide_mask.sum(dim=-1)
        raw_centers = (segment_starts.to(torch.float32) + segment_ends.to(torch.float32)) / 2.0
        segment_centers = torch.where(segment_mask, raw_centers, torch.zeros_like(raw_centers))
        return self.backbone(segment_vectors, segment_mask, segment_centers)

    def encode(self, base_ids, valid_mask, segment_starts, nucleotide_mask, segment_mask) -> tuple[torch.Tensor, torch.Tensor]:
        """(per-base vectors [batch, length, model_dim], contextual segment vectors [batch, segments, model_dim]); a
        base's vector is the mean of the contextual vectors of the segments covering it."""
        segments = self.encode_segments(base_ids, valid_mask, segment_starts, nucleotide_mask, segment_mask)
        return covering_segment_mean(segments, segment_starts, nucleotide_mask, base_ids.shape[1]), segments

    def forward(self, base_ids, valid_mask, segment_starts, nucleotide_mask, segment_mask, target_positions) -> torch.Tensor:
        """Masked-base logits [batch, targets, 4]."""
        segments = self.encode_segments(base_ids, valid_mask, segment_starts, nucleotide_mask, segment_mask)
        return self.segment_decoder(segments, segment_starts, nucleotide_mask, target_positions, base_ids.shape[1])

    def _pool_segments_bucketed(self, base_vectors, segment_starts, nucleotide_mask, segment_mask, length):
        batch_size, segment_count, _ = nucleotide_mask.shape
        segment_vectors = torch.zeros(batch_size, segment_count, self.config.model_dim,
                                      device=base_vectors.device, dtype=base_vectors.dtype)
        b_index, s_index = segment_mask.nonzero(as_tuple=True)
        lengths = nucleotide_mask.sum(dim=-1)[b_index, s_index]
        lower = 0
        for upper in POOL_BUCKET_EDGES[:-1] + (int(lengths.max()) if lengths.numel() else 0,):
            chosen = (lengths > lower) & (lengths <= upper) if lower else lengths <= upper
            lower = max(upper, lower)
            if not torch.any(chosen):
                continue
            bb, ss = b_index[chosen], s_index[chosen]
            width = int(lengths[chosen].max())
            offsets = torch.arange(width, device=base_vectors.device)
            positions = (segment_starts[bb, ss].clamp(min=0).unsqueeze(-1) + offsets).clamp(max=length - 1)
            segment_vectors[bb, ss] = self.segment_pooler(base_vectors[bb.unsqueeze(-1), positions], nucleotide_mask[bb, ss, :width])
        return segment_vectors


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
