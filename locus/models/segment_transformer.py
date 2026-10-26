"""Transformer over the segment sequence, with rotary embeddings at the segment centers in base pairs, so attention sees
the genomic distance between segments rather than their ordinal positions."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from locus.models.rope import apply_rotary_pos_emb, build_rope_cache


class RoPESelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.dropout_p = dropout
        self.in_proj = nn.Linear(model_dim, 3 * model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x, cos, sin, key_padding_mask):
        batch_size, seq_len, model_dim = x.shape
        qkv = self.in_proj(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        attn_mask = None if key_padding_mask is None else (~key_padding_mask)[:, None, None, :]
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                 dropout_p=self.dropout_p if self.training else 0.0)
        return self.out_proj(context.transpose(1, 2).reshape(batch_size, seq_len, model_dim))


class RoPEEncoderLayer(nn.Module):
    """Pre-norm transformer layer."""

    def __init__(self, model_dim: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.self_attn = RoPESelfAttention(model_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.linear1 = nn.Linear(model_dim, ffn_dim)
        self.activation = nn.GELU()
        self.dropout_ffn = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, model_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, cos, sin, key_padding_mask):
        x = x + self.dropout1(self.self_attn(self.norm1(x), cos, sin, key_padding_mask))
        ffn_hidden = self.activation(self.linear1(self.norm2(x)))
        return x + self.dropout2(self.linear2(self.dropout_ffn(ffn_hidden)))


class RoPEEncoder(nn.Module):
    def __init__(self, num_layers: int, model_dim: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([RoPEEncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, cos, sin, key_padding_mask):
        for layer in self.layers:
            x = layer(x, cos, sin, key_padding_mask)
        return x


class SegmentTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_dim = config.model_dim
        self.head_dim = config.model_dim // config.attention_heads
        self.rope_theta = config.rope_theta
        self.rope_position_scale = config.rope_position_scale
        self.encoder = RoPEEncoder(config.segment_layers, config.model_dim, config.attention_heads, config.ffn_dim, config.dropout)

    def forward(self, segment_vectors: torch.Tensor, segment_mask: torch.Tensor, segment_centers: torch.Tensor) -> torch.Tensor:
        """segment_vectors [batch, segments, model_dim], segment_mask and segment_centers [batch, segments]."""
        if segment_vectors.dim() != 3 or segment_vectors.shape[-1] != self.model_dim:
            raise ValueError(f"segment_vectors must have shape [batch, segments, {self.model_dim}]")
        if segment_mask.shape != segment_vectors.shape[:2] or segment_centers.shape != segment_vectors.shape[:2]:
            raise ValueError("segment_mask and segment_centers must have shape [batch, segments]")
        segment_mask = segment_mask.bool()
        if not torch.all(segment_mask.any(dim=1)):
            raise ValueError("Each sequence must have at least one valid segment")
        real_segment_mask = segment_mask.unsqueeze(-1).to(dtype=segment_vectors.dtype)
        segment_vectors = segment_vectors * real_segment_mask
        scaled_centers = segment_centers.to(torch.float32) / self.rope_position_scale
        cos, sin = build_rope_cache(scaled_centers, self.head_dim, self.rope_theta, segment_vectors.dtype)
        x = self.encoder(segment_vectors, cos, sin, key_padding_mask=~segment_mask)
        return x * real_segment_mask
