"""Base-level windowed self-attention: each base attends to the bases at most `window_radius` positions away.

The sequence is split into blocks of `window_radius` bases and every query block attends to its own block and its two
neighbours (3 x window_radius keys), with an exact |i - j| <= window_radius mask inside that context. The cost is
O(N x window_radius) and the result equals dense attention with the same mask. Rotary embeddings use absolute base
positions and are applied before blocking."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from locus.alphabet import ALPHABET_SIZE, PAD_ID
from locus.models.rope import apply_rotary_pos_emb, build_rope_cache


class BlockedRoPESelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float, window_radius: int):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.dropout_p = dropout
        self.window_radius = window_radius
        self.block_size = window_radius
        self.in_proj = nn.Linear(model_dim, 3 * model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    @staticmethod
    def _block_context(x: torch.Tensor, num_blocks: int, pad_value: float = 0.0) -> torch.Tensor:
        """[..., num_blocks, block_size, F] -> [..., num_blocks, 3 x block_size, F]: previous, own and next block.
        The missing neighbour at either end is filled with `pad_value` (True for the key-padding mask, so it is
        never attended)."""
        padded = F.pad(x, (0, 0, 0, 0, 1, 1), value=pad_value)
        return torch.cat((padded[..., 0:num_blocks, :, :], padded[..., 1:num_blocks + 1, :, :],
                          padded[..., 2:num_blocks + 2, :, :]), dim=-2)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, model_dim = x.shape
        block_size = self.block_size
        pad_len = (-seq_len) % block_size
        num_blocks = (seq_len + pad_len) // block_size

        qkv = self.in_proj(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            key_padding_mask = F.pad(key_padding_mask, (0, pad_len), value=True)

        q_blocks = q.reshape(batch_size, self.num_heads, num_blocks, block_size, self.head_dim)
        k_context = self._block_context(k.reshape(batch_size, self.num_heads, num_blocks, block_size, self.head_dim), num_blocks)
        v_context = self._block_context(v.reshape(batch_size, self.num_heads, num_blocks, block_size, self.head_dim), num_blocks)
        scores = torch.matmul(q_blocks, k_context.transpose(-1, -2)) / (self.head_dim ** 0.5)

        positions = torch.arange(num_blocks * block_size, device=x.device).reshape(1, num_blocks, block_size)
        position_context = self._block_context(positions.unsqueeze(1).unsqueeze(-1), num_blocks).squeeze(1).squeeze(-1)
        outside_radius = (positions.unsqueeze(-1) - position_context.unsqueeze(-2)).abs() > self.window_radius
        key_invalid = key_padding_mask.reshape(batch_size, num_blocks, block_size)
        key_invalid_context = self._block_context(key_invalid.unsqueeze(1).unsqueeze(-1), num_blocks,
                                                  pad_value=True).squeeze(1).squeeze(-1)
        invalid = outside_radius | key_invalid_context.unsqueeze(-2)
        scores = scores.masked_fill(invalid.unsqueeze(1), float("-inf"))

        # A padding query has no valid key; its NaN softmax row is set to zero.
        weights = torch.softmax(scores.float(), dim=-1)
        weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)
        weights = F.dropout(weights.to(v_context.dtype), p=self.dropout_p, training=self.training)

        context = torch.matmul(weights, v_context).reshape(batch_size, self.num_heads, num_blocks * block_size, self.head_dim)
        context = context[:, :, :seq_len, :].transpose(1, 2).reshape(batch_size, seq_len, model_dim)
        return self.out_proj(context)


class BlockedRoPEEncoderLayer(nn.Module):
    """Pre-norm transformer layer with windowed attention."""

    def __init__(self, model_dim: int, num_heads: int, ffn_dim: int, dropout: float, window_radius: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.self_attn = BlockedRoPESelfAttention(model_dim, num_heads, dropout, window_radius)
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


class BaseLocalEncoder(nn.Module):
    """Embeds the bases of the whole input and contextualizes each one with windowed attention, before segmentation."""

    def __init__(self, config):
        super().__init__()
        self.head_dim = config.model_dim // config.attention_heads
        self.rope_theta = config.rope_theta
        self.embedding = nn.Embedding(ALPHABET_SIZE, config.base_embedding_dim, padding_idx=PAD_ID)
        self.input_projection = nn.Linear(config.base_embedding_dim, config.model_dim)
        self.layers = nn.ModuleList([
            BlockedRoPEEncoderLayer(config.model_dim, config.attention_heads, config.ffn_dim, config.dropout, radius)
            for radius in config.base_attention_radii
        ])

    def forward(self, base_ids: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """base_ids, valid_mask: [batch, length] -> base vectors [batch, length, model_dim] (float32, zero at padding)."""
        if base_ids.dim() != 2:
            raise ValueError("base_ids must have shape [batch, length]")
        batch_size, seq_len = base_ids.shape
        valid_mask = valid_mask.bool()
        x = self.input_projection(self.embedding(base_ids))
        positions = torch.arange(seq_len, device=base_ids.device).unsqueeze(0).expand(batch_size, -1)
        cos, sin = build_rope_cache(positions, self.head_dim, self.rope_theta, x.dtype)
        key_padding_mask = ~valid_mask
        zeros = torch.zeros_like(x)
        validity = valid_mask.unsqueeze(-1)
        x = torch.where(validity, x, zeros)
        for layer in self.layers:
            x = layer(x, cos, sin, key_padding_mask=key_padding_mask)
            x = torch.where(validity, x, zeros)
        return x.to(torch.float32)
