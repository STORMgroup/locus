"""Rotary position embeddings (RoPE) with arbitrary real-valued positions."""
import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(positions: torch.Tensor, head_dim: int, base: float, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of shape [batch, sequence, head_dim] for positions [batch, sequence]; angles are computed in float32."""
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")
    half_dim = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, device=positions.device, dtype=torch.float32) / half_dim))
    freqs = positions.to(torch.float32).unsqueeze(-1) * inv_freq
    angles = torch.cat((freqs, freqs), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [batch, heads, sequence, head_dim]; cos, sin: [batch, sequence, head_dim]."""
    if x.dim() != 4:
        raise ValueError("x must have shape [batch, heads, sequence, head_dim]")
    if cos.shape != sin.shape or cos.dim() != 3:
        raise ValueError("cos and sin must have shape [batch, sequence, head_dim]")
    return (x * cos.unsqueeze(1)) + (rotate_half(x) * sin.unsqueeze(1))
