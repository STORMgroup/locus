"""Public pretrained DNA language models as encoders of the LoCUS fine-tuning pipeline, so that a difference between them
and LoCUS reflects the model rather than the evaluation protocol.

Caduceus-PS (Schiff et al. 2024) and HyenaDNA (Nguyen et al. 2023) are loaded from their Hugging Face repositories
(trust_remote_code, pinned revisions) and read the per-base inputs of a SequenceBatch (base_ids, valid_mask); segment
boundaries are not used (WholeSequenceSegmenter gives the collator one segment per sequence). Both share a character
tokenizer: special tokens 0-6 ([PAD] = 4), then A, C, G, T, N = 7-11. The sequence vector is the mean of the final
hidden states over the real bases, followed by the same dropout and linear classifier as LoCUS (model.SequenceClassifier).
Caduceus-PS is reverse-complement equivariant: its 2 x d_model output channels hold the forward half and the
channel-flipped reverse-complement half, which are averaged per base (with mean pooling and a linear head this equals
CaduceusForSequenceClassification's averaging of the two halves' logits). HyenaDNA is causal, so right padding never
reaches the real bases. Caduceus needs the mamba-ssm CUDA kernels and runs on GPU only. HyenaDNA reads at most
config.max_seq_len positions (1,026 for HyenaDNA-tiny); a longer input keeps its first max_seq_len bases, as truncation
by its tokenizer would. Clean NT18 sequences (at most 1,000 bp) are never truncated; only reads lengthened by simulated
insertions can be (benchmarks/noisy_reads).

Fine-tuning recipe (scripts/finetune_baseline.py, configs/finetune/<baseline>.yaml): Caduceus-PS 1.9M and HyenaDNA-tiny
0.44M use the LoCUS-2M recipe; HyenaDNA-small 3.3M uses learning rate 3e-4 with 10% warmup, the recipe of the larger models in
our development runs. All baselines run in float32 (no autocast) with micro-batches of 32 sequences (the update is that of
the full batch)."""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from locus.alphabet import ALPHABET_SIZE, BASE_TO_ID


@dataclass(frozen=True)
class Baseline:
    repository: str
    revision: str
    family: str  # "caduceus" or "hyena"
    recipe: str  # fine-tuning recipe in configs/finetune


BASELINES = {
    "caduceus-ps-1.9m": Baseline("kuleshov-group/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3",
                                 "2de36509b76b4395a97bfa39acafeb6f17b537c5", "caduceus", "caduceus-ps-1.9m"),
    "hyenadna-tiny-0.44m": Baseline("LongSafari/hyenadna-tiny-1k-seqlen-hf", "e8c1effa8673814e257e627d2e1eda9ea5a373f6", "hyena",
                                    "hyenadna-tiny-0.44m"),
    "hyenadna-small-3.3m": Baseline("LongSafari/hyenadna-small-32k-seqlen-hf", "8fe770c78eb13fe33bf81501612faeddf4d6f331", "hyena",
                                    "hyenadna-small-3.3m"),
}

PAD_TOKEN = 4
BASE_TOKENS = {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}


class WholeSequenceSegmenter:
    """One segment spanning the whole sequence: lets the fine-tuning collator build batches for encoders that read bases."""

    name = "none"

    def bounds(self, sequence: str) -> list[tuple[int, int]]:
        return [(0, len(sequence))]


class BaselineEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, model_dim: int, reverse_complement_equivariant: bool, max_length: Optional[int] = None):
        super().__init__()
        self.backbone = backbone
        self.output_dim = model_dim
        self.reverse_complement_equivariant = reverse_complement_equivariant
        self.max_length = max_length
        lookup = torch.full((ALPHABET_SIZE,), PAD_TOKEN, dtype=torch.long)
        for base, token in BASE_TOKENS.items():
            lookup[BASE_TO_ID[base]] = token
        self.register_buffer("token_lookup", lookup, persistent=False)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def per_base(self, batch) -> torch.Tensor:
        """Final hidden state of every base read: [batch, min(length, max_length), output_dim]."""
        hidden = self.backbone(self.token_lookup[batch.base_ids[:, :self.max_length]])[0]
        if self.reverse_complement_equivariant:
            d = self.output_dim
            hidden = 0.5 * (hidden[..., :d] + hidden[..., d:].flip(dims=[-1]))
        return hidden

    def forward(self, batch) -> torch.Tensor:
        """Mean over the real bases: [batch, output_dim]."""
        hidden = self.per_base(batch)
        mask = batch.valid_mask[:, :hidden.shape[1]].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def load_baseline(name: str, path: Optional[str] = None) -> BaselineEncoder:
    """The pretrained baseline `name` (a key of BASELINES), from `path` (a local copy of its Hugging Face repository) or
    downloaded at the pinned revision."""
    from transformers import AutoModel

    if name not in BASELINES:
        raise ValueError(f"Unknown baseline {name!r}; choose one of: {', '.join(BASELINES)}")
    baseline = BASELINES[name]
    if path:
        backbone = AutoModel.from_pretrained(path, trust_remote_code=True)
    else:
        backbone = AutoModel.from_pretrained(baseline.repository, revision=baseline.revision, trust_remote_code=True)
    equivariant = baseline.family == "caduceus" and bool(getattr(backbone.config, "rcps", False))
    max_length = backbone.config.max_seq_len if baseline.family == "hyena" else None
    return BaselineEncoder(backbone, backbone.config.d_model, reverse_complement_equivariant=equivariant, max_length=max_length)
