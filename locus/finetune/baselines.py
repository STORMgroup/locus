"""Public pretrained DNA language models as encoders of the LoCUS fine-tuning pipeline, so that a difference between them
and LoCUS reflects the model rather than the evaluation protocol.

Caduceus-PS (Schiff et al. 2024) is loaded from its Hugging Face repositories (trust_remote_code, pinned revisions) and
reads the per-base inputs of a SequenceBatch (base_ids, valid_mask); segment boundaries are not used
(WholeSequenceSegmenter gives the collator one segment per sequence). Its character tokenizer has the special tokens 0-6
([PAD] = 4), then A, C, G, T, N = 7-11. The sequence vector is the mean of the final hidden states over the real bases,
followed by the same dropout and linear classifier as LoCUS (model.SequenceClassifier). Caduceus-PS is reverse-complement
equivariant: its 2 x d_model output channels hold the forward half and the channel-flipped reverse-complement half, which
are averaged per base (with mean pooling and a linear head this equals CaduceusForSequenceClassification's averaging of the
two halves' logits). Caduceus needs the mamba-ssm CUDA kernels and runs on GPU only.

Fine-tuning recipes (scripts/finetune_baseline.py, configs/finetune): configs/finetune/<baseline>.yaml is the protocol shared
with LoCUS: Caduceus-PS 0.47M and 1.9M use the LoCUS-2M recipe in float32 (no autocast).
configs/finetune/caduceus-ps-<size>-authors.yaml (supplementary runs of Caduceus-PS 0.47M and 1.9M, the sizes of LoCUS-0.5M
and LoCUS-2M) holds Caduceus's own fine-tuning settings for the Nucleotide Transformer tasks (github.com/kuleshov-group/caduceus,
configs/experiment/hg38/nucleotide_transformer.yaml and train.py with src/utils/optim_groups.add_optimizer_hooks; per task the
learning rate and batch size of its paper's Table 6): weight decay 0.1 on its parameter groups (caduceus_no_weight_decay), no
dropout, and the forward strand only in training (rc_aug false) and in validation and test (one strand, as Caduceus's code
reads it), in fp16 with loss scaling as in Caduceus's code. Every recipe uses micro-batches of 32 sequences (the update is
that of the full batch)."""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from locus.alphabet import ALPHABET_SIZE, BASE_TO_ID


@dataclass(frozen=True)
class Baseline:
    repository: str
    revision: str
    family: str  # "caduceus"
    recipe: str  # fine-tuning recipe in configs/finetune


BASELINES = {
    # the size of LoCUS-0.5M: the shared protocol on NT18 (added 2026-10-31) and Caduceus's own settings (supplementary table)
    "caduceus-ps-0.47m": Baseline("kuleshov-group/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3",
                                  "51d88e326c0fadb0dcc63efdd7345119ce672ce5", "caduceus", "caduceus-ps-0.47m"),
    "caduceus-ps-1.9m": Baseline("kuleshov-group/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3",
                                 "2de36509b76b4395a97bfa39acafeb6f17b537c5", "caduceus", "caduceus-ps-1.9m"),
}

PAD_TOKEN = 4
BASE_TOKENS = {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}


class WholeSequenceSegmenter:
    """One segment spanning the whole sequence: lets the fine-tuning collator build batches for encoders that read bases."""

    name = "none"

    def bounds(self, sequence: str) -> list[tuple[int, int]]:
        return [(0, len(sequence))]


class BaselineEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, model_dim: int, reverse_complement_equivariant: bool):
        super().__init__()
        self.backbone = backbone
        self.output_dim = model_dim
        self.reverse_complement_equivariant = reverse_complement_equivariant
        lookup = torch.full((ALPHABET_SIZE,), PAD_TOKEN, dtype=torch.long)
        for base, token in BASE_TOKENS.items():
            lookup[BASE_TO_ID[base]] = token
        self.register_buffer("token_lookup", lookup, persistent=False)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def per_base(self, batch) -> torch.Tensor:
        """Final hidden state of every base: [batch, length, output_dim]."""
        hidden = self.backbone(self.token_lookup[batch.base_ids])[0]
        if self.reverse_complement_equivariant:
            d = self.output_dim
            hidden = 0.5 * (hidden[..., :d] + hidden[..., d:].flip(dims=[-1]))
        return hidden

    def forward(self, batch) -> torch.Tensor:
        """Mean over the real bases: [batch, output_dim]."""
        hidden = self.per_base(batch)
        mask = batch.valid_mask[:, :hidden.shape[1]].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


# nn modules whose parameters Caduceus's add_optimizer_hooks exempts from weight decay (normalization_weight_decay=False)
_NO_DECAY_MODULES = (nn.Embedding, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.SyncBatchNorm, nn.InstanceNorm1d,
                     nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LayerNorm, nn.LocalResponseNorm)


def _hook_exempt(model: nn.Module) -> set:
    """ids of the parameters that add_optimizer_hooks (bias_weight_decay=False, normalization_weight_decay=False) of
    Caduceus's code exempts from weight decay: biases, parameters marked _no_weight_decay, and every parameter of
    an embedding or normalization module. Applied, as there, to the whole model including the classifier."""
    exempt = set()
    for _, module in model.named_modules():
        for name, parameter in module.named_parameters():
            if name.endswith("bias") or getattr(parameter, "_no_weight_decay", False) or isinstance(module, _NO_DECAY_MODULES):
                exempt.add(id(parameter))
    return exempt


def caduceus_no_weight_decay(model: nn.Module) -> set:
    """Caduceus's fine-tuning: add_optimizer_hooks, with Mamba's A_log and D. Mamba marks them _no_weight_decay when it builds
    them, but Hugging Face's from_pretrained replaces the parameters while loading the weights and the mark is lost, so they
    are taken from every Mamba module here."""
    exempt = _hook_exempt(model)
    for module in model.modules():
        if type(module).__name__ == "Mamba":
            exempt.update(id(parameter) for parameter in (module.A_log, module.D))
    return exempt


NO_WEIGHT_DECAY = {"caduceus": caduceus_no_weight_decay}


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
    return BaselineEncoder(backbone, backbone.config.d_model, reverse_complement_equivariant=equivariant)
