"""Sequence classifier on a pretrained LoCUS encoder: the mean of the per-base vectors (each base's vector is the mean
of the segments covering it) followed by dropout and a linear layer."""
from pathlib import Path

import torch
import torch.nn as nn

from locus.config import config_from_dict
from locus.hub import load_state
from locus.models.locus_model import LoCUSModel, covering_segment_mean


class LoCUSEncoder(nn.Module):
    """The pretrained model without its masked-base head."""

    def __init__(self, model: LoCUSModel):
        super().__init__()
        self.base_local_mixer = model.base_local_mixer
        self.segment_pooler = model.segment_pooler
        self.segment_neighbor_mix = model.segment_neighbor_mix
        self.backbone = model.backbone
        object.__setattr__(self, "_model", model)  # for encode_segments; not a registered submodule
        self.output_dim = model.config.model_dim

    def forward(self, batch) -> torch.Tensor:
        """Mean over the real bases of the per-base vectors: [batch, model_dim]."""
        segments = self._model.encode_segments(batch.base_ids, batch.valid_mask, batch.segment_starts, batch.nucleotide_mask,
                                               batch.segment_mask)
        base_vectors = covering_segment_mean(segments, batch.segment_starts, batch.nucleotide_mask, batch.base_ids.shape[1])
        weights = batch.valid_mask.to(dtype=base_vectors.dtype)
        counts = weights.sum(dim=1, keepdim=True).clamp_min(1)
        return (base_vectors * weights.unsqueeze(-1)).sum(dim=1) / counts


class SequenceClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, classes: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.output_dim, classes)

    def forward(self, batch) -> torch.Tensor:
        return self.classifier(self.dropout(self.encoder(batch)))


def set_dropout(module: nn.Module, probability: float) -> None:
    """Sets the rate of every dropout layer and of the attention modules that apply dropout themselves."""
    for submodule in module.modules():
        if isinstance(submodule, nn.Dropout):
            submodule.p = probability
        elif isinstance(getattr(submodule, "dropout_p", None), float):
            submodule.dropout_p = probability


def load_pretrained(source, random_init: bool = False):
    """(LoCUS model with the pretrained weights, its config, the pretraining step). `source` is a released model name
    (locus.hub.MODELS, downloaded from the Hugging Face Hub), a directory in the Hub layout, or a checkpoint written by
    scripts/pretrain.py. With random_init the model keeps its random initialization (the pretraining ablation): only the
    architecture and segmenter of `source` are used, and the step is 0.

    Two models are initialized and the first one receives the weights. The second initialization only advances the
    random number generator, exactly as the pipeline that produced the reported results did, so a fine-tuning seed
    gives the same classifier initialization and therefore reproduces the reported runs."""
    state, config_dict, step = load_state(source)
    config = config_from_dict(config_dict)
    model = LoCUSModel(config.model)
    if random_init:
        step = 0
    else:
        model.load_state_dict(state, strict=True)
    LoCUSModel(config.model)
    return model, config, step
