"""Hugging Face Hub layout: export, loading by name from a local copy, weights identity, and embeddings."""
import json

import numpy as np
import torch

import locus
from locus.config import GRCH38_CHROMOSOMES, Config, DataConfig, ModelConfig
from locus.finetune.model import load_pretrained
from locus.hub import MODELS, SPLIT_MODELS, export, weights_sha256
from locus.models.locus_model import LoCUSModel


def _checkpoint(tmp_path):
    torch.manual_seed(0)
    config = Config(model=ModelConfig(model_dim=32, attention_heads=2, ffn_dim=64, segment_layers=2),
                    data=DataConfig(segmenter="fixed"))
    path = tmp_path / "final.pt"
    torch.save({"step": 7, "model_state": LoCUSModel(config.model).state_dict(), "config": config.to_dict()}, path)
    return path


def test_export_and_load_by_name(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path)
    export(checkpoint, tmp_path / "hub" / "locus-2m", "locus-2m")
    monkeypatch.setenv("LOCUS_HUB_DIR", str(tmp_path / "hub"))
    a, config_a, step_a = load_pretrained(checkpoint)
    b, config_b, step_b = load_pretrained("locus-2m")
    assert step_a == step_b == 7 and config_a == config_b
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert weights_sha256(checkpoint) == weights_sha256("locus-2m")


def test_embed(tmp_path, monkeypatch):
    export(_checkpoint(tmp_path), tmp_path / "hub" / "locus-2m", "locus-2m")
    monkeypatch.setenv("LOCUS_HUB_DIR", str(tmp_path / "hub"))
    model = locus.load("locus-2m")
    sequences = ["ACGTTGCA" * 30, "GGGATCCA" * 12]
    vectors = model.embed(sequences)
    assert vectors.shape == (2, 32) and np.isfinite(vectors).all()
    per_base = model.embed(sequences, per_base=True)
    assert [v.shape for v in per_base] == [(240, 32), (96, 32)]
    np.testing.assert_allclose(per_base[0].mean(axis=0), vectors[0], rtol=1e-5, atol=1e-6)
    forward = model.embed(sequences, both_strands=False)
    assert not np.allclose(forward, vectors)


def test_export_records_repository_paths_and_chromosomes(tmp_path):
    """A model exported from a checkpoint trained elsewhere carries repository-relative paths and its own chromosomes."""
    checkpoint = _checkpoint(tmp_path)
    state = torch.load(checkpoint, weights_only=False)
    state["config"]["data"].update(genome_windows=str(tmp_path / "shared/data/GRCh38.p14.acgt.w4096.s4096"),
                                   train_sources=[GRCH38_CHROMOSOMES[str(c)] for c in list(range(1, 20)) + [22]])
    torch.save(state, checkpoint)
    directory = export(checkpoint, tmp_path / "hub" / "locus-2m-no-chr20-21", "locus-2m-no-chr20-21")
    assert json.loads((directory / "config.json").read_text())["config"]["data"]["genome_windows"] == "data/GRCh38.p14.acgt.w4096.s4096"
    assert "chromosomes 1-19 and 22" in (directory / "README.md").read_text()
    assert set(SPLIT_MODELS) <= set(MODELS)
