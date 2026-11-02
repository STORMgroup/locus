"""Caduceus-PS 1.9M with its own settings trains in fp16, which can overflow: such folds are run again in bfloat16 (plan
baselines-authors-bf16), the table uses those runs for them, and a fold without any finite validation loss still finishes."""
import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import collect_results  # noqa: E402
import reproduce  # noqa: E402
from test_pipeline import _config, _genome  # noqa: E402

import locus.finetune.train as train_module  # noqa: E402
from locus.finetune.collate import SegmentCache  # noqa: E402
from locus.finetune.data import SequenceRecord, complement_indices, stratified_folds  # noqa: E402
from locus.finetune.model import LoCUSEncoder, SequenceClassifier, load_pretrained  # noqa: E402
from locus.segmentation import Segmenter  # noqa: E402
from locus.training.pretrain import train  # noqa: E402

NAN = float("nan")


def _report(validation_losses, test_mcc, nonfinite_steps=0, no_finite=False):
    history = [{"step": 10 * (i + 1), "epoch": i + 1, "validation_loss": loss, "validation_metrics": {"mcc": 0.1 * i},
                "test_metrics": {"mcc": test_mcc + 0.01 * i}} for i, loss in enumerate(validation_losses)]
    return {"history": history, "nonfinite_steps": nonfinite_steps, "no_finite_validation_loss": no_finite,
            "test_metrics": {"mcc": test_mcc}}


def _write(results, run, task, fold, report):
    path = results / run / "nt18" / f"{task}_revised" / f"fold-{fold:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report))


def test_bf16_plan_takes_only_the_overflowed_folds(tmp_path):
    results = tmp_path / "results"
    _write(results, "caduceus-ps-1.9m-authors", "H2AFZ", 0, _report([0.6, 0.5, 0.55], 0.4))                     # clean
    _write(results, "caduceus-ps-1.9m-authors", "H2AFZ", 1, _report([0.6, 0.5, 0.55], 0.4, nonfinite_steps=3))  # skipped steps
    _write(results, "caduceus-ps-1.9m-authors", "H3K27ac", 0, _report([NAN, 0.5, 0.55], 0.4))                  # one NaN epoch
    _write(results, "caduceus-ps-1.9m-authors", "H3K27ac", 2, _report([NAN, NAN], 0.3, no_finite=True))         # no finite one
    _write(results, "caduceus-ps-0.47m-authors", "H2AFZ", 0, _report([NAN, 0.5], 0.4, nonfinite_steps=9))      # 0.47M too
    arguments = argparse.Namespace(results=results, data=tmp_path / "data", runs=tmp_path / "runs", models_from=None)
    jobs = reproduce.build_plans(arguments)["baselines-authors-bf16"].stages[0]
    assert [job.name for job in jobs] == ["caduceus-ps-0.47m-authors-bf16.nt18.H2AFZ_revised.fold0",
                                          "caduceus-ps-1.9m-authors-bf16.nt18.H3K27ac_revised.fold0",
                                          "caduceus-ps-1.9m-authors-bf16.nt18.H2AFZ_revised.fold1",
                                          "caduceus-ps-1.9m-authors-bf16.nt18.H3K27ac_revised.fold2"]
    for job in jobs:
        command = [str(part) for part in job.command]
        size = "0.47m" if "0.47m" in job.name else "1.9m"
        assert command[command.index("--baseline") + 1] == f"caduceus-ps-{size}"
        assert command[command.index("--recipe") + 1].endswith(f"configs/finetune/caduceus-ps-{size}-authors-bf16.yaml")
        assert f"/caduceus-ps-{size}-authors-bf16/nt18/" in command[command.index("--output") + 1] and "--save-best" in command


def test_bf16_recipes_differ_only_in_precision():
    import yaml
    load = lambda name: yaml.safe_load((ROOT / "configs/finetune" / f"{name}.yaml").read_text())
    for size in ("0.47m", "1.9m"):
        fp16, bf16 = load(f"caduceus-ps-{size}-authors"), load(f"caduceus-ps-{size}-authors-bf16")
        assert fp16.pop("autocast") == "float16" and bf16.pop("autocast") == "bfloat16" and fp16 == bf16


def test_table_uses_the_bf16_reruns(tmp_path):
    results = tmp_path / "results"
    _write(results, "caduceus-ps-1.9m-authors", "H2AFZ", 0, _report([0.6, 0.5], 0.40))
    _write(results, "caduceus-ps-1.9m-authors", "H2AFZ", 1, _report([NAN, 0.5], 0.20, nonfinite_steps=7))
    _write(results, "caduceus-ps-1.9m-authors-bf16", "H2AFZ", 1, _report([0.6, 0.5], 0.45))
    scores, reruns = collect_results.authors_scores(results, "caduceus-ps-1.9m-authors", "H2AFZ")
    # the selected epoch is the one with the lowest finite validation loss (epoch 2: test 0.40 + 0.01; bf16 0.45 + 0.01)
    assert reruns == 1 and scores == pytest.approx([0.41, 0.46])


def test_selection_skips_non_finite_validation_losses():
    report = _report([NAN, 0.7, 0.5, NAN, 0.6], 0.3)
    assert collect_results.selected_entry(report)["epoch"] == 3
    assert collect_results.selected_entry(_report([NAN, NAN], 0.3, no_finite=True))["epoch"] in (1, 2)


def test_fold_without_finite_validation_loss_finishes(tmp_path, monkeypatch):
    """Every validation loss non-finite (as an fp16 overflow in every evaluation): the fold keeps its last weights, reports
    test metrics and is marked, instead of failing."""
    _genome(tmp_path / "genome")
    config = _config(tmp_path)
    checkpoint = train(replace(config, training=replace(config.training, max_steps=4)), tmp_path / "pretrain")
    rng = random.Random(1)
    records = [SequenceRecord(f"r{i}", "".join(rng.choice("ACGT") for _ in range(150)), i % 2, "toy", "unknown") for i in range(120)]
    validation = stratified_folds(torch.tensor([r.label for r in records]), 10, 42)[0]
    model, pretrained_config, _ = load_pretrained(checkpoint)
    classifier = SequenceClassifier(LoCUSEncoder(model), 2)
    segmenter = Segmenter(pretrained_config.data)
    original = train_module.evaluate
    validation_ids = set(validation.tolist())

    def overflowing(model, cache, indices, *args, **kwargs):
        metrics, loss = original(model, cache, indices, *args, **kwargs)
        return metrics, (NAN if set(indices) == validation_ids else loss)

    monkeypatch.setattr(train_module, "evaluate", overflowing)
    report = train_module.train_fold(classifier, SegmentCache(records, segmenter), complement_indices(len(records), validation), validation,
                                     SegmentCache(records[:20], segmenter), epochs=2, batch_size=16, learning_rate=1e-3, warmup_fraction=0.1,
                                     weight_decay=0.1, min_learning_rate_ratio=0.1, gradient_clip=1.0, seed=42, device=torch.device("cpu"))
    assert report["no_finite_validation_loss"] and report["best_epoch"] == 2 and "mcc" in report["test_metrics"]
    monkeypatch.undo()  # a normal fold is not marked
    healthy = train_module.train_fold(SequenceClassifier(LoCUSEncoder(load_pretrained(checkpoint)[0]), 2), SegmentCache(records, segmenter),
                                      complement_indices(len(records), validation), validation, SegmentCache(records[:20], segmenter),
                                      epochs=1, batch_size=16, learning_rate=1e-3, warmup_fraction=0.1, weight_decay=0.1,
                                      min_learning_rate_ratio=0.1, gradient_clip=1.0, seed=42, device=torch.device("cpu"))
    assert healthy["no_finite_validation_loss"] is False
