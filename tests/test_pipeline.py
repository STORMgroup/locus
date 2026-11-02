"""End-to-end smoke test on a small synthetic genome: a few pretraining steps, resuming, and a short fine-tuning run."""
import gzip
import json
import random
from dataclasses import replace

import torch

from locus.config import Config, DataConfig, ModelConfig, TrainingConfig
from locus.finetune.collate import SegmentCache
from locus.finetune.data import SequenceRecord, complement_indices, stratified_folds
from locus.finetune.model import LoCUSEncoder, SequenceClassifier, load_pretrained
from locus.finetune.train import evaluation_logits, predict, scores, train_fold
from locus.segmentation import Segmenter
from locus.training.pretrain import train


def _genome(root, windows=64, length=512):
    rng = random.Random(0)
    (root / "shards").mkdir(parents=True)
    (root / "manifests").mkdir()
    rows = ["read_id\tsource\tshard"]
    with gzip.open(root / "shards/shard-0.fasta.gz", "wt") as handle:
        for index in range(windows):
            read_id = f"window{index:03d}"
            handle.write(f">{read_id} source=chr3\n{''.join(rng.choice('ACGT') for _ in range(length))}\n")
            rows.append(f"{read_id}\tchr3\tshards/shard-0.fasta.gz")
    (root / "manifests/reads.tsv").write_text("\n".join(rows) + "\n")
    (root / "_SUCCESS").write_text("")


def _config(root):
    return Config(model=ModelConfig(model_dim=32, attention_heads=2, ffn_dim=64, segment_layers=2),
                  data=DataConfig(genome_windows=str(root / "genome"), context_length=512, min_context_length=128, train_sources=("chr3",),
                                  segmenter="fixed", example_workers=1),
                  training=TrainingConfig(batch_size=4, max_steps=12, warmup_steps=2, device="cpu", checkpoint_every=4,
                                          keep_checkpoint_every=0, shuffle_buffer_size=16, log_every=100))


def test_pretraining_resumes_exactly(tmp_path):
    _genome(tmp_path / "genome")
    config = _config(tmp_path)
    train(config, tmp_path / "full")
    train(config, tmp_path / "resumed", stop_after_step=4)
    train(config, tmp_path / "resumed", resume=tmp_path / "resumed/checkpoints/step-00000004.pt")
    full = [json.loads(line) for line in open(tmp_path / "full/train_metrics.jsonl")]
    resumed = [json.loads(line) for line in open(tmp_path / "resumed/train_metrics.jsonl")]
    assert [(r["step"], r["loss"], r["raw_bases"]) for r in full] == [(r["step"], r["loss"], r["raw_bases"]) for r in resumed]
    a = torch.load(tmp_path / "full/checkpoints/final.pt", weights_only=False)["model_state"]
    b = torch.load(tmp_path / "resumed/checkpoints/final.pt", weights_only=False)["model_state"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_finetuning_runs(tmp_path):
    _genome(tmp_path / "genome")
    config = _config(tmp_path)
    checkpoint = train(replace(config, training=replace(config.training, max_steps=4)), tmp_path / "pretrain")
    rng = random.Random(1)
    records = [SequenceRecord(f"r{i}", "".join(rng.choice("ACGT") for _ in range(150)), i % 2, "toy", "unknown") for i in range(120)]
    labels = torch.tensor([r.label for r in records])
    validation = stratified_folds(labels, 10, 42)[0]
    model, pretrained_config, _ = load_pretrained(checkpoint)
    classifier = SequenceClassifier(LoCUSEncoder(model), 2)
    segmenter = Segmenter(pretrained_config.data)
    report = train_fold(classifier, SegmentCache(records, segmenter), complement_indices(len(records), validation), validation,
                        SegmentCache(records[:20], segmenter), epochs=2, batch_size=16, learning_rate=1e-3, warmup_fraction=0.1,
                        weight_decay=0.1, min_learning_rate_ratio=0.1, gradient_clip=1.0, seed=42, device=torch.device("cpu"), micro_batches=2)
    assert len(report["history"]) == 2 and report["best_epoch"] in (1, 2)
    assert all(entry["every_training_window_once"] for entry in report["data_history"])


def test_forward_strand_finetuning(tmp_path):
    """A recipe that trains on the forward strand only (the baselines' rc_aug false) can be validated and tested on the
    forward strand only: no reverse-complement window in training, and the reported test score is that of the forward
    logits of the selected weights, not of the strand average."""
    _genome(tmp_path / "genome")
    config = _config(tmp_path)
    checkpoint = train(replace(config, training=replace(config.training, max_steps=4)), tmp_path / "pretrain")
    rng = random.Random(1)
    records = [SequenceRecord(f"r{i}", "".join(rng.choice("ACGT") for _ in range(150)), i % 2, "toy", "unknown") for i in range(120)]
    validation = stratified_folds(torch.tensor([r.label for r in records]), 10, 42)[0]
    model, pretrained_config, _ = load_pretrained(checkpoint)
    classifier = SequenceClassifier(LoCUSEncoder(model), 2)
    segmenter = Segmenter(pretrained_config.data)
    test_cache = SegmentCache(records[:20], segmenter)
    report = train_fold(classifier, SegmentCache(records, segmenter), complement_indices(len(records), validation), validation, test_cache,
                        epochs=2, batch_size=16, learning_rate=1e-3, warmup_fraction=0.1, weight_decay=0.1, min_learning_rate_ratio=0.1,
                        gradient_clip=1.0, seed=42, device=torch.device("cpu"), train_orientation="forward", evaluation_orientation="forward")
    assert all(entry["reverse_complement_fraction"] == 0 for entry in report["data_history"])
    indices = list(range(len(test_cache.records)))
    forward = predict(classifier, test_cache, indices, "forward", 16, torch.device("cpu"), torch.float32)
    assert all((a == b).all() for a, b in zip(evaluation_logits(classifier, test_cache, indices, 16, torch.device("cpu"), torch.float32, "forward"), forward))
    metrics, loss = scores(*forward)
    assert report["test_metrics"] == metrics and report["test_loss"] == loss
    averaged, _ = evaluation_logits(classifier, test_cache, indices, 16, torch.device("cpu"), torch.float32)
    assert not (averaged == forward[0]).all()
