"""Fine-tune a pretrained LoCUS model on one validation fold of a benchmark task and score the test set.

    python scripts/finetune.py --pretrained locus-2m --recipe configs/finetune/locus-2m.yaml \\
        --data-root data/benchmarks/nt18 --task H3K27ac_revised --fold 0 --output results/locus-2m/nt18/H3K27ac_revised/fold-00.json

--pretrained takes a released model name (downloaded on first use, locus/hub.py), a directory in that layout, or a
checkpoint of scripts/pretrain.py (runs/<model>/checkpoints/final.pt).

The task directory holds train.jsonl.gz and test.jsonl.gz (benchmarks/nt18/prepare.py, benchmarks/genomic_benchmarks/prepare.py).
Fold k validates on fold k of a stratified 10-fold split of the training set (seed 42) and trains with seed 42 + k.
--micro-batches splits each batch for gradient accumulation (same update, less memory). --extra-test NAME=DIR scores
another test set (for example one with simulated sequencing errors) at every epoch; it is never used for selection.
--random-init fine-tunes the architecture of --pretrained from random initialization, without its weights (the
pretraining ablation).
A fold that is interrupted resumes from its last finished epoch when started again with the same arguments."""
import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locus.config import with_paths_relative_to  # noqa: E402
from locus.finetune.collate import SegmentCache  # noqa: E402
from locus.finetune.data import complement_indices, get_task, load_records, sha256_file, stratified_folds  # noqa: E402
from locus.finetune.model import LoCUSEncoder, SequenceClassifier, load_pretrained, set_dropout  # noqa: E402
from locus.finetune.train import seed_everything, train_fold  # noqa: E402
from locus.hub import weights_sha256  # noqa: E402
from locus.segmentation import Segmenter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrained", required=True,
                        help="released model name (e.g. locus-2m, downloaded from the Hugging Face Hub), Hub-layout directory or checkpoint")
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--micro-batches", type=int, default=1)
    parser.add_argument("--extra-test", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--save-best", action="store_true", help="also save the selected fine-tuned weights next to --output")
    parser.add_argument("--random-init", action="store_true", help="random initialization with the architecture of --pretrained")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()

    recipe = yaml.safe_load(arguments.recipe.read_text())
    task = get_task(arguments.task)
    task_dir = arguments.data_root / arguments.task
    train_records = load_records(task_dir / "train.jsonl.gz")
    test_records = load_records(task_dir / "test.jsonl.gz")
    labels = torch.tensor([record.label for record in train_records], dtype=torch.long)
    validation_indices = stratified_folds(labels, recipe["validation_folds"], recipe["seed"])[arguments.fold]
    train_indices = complement_indices(len(labels), validation_indices)

    seed = recipe["seed"] + arguments.fold
    seed_everything(seed)
    pretrained, config, pretraining_step = load_pretrained(arguments.pretrained, random_init=arguments.random_init)
    config = with_paths_relative_to(config, ROOT)
    encoder = LoCUSEncoder(pretrained)
    set_dropout(encoder, recipe["dropout"])
    model = SequenceClassifier(encoder, task.classes, dropout=recipe["dropout"])

    segmenter = Segmenter(config.data)
    train_cache, test_cache = SegmentCache(train_records, segmenter), SegmentCache(test_records, segmenter)
    extra = {}
    for item in arguments.extra_test:
        name, _, directory = item.partition("=")
        extra[name] = SegmentCache(load_records(Path(directory) / "test.jsonl.gz"), segmenter)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    stem = arguments.output.with_suffix("")
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint_sha256 = weights_sha256(arguments.pretrained)
    start = time.time()
    report = train_fold(
        model, train_cache, train_indices, validation_indices, test_cache, epochs=recipe["epochs"], batch_size=recipe["batch_size"],
        learning_rate=recipe["learning_rate"], warmup_fraction=recipe["warmup_fraction"], weight_decay=recipe["weight_decay"],
        min_learning_rate_ratio=recipe["min_learning_rate_ratio"], gradient_clip=recipe["gradient_clip"], seed=seed, device=device,
        micro_batches=arguments.micro_batches, autocast_dtype=recipe["autocast"], extra_test_caches=extra,
        state_path=Path(f"{stem}.resume_state.pt"), data_history_path=Path(f"{stem}.data_history.jsonl.gz"),
        best_state_path=Path(f"{stem}.best.pt") if arguments.save_best else None,
        fingerprint_extra={"pretrained_sha256": checkpoint_sha256, "train_sha256": sha256_file(task_dir / "train.jsonl.gz"),
                           **({"random_init": True} if arguments.random_init else {})},
        test_every_epoch=recipe.get("test_every_epoch", True), learning_rate_schedule=recipe.get("learning_rate_schedule", "cosine"))
    resources = {"wall_seconds": time.time() - start, "resumed": bool(report["resumed_at_steps"]), "node": socket.gethostname(),
                 "segmentation_seconds": sum(cache.seconds for cache in (train_cache, test_cache, *extra.values()))}
    if device.type == "cuda":
        resources.update(gpu=torch.cuda.get_device_name(device), peak_gpu_memory_allocated_gb=torch.cuda.max_memory_allocated(device) / 1e9)
    chance_loss = math.log(task.classes)
    output = {
        "task": task.name, "benchmark": task.benchmark, "fold": arguments.fold, "seed": seed,
        "pretrained": str(arguments.pretrained), "pretrained_sha256": checkpoint_sha256, "pretraining_step": pretraining_step,
        "random_init": arguments.random_init,
        "segmenter": config.data.segmenter, "train_size": len(train_indices), "validation_size": len(validation_indices),
        "test_size": len(test_records), "recipe": recipe, "micro_batches": arguments.micro_batches,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "collapsed_to_constant": min(entry["training_loss"] for entry in report["history"]) > chance_loss - 0.01,
        "resources": resources, **report,
    }
    arguments.output.write_text(json.dumps(output, indent=2) + "\n")
    Path(f"{stem}.resume_state.pt").unlink(missing_ok=True)
    print(f"{task.name} fold {arguments.fold}: test {task.metric} {report['test_metrics'][task.metric]:.4f} "
          f"(epoch {report['best_epoch']}, lowest validation loss)", flush=True)


if __name__ == "__main__":
    main()
