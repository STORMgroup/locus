"""Fine-tune a public pretrained baseline (Caduceus-PS; locus/finetune/baselines.py) with the LoCUS fine-tuning protocol on
one validation fold of a benchmark task and score the test set.

    python scripts/finetune_baseline.py --baseline caduceus-ps-1.9m --data-root data/benchmarks/nt18 \\
        --task H3K27ac_revised --fold 0 --output results/caduceus-ps-1.9m-bf16/nt18/H3K27ac_revised/fold-00.json

Everything except the encoder is shared with scripts/finetune.py: the folds (fold k validates on fold k of a stratified
10-fold split of the training set, seed 42, and trains with seed 42 + k), random-strand training, strand-averaged
evaluation, the dropout 0.1 set in every dropout layer of the encoder and before the linear classifier, selection of the
epoch with the lowest validation loss, and resuming. The recipe defaults to configs/finetune/<baseline>-bf16.yaml, the
protocol of LoCUS in bfloat16 like LoCUS; --recipe configs/finetune/<baseline>.yaml runs it in float32, and with
--recipe configs/finetune/<baseline>-authors.yaml the baseline is fine-tuned with its authors' settings instead (recipe
entries optimizer_groups: their weight-decay groups; train_orientation and evaluation_orientation: the strand read in
training and in validation and test, "forward" or the default random strand and average of both strands; task_settings:
values for single tasks). The recipe's
micro_batches entry (gradient accumulation) can be overridden with --micro-batches. --baseline-path loads a local copy of the baseline's Hugging Face repository instead
of downloading it. Caduceus needs a GPU (mamba-ssm kernels)."""
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

from locus.finetune.baselines import BASELINES, NO_WEIGHT_DECAY, WholeSequenceSegmenter, load_baseline  # noqa: E402
from locus.finetune.collate import SegmentCache  # noqa: E402
from locus.finetune.data import complement_indices, get_task, load_records, sha256_file, stratified_folds  # noqa: E402
from locus.finetune.model import SequenceClassifier, set_dropout  # noqa: E402
from locus.finetune.train import seed_everything, train_fold  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, choices=list(BASELINES))
    parser.add_argument("--baseline-path", help="local copy of the baseline's Hugging Face repository (default: download)")
    parser.add_argument("--recipe", type=Path, help="default: configs/finetune/<baseline>-bf16.yaml (the protocol of LoCUS)")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--micro-batches", type=int, help="default: the recipe's micro_batches (1 when absent)")
    parser.add_argument("--extra-test", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--save-best", action="store_true", help="also save the selected fine-tuned weights next to --output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()

    baseline = BASELINES[arguments.baseline]
    recipe_path = arguments.recipe or ROOT / "configs/finetune" / f"{baseline.recipe}.yaml"
    recipe = yaml.safe_load(recipe_path.read_text())
    recipe.update(recipe.pop("task_settings", {}).get(arguments.task, {}))  # per-task values of a recipe (the authors' settings)
    micro_batches = arguments.micro_batches or recipe.get("micro_batches", 1)
    task = get_task(arguments.task)
    task_dir = arguments.data_root / arguments.task
    train_records = load_records(task_dir / "train.jsonl.gz")
    test_records = load_records(task_dir / "test.jsonl.gz")
    labels = torch.tensor([record.label for record in train_records], dtype=torch.long)
    validation_indices = stratified_folds(labels, recipe["validation_folds"], recipe["seed"])[arguments.fold]
    train_indices = complement_indices(len(labels), validation_indices)

    seed = recipe["seed"] + arguments.fold
    seed_everything(seed)
    encoder = load_baseline(arguments.baseline, arguments.baseline_path)
    set_dropout(encoder, recipe["dropout"])
    model = SequenceClassifier(encoder, task.classes, dropout=recipe["dropout"])

    segmenter = WholeSequenceSegmenter()
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
    start = time.time()
    report = train_fold(
        model, train_cache, train_indices, validation_indices, test_cache, epochs=recipe["epochs"], batch_size=recipe["batch_size"],
        learning_rate=recipe["learning_rate"], warmup_fraction=recipe["warmup_fraction"], weight_decay=recipe["weight_decay"],
        min_learning_rate_ratio=recipe["min_learning_rate_ratio"], gradient_clip=recipe["gradient_clip"], seed=seed, device=device,
        micro_batches=micro_batches, autocast_dtype=recipe["autocast"], extra_test_caches=extra,
        state_path=Path(f"{stem}.resume_state.pt"), data_history_path=Path(f"{stem}.data_history.jsonl.gz"),
        best_state_path=Path(f"{stem}.best.pt") if arguments.save_best else None,
        fingerprint_extra={"baseline": baseline.repository, "revision": baseline.revision,
                           "train_sha256": sha256_file(task_dir / "train.jsonl.gz")},
        test_every_epoch=recipe.get("test_every_epoch", True), learning_rate_schedule=recipe.get("learning_rate_schedule", "cosine"),
        no_weight_decay=NO_WEIGHT_DECAY[recipe["optimizer_groups"]](model) if recipe.get("optimizer_groups") else None,
        train_orientation=recipe.get("train_orientation", "random"), evaluation_orientation=recipe.get("evaluation_orientation", "both"))
    resources = {"wall_seconds": time.time() - start, "resumed": bool(report["resumed_at_steps"]), "node": socket.gethostname(),
                 "segmentation_seconds": sum(cache.seconds for cache in (train_cache, test_cache, *extra.values()))}
    if device.type == "cuda":
        resources.update(gpu=torch.cuda.get_device_name(device), peak_gpu_memory_allocated_gb=torch.cuda.max_memory_allocated(device) / 1e9)
    chance_loss = math.log(task.classes)
    output = {
        "task": task.name, "benchmark": task.benchmark, "fold": arguments.fold, "seed": seed, "baseline": arguments.baseline,
        "repository": baseline.repository, "revision": baseline.revision, "baseline_path": arguments.baseline_path,
        "train_size": len(train_indices), "validation_size": len(validation_indices), "test_size": len(test_records),
        "recipe": recipe, "micro_batches": micro_batches,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "collapsed_to_constant": min(entry["training_loss"] for entry in report["history"]) > chance_loss - 0.01,
        "resources": resources, **report,
    }
    arguments.output.write_text(json.dumps(output, indent=2) + "\n")
    Path(f"{stem}.resume_state.pt").unlink(missing_ok=True)
    print(f"{arguments.baseline} {task.name} fold {arguments.fold}: test {task.metric} {report['test_metrics'][task.metric]:.4f} "
          f"(epoch {report['best_epoch']}, lowest validation loss)", flush=True)


if __name__ == "__main__":
    main()
