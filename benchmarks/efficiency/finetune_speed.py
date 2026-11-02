"""Fine-tuning throughput of LoCUS and of the same-protocol baselines under the same conditions: one GPU, the NT18
splice_sites_all_revised training sequences (600 bp), batch 64, bf16 autocast, AdamW (lr 1e-4), full fine-tuning of the
encoder with a linear head on the mean of the final per-base vectors, cross-entropy. After `--warmup-steps` untimed steps,
`--timed-steps` steps are timed (wall clock of collation, forward, backward and optimizer step). Reported per model:
parameters of the fine-tuned model, seconds per step, sequences per second, minutes per epoch of the task's training set,
peak GPU memory, and the last training loss.

Every model runs through the fine-tuning pipeline of scripts/finetune.py (classifier and collation), with its input
prepared before timing: the LCP segment boundaries of LoCUS are computed once per sequence (as the fine-tuning pipeline
does), and the baselines (Caduceus-PS 1.9M and 0.47M, locus/finetune/baselines.py) read single bases, so no model segments
or tokenizes inside the timed step.

    python benchmarks/efficiency/finetune_speed.py [--models "LoCUS-2M" "Caduceus-PS 1.9M" ...] \\
        [--output results/efficiency/finetune_speed.json]

The LoCUS models are the released ones (downloaded by name), or with --models-from DIR the checkpoints
DIR/<name>/checkpoints/final.pt of scripts/pretrain.py. The baselines are downloaded at their pinned revisions (or read
from --baseline-path). Caduceus needs a CUDA GPU.

A model that fails is reported with its error and the next one is measured; results already in the output file are kept
unless the same model is measured again."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from locus.finetune.data import get_task, load_records  # noqa: E402

TASK = "splice_sites_all_revised"
LOCUS = {"LoCUS-0.5M": "locus-0.5m", "LoCUS-2M": "locus-2m"}
BASELINES = {"Caduceus-PS 1.9M": "caduceus-ps-1.9m", "Caduceus-PS 0.47M": "caduceus-ps-0.47m"}


def synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(name, model, make_batch, step_loss, train_size, arguments, device) -> dict:
    model = model.to(device).train()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for step in range(arguments.warmup_steps + arguments.timed_steps):
        synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = step_loss(model, make_batch(step))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        if step >= arguments.warmup_steps:
            times.append(time.perf_counter() - start)
    seconds = sum(times) / len(times)
    batch = arguments.batch_size
    result = {"parameters": parameters, "seconds_per_step": seconds, "sequences_per_second": batch / seconds,
              "epoch_minutes": train_size / (batch / seconds) / 60,
              "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None,
              "batch": batch, "steps_timed": arguments.timed_steps, "loss_last": float(loss)}
    print(f"{name:18s} {parameters / 1e6:7.2f}M  {result['sequences_per_second']:8.1f} seq/s  {seconds * 1000:7.1f} ms/step  "
          f"epoch {result['epoch_minutes']:6.2f} min" + (f"  peak {result['peak_memory_gb']:.2f} GB" if device.type == "cuda" else ""), flush=True)
    return result


def batch_indices(step: int, batch: int, size: int) -> list:
    """The training sequences of a step: consecutive sequences, wrapping around the training set."""
    return [j % size for j in range(step * batch, (step + 1) * batch)]


def pipeline_model(name, arguments, device):
    """The fine-tuning classifier and collation of scripts/finetune.py, with a LoCUS or a baseline encoder."""
    from locus.config import with_paths_relative_to
    from locus.finetune.collate import Collator, SegmentCache
    from locus.finetune.model import LoCUSEncoder, SequenceClassifier, load_pretrained
    from locus.segmentation import Segmenter

    records = load_records(arguments.data_root / TASK / "train.jsonl.gz")
    labels = torch.tensor([record.label for record in records])
    task = get_task(TASK)
    if name in LOCUS:
        source = arguments.models_from / LOCUS[name] / "checkpoints/final.pt" if arguments.models_from else LOCUS[name]
        pretrained, config, _ = load_pretrained(source)
        encoder, segmenter = LoCUSEncoder(pretrained), Segmenter(with_paths_relative_to(config, ROOT).data)
    else:
        from locus.finetune.baselines import WholeSequenceSegmenter, load_baseline

        encoder, segmenter = load_baseline(BASELINES[name], arguments.baseline_path.get(name)), WholeSequenceSegmenter()
    model = SequenceClassifier(encoder, task.classes, dropout=0.1)
    cache = SegmentCache(records, segmenter)
    for step in range(arguments.warmup_steps + arguments.timed_steps):
        for index in batch_indices(step, arguments.batch_size, len(records)):
            cache.bounds(index, "forward")
    collate = Collator(cache, "forward")

    def make_batch(step):
        indices = batch_indices(step, arguments.batch_size, len(records))
        return collate(indices).to(device), labels[indices].to(device)

    def step_loss(model, batch):
        return nn.functional.cross_entropy(model(batch[0]).float(), batch[1])

    return model, make_batch, step_loss, len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    names = list(LOCUS) + list(BASELINES)
    parser.add_argument("--models", nargs="+", default=names, choices=names)
    parser.add_argument("--models-from", type=Path, help="pretraining directory with <name>/checkpoints/final.pt (default: the released models)")
    parser.add_argument("--baseline-path", nargs=2, action="append", default=[], metavar=("MODEL", "DIR"),
                        help="local copy of a baseline's Hugging Face repository, e.g. 'Caduceus-PS 1.9M' DIR")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/benchmarks/nt18")
    parser.add_argument("--output", type=Path, default=ROOT / "results/efficiency/finetune_speed.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--timed-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0, help="seeds the classification heads (and dropout) of every model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()
    arguments.baseline_path = dict(arguments.baseline_path)
    device = torch.device(arguments.device)

    result = json.loads(arguments.output.read_text()) if arguments.output.is_file() else {}
    for name in arguments.models:
        torch.manual_seed(arguments.seed)
        try:
            result[name] = measure(name, *pipeline_model(name, arguments, device), arguments, device)
        except Exception as error:  # report and continue with the next model
            print(f"{name}: failed, {type(error).__name__}: {error}", flush=True)
            result[name] = {"error": f"{type(error).__name__}: {error}"}
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result["_setup"] = {"task": TASK, "batch": arguments.batch_size, "warmup": arguments.warmup_steps, "steps": arguments.timed_steps,
                        "precision": "bf16 autocast" if device.type == "cuda" else "float32",
                        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=1) + "\n")
    failed = [name for name in arguments.models if "error" in result[name]]
    if failed:
        sys.exit(f"failed: {', '.join(failed)} (the other models are in {arguments.output})")


if __name__ == "__main__":
    main()
