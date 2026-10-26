"""Masked-segment pretraining.

Each optimizer step takes `batch_size` examples (crops of genome windows with whole segments masked) and minimizes the
cross-entropy of the hidden bases, summed over the step's targets and divided by their number. Passes over the training
chromosomes repeat until `max_steps`; every pass draws new crops and masks (pass seed = seed + pass index).

Checkpoints hold the model, optimizer, gradient scaler and random number generator states and the position inside the
current pass, so a resumed run continues with exactly the examples it would have seen."""
import json
import math
import queue
import random
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from locus.config import Config
from locus.data.batching import build_model_batch
from locus.data.examples import batch_examples, pass_examples
from locus.data.genome import GenomeDataset
from locus.models.locus_model import LoCUSModel, parameter_count


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_states() -> dict:
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(), "python": random.getstate()}


def set_rng_states(states: dict) -> None:
    torch.set_rng_state(states["torch"].cpu())
    if states["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in states["cuda"]])
    np.random.set_state(states["numpy"])
    random.setstate(states["python"])


def learning_rate_for_step(training, step: int) -> float:
    """Linear warm-up to the peak over `warmup_steps`, then linear decay to 0 at `max_steps`."""
    peak = training.learning_rate
    if training.warmup_steps > 0 and step <= training.warmup_steps:
        return peak * step / training.warmup_steps
    decay_steps = training.max_steps - training.warmup_steps
    if decay_steps <= 1:
        return 0.0
    return peak * max(0.0, 1.0 - (step - training.warmup_steps - 1) / (decay_steps - 1))


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path: Path, model, optimizer, scaler, step: int, config: Config, training_state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"step": step, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "config": config.to_dict(), "training_state": training_state, "scaler_state": scaler.state_dict()}, temporary)
    temporary.replace(path)


def newest_checkpoint(directory: Path) -> Optional[Path]:
    checkpoints = sorted(Path(directory).glob("step-*.pt"), reverse=True)
    return checkpoints[0] if checkpoints else None


def trim_metrics_after_step(path: Path, step: int) -> None:
    if not path.is_file():
        return
    kept = [line for line in path.read_text().splitlines() if line.strip() and json.loads(line)["step"] <= step]
    path.write_text("".join(line + "\n" for line in kept))


def prefetch(iterable, depth: int):
    """Runs `iterable` in a background thread with up to `depth` items ready; order and exceptions are preserved."""
    ready = queue.Queue(maxsize=depth)
    done = object()
    stop = threading.Event()

    def put(item) -> bool:
        while not stop.is_set():
            try:
                ready.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def produce():
        try:
            for item in iterable:
                if not put(item):
                    return
        except BaseException as error:
            put(error)
        put(done)

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    try:
        while True:
            item = ready.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        thread.join(timeout=300)


def step_inputs(batches):
    """(examples, model batch, index of the last pass item consumed) per optimizer step."""
    for batch in batches:
        examples = [example for _, example in batch]
        yield examples, build_model_batch(examples), batch[-1][0]


def train(config: Config, output_directory: Path, resume: Optional[Path] = None, stop_after_step: Optional[int] = None) -> Path:
    """Pretrain until training.max_steps (or `stop_after_step`, which leaves the learning-rate schedule unchanged) and
    return the final checkpoint."""
    training, data = config.training, config.data
    set_random_seed(training.seed)
    device = choose_device(training.device)
    dataset = GenomeDataset(Path(data.genome_windows), data.train_sources, data.context_length)
    model = LoCUSModel(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, betas=(training.adam_beta1, training.adam_beta2),
                                  eps=training.adam_epsilon, weight_decay=training.weight_decay)
    mixed_precision = training.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    metric_path = output_directory / "train_metrics.jsonl"
    checkpoint_directory = output_directory / "checkpoints"

    step, pass_index, resume_skip_items = 0, 0, 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location=str(device), weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        step = int(checkpoint["step"])
        state = checkpoint["training_state"]
        pass_index = int(state["pass"])
        if "pass_items_consumed" in state:
            # continue the interrupted pass; the loop below increments the pass index again
            resume_skip_items = int(state["pass_items_consumed"])
            pass_index -= 1
            set_rng_states(state["rng_states"])
        trim_metrics_after_step(metric_path, step)
        print(f"resumed from {resume} at step {step}", flush=True)

    manifest = {"config": config.to_dict(), "parameters": parameter_count(model), "device": str(device),
                "mixed_precision": mixed_precision, "training_shards": len(dataset.shards)}
    (output_directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"parameters: {parameter_count(model):,}; segmenter: {data.segmenter}; device: {device}; "
          f"mixed precision: {mixed_precision}; output: {output_directory}", flush=True)

    last_step = training.max_steps if stop_after_step is None else min(training.max_steps, stop_after_step)
    model.train()
    autocast = (lambda: torch.autocast(device_type=device.type, dtype=torch.float16)) if mixed_precision else nullcontext
    while step < last_step:
        pass_index += 1
        pass_seed = training.seed + pass_index
        skip_items, resume_skip_items = resume_skip_items, 0
        pass_items_consumed = skip_items
        batches = batch_examples(pass_examples(dataset, data, training, pass_index, pass_seed, skip_items), training.batch_size)
        inputs = prefetch(step_inputs(batches), depth=2)
        batch_count = 0
        while True:
            data_start = time.perf_counter()
            item = next(inputs, None)
            if item is None:
                break
            examples, model_batch, last_index = item
            pass_items_consumed = last_index + 1
            data_seconds = time.perf_counter() - data_start
            batch_count += 1
            step += 1
            learning_rate = learning_rate_for_step(training, step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            batch = model_batch.to(device)
            target_total = int(batch.target_mask.sum())
            with autocast():
                logits = model(batch.base_ids, batch.valid_mask, batch.segment_starts, batch.nucleotide_mask,
                               batch.segment_mask, batch.target_positions)
                loss_sum = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch.labels.reshape(-1),
                                           ignore_index=-100, reduction="sum")
            scaler.scale(loss_sum / target_total).backward()
            loss_value = loss_sum.item()
            correct = ((logits.argmax(dim=-1) == batch.labels) & batch.target_mask).sum().item()
            target_count = batch.target_mask.sum().item()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=training.gradient_clip_norm).item() \
                if training.gradient_clip_norm > 0 else 0.0
            scaler.step(optimizer)
            scaler.update()
            compute_seconds = time.perf_counter() - step_start

            metrics = {
                "step": step, "pass": pass_index, "loss": loss_value / target_count, "accuracy": correct / target_count,
                "gradient_norm": gradient_norm, "learning_rate": learning_rate, "examples": len(examples),
                "targets": target_count, "eligible_positions": sum(example.eligible_position_count for example in examples),
                "raw_bases": sum(len(example.sequence) for example in examples),
                "segments_per_example": sum(len(example.segments) for example in examples) / len(examples),
                "data_seconds": data_seconds, "compute_seconds": compute_seconds,
            }
            with metric_path.open("a") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            if step == 1 or step % training.log_every == 0:
                print(f"step={step} pass={pass_index} loss={metrics['loss']:.4f} acc={metrics['accuracy']:.3f} "
                      f"grad={gradient_norm:.3f} lr={learning_rate:.2e} bases={metrics['raw_bases']} "
                      f"data_s={data_seconds:.3f} compute_s={compute_seconds:.3f}", flush=True)
            if step % training.checkpoint_every == 0:
                save_checkpoint(checkpoint_directory / f"step-{step:08d}.pt", model, optimizer, scaler, step, config,
                                {"pass": pass_index, "pass_items_consumed": pass_items_consumed, "rng_states": rng_states()})
                for old in checkpoint_directory.glob("step-*.pt"):
                    old_step = int(old.stem.removeprefix("step-"))
                    if (old_step < step and training.keep_checkpoint_every and old_step % training.keep_checkpoint_every != 0
                            and old_step not in set(training.keep_checkpoint_steps)):
                        old.unlink(missing_ok=True)
            if step >= last_step:
                break
        inputs.close()
        if batch_count == 0 and skip_items == 0:
            raise RuntimeError("The training partition contains no examples")
    final = checkpoint_directory / "final.pt"
    save_checkpoint(final, model, optimizer, scaler, step, config, {"pass": pass_index})
    print(f"finished at step {step}; checkpoint: {final}", flush=True)
    return final
