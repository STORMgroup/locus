"""Full fine-tuning of one validation fold.

Recipe (per model size, see configs/finetune): AdamW (weight decay on matrices only), linear warm-up then cosine decay to
`min_learning_rate_ratio` of the peak (or a constant learning rate after the warm-up), gradient clipping at 1, dropout 0.1
in the encoder and before the classifier, bf16 autocast. Each training window is read on a random strand; validation and test average the logits of both
strands. After every epoch the model is scored on the validation fold (and, unless switched off for long inputs, on the test
set); the reported checkpoint is the epoch with the lowest validation loss. The test set never influences training or selection.

The training state is saved after every epoch, so a fold interrupted by a time limit resumes from its last finished
epoch and continues exactly like an uninterrupted run."""
import copy
import gzip
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from locus.finetune.collate import Collator, SegmentCache, epoch_permutation
from locus.finetune.metrics import classification_metrics

AUTOCAST_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batches(indices: list, batch_size: int):
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


@torch.no_grad()
def predict(model, cache: SegmentCache, indices: list, orientation: str, batch_size: int, device, autocast_dtype) -> tuple:
    collate = Collator(cache, orientation)
    model.eval()
    logits, labels = [], []
    for chunk in _batches(indices, batch_size):
        batch = collate(chunk)
        labels.append(batch.labels.numpy())
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=device.type == "cuda" and autocast_dtype != torch.float32):
            logits.append(model(batch.to(device)).float().cpu().numpy())
    return np.concatenate(logits), np.concatenate(labels)


def evaluate(model, cache: SegmentCache, indices: list, batch_size: int, device, autocast_dtype) -> tuple[dict, float]:
    """(metrics, mean cross-entropy) of the strand-averaged logits."""
    logits, labels = predict(model, cache, indices, "forward", batch_size, device, autocast_dtype)
    reverse_logits, reverse_labels = predict(model, cache, indices, "reverse", batch_size, device, autocast_dtype)
    if not np.array_equal(labels, reverse_labels):
        raise RuntimeError("Forward and reverse evaluation labels are misaligned")
    logits = 0.5 * (logits + reverse_logits)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return classification_metrics(labels, logits), float(-log_probabilities[np.arange(len(labels)), labels].mean())


def effective_autocast_dtype(name: str, device) -> torch.dtype:
    """bf16 needs compute capability 8.0; older GPUs use fp16 (7.x, with loss scaling) or fp32."""
    dtype = AUTOCAST_DTYPES[name]
    if device.type == "cuda" and dtype == torch.bfloat16:
        capability = torch.cuda.get_device_capability(device)
        if capability < (8, 0):
            dtype = torch.float16 if capability >= (7, 0) else torch.float32
    return dtype


def train_fold(model, train_cache: SegmentCache, train_indices: torch.Tensor, validation_indices: torch.Tensor,
               test_cache: SegmentCache, *, epochs: int, batch_size: int, learning_rate: float, warmup_fraction: float,
               weight_decay: float, min_learning_rate_ratio: float, gradient_clip: float, seed: int, device,
               micro_batches: int = 1, autocast_dtype: str = "bfloat16", extra_test_caches: Optional[dict] = None,
               state_path: Optional[Path] = None, data_history_path: Optional[Path] = None,
               best_state_path: Optional[Path] = None, fingerprint_extra: Optional[dict] = None,
               test_every_epoch: bool = True, learning_rate_schedule: str = "cosine") -> dict:
    seed_everything(seed)
    model.to(device)
    steps_per_epoch = math.ceil(len(train_indices) / batch_size)
    steps = epochs * steps_per_epoch
    warmup_steps = round(steps * warmup_fraction)
    groups = {}
    for _, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups.setdefault(parameter.ndim > 1, []).append(parameter)
    optimizer = torch.optim.AdamW([{"params": groups[decays], "weight_decay": weight_decay if decays else 0.0, "lr": learning_rate}
                                   for decays in sorted(groups)], lr=learning_rate)
    amp_dtype = effective_autocast_dtype(autocast_dtype, device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    eval_batch_size = max(1, batch_size // micro_batches)
    train_list, validation_list = train_indices.tolist(), validation_indices.tolist()
    collate = Collator(train_cache, "random", seed=seed)

    if learning_rate_schedule not in {"cosine", "constant"}:
        raise ValueError(f"Unknown learning-rate schedule: {learning_rate_schedule}")

    def learning_rate_factor(step: int) -> float:
        if warmup_steps > 0 and step <= warmup_steps:
            return step / warmup_steps
        if learning_rate_schedule == "constant" or steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / (steps - warmup_steps)
        return min_learning_rate_ratio + (1.0 - min_learning_rate_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def digest(values) -> str:
        return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()

    fingerprint = {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "warmup_steps": warmup_steps,
                   "schedule": learning_rate_schedule,
                   "weight_decay": weight_decay, "min_learning_rate_ratio": min_learning_rate_ratio, "seed": seed,
                   "micro_batches": micro_batches, "train_indices": digest(train_list), "validation_indices": digest(validation_list),
                   "parameters": digest(sorted((n, list(p.shape)) for n, p in model.named_parameters())), **(fingerprint_extra or {})}
    best_loss, best_step, best_state, best_metrics = float("inf"), 0, None, None
    history, data_history, resumed_at = [], [], []
    interval_loss_sum, interval_examples, nonfinite_steps, interval_nonfinite = 0.0, 0, 0, 0
    start_step = 1
    if state_path is not None and state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint:
            changed = sorted(k for k in fingerprint if state["fingerprint"].get(k) != fingerprint[k])
            raise RuntimeError(f"Saved state {state_path} was made with a different setup ({changed}); delete it to start over")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        loop = state["loop"]
        best_loss, best_step, best_metrics, history = loop["best_loss"], loop["best_step"], loop["best_metrics"], loop["history"]
        best_state, data_history, resumed_at = state["best_state"], state["data_history"], state["resumed_at"] + [state["step"]]
        nonfinite_steps = loop["nonfinite_steps"]
        torch.set_rng_state(state["rng"]["torch"])
        if torch.cuda.is_available() and state["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])
        start_step = state["step"] + 1
        print(f"resumed from {state_path} at step {state['step']}", flush=True)

    def save_state(step: int) -> None:
        state = {"fingerprint": fingerprint, "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "best_state": best_state, "data_history": data_history, "resumed_at": resumed_at,
                 "loop": {"best_loss": best_loss, "best_step": best_step, "best_metrics": best_metrics, "history": history,
                          "nonfinite_steps": nonfinite_steps},
                 "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                         "numpy": np.random.get_state(), "python": random.getstate()}}
        temporary = state_path.with_name(state_path.name + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, state_path)

    epoch_batches, epoch_indices, epoch_orientations, diverged_at_step = None, [], [], None
    for step in range(start_step, steps + 1):
        epoch = (step - 1) // steps_per_epoch + 1
        if epoch_batches is None or (step - 1) % steps_per_epoch == 0:
            collate.set_epoch(epoch)
            order = [train_list[position] for position in epoch_permutation(len(train_list), seed, epoch)]
            epoch_batches = _batches(order, batch_size)
        batch = collate(next(epoch_batches))
        epoch_indices.extend(batch.indices.tolist())
        epoch_orientations.append(collate.last_orientations)
        device_batch = batch.to(device)

        model.train()
        factor = learning_rate_factor(step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * factor
        optimizer.zero_grad(set_to_none=True)
        chunks = [device_batch] if micro_batches == 1 else device_batch.split(micro_batches)
        loss = None
        for chunk in chunks:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda" and amp_dtype != torch.float32):
                chunk_loss = torch.nn.functional.cross_entropy(model(chunk), chunk.labels) * (len(chunk.labels) / len(device_batch.labels))
            if micro_batches > 1:
                scaler.scale(chunk_loss).backward()
                chunk_loss = chunk_loss.detach()
            loss = chunk_loss if loss is None else loss + chunk_loss
        if torch.isfinite(loss):
            if micro_batches == 1:
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            if scaler.is_enabled() or torch.isfinite(gradient_norm):
                scaler.step(optimizer)
            else:
                nonfinite_steps += 1
                interval_nonfinite += 1
            scaler.update()
            interval_loss_sum += float(loss.detach().cpu()) * len(device_batch.labels)
            interval_examples += len(device_batch.labels)
        else:
            optimizer.zero_grad(set_to_none=True)
            nonfinite_steps += 1
            interval_nonfinite += 1

        if step % steps_per_epoch == 0 or step == steps:
            validation_metrics, validation_loss = evaluate(model, train_cache, validation_list, eval_batch_size, device, amp_dtype)
            entry = {"step": step, "epoch": epoch, "training_loss": interval_loss_sum / interval_examples if interval_examples else float("nan"),
                     "validation_loss": validation_loss, "validation_metrics": validation_metrics, "learning_rate": learning_rate * factor,
                     "nonfinite_steps": interval_nonfinite}
            if test_every_epoch:
                # recorded for every epoch, never used for selection
                entry["test_metrics"], entry["test_loss"] = evaluate(model, test_cache, list(range(len(test_cache.records))), eval_batch_size,
                                                                     device, amp_dtype)
            for name, cache in (extra_test_caches or {}).items():
                metrics, extra_loss = evaluate(model, cache, list(range(len(cache.records))), eval_batch_size, device, amp_dtype)
                entry.setdefault("extra_test_metrics", {})[name] = metrics
                entry.setdefault("extra_test_loss", {})[name] = extra_loss
            history.append(entry)
            print(f"epoch={epoch} step={step} train_loss={entry['training_loss']:.5f} val_loss={validation_loss:.5f} "
                  f"val_mcc={validation_metrics['mcc']:.5f}", flush=True)
            interval_loss_sum, interval_examples, interval_nonfinite = 0.0, 0, 0
            if validation_loss < best_loss:
                best_loss, best_step, best_metrics = validation_loss, step, validation_metrics
                best_state = copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
            if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                diverged_at_step = step
                print(f"stopping at step {step}: the weights are no longer finite", flush=True)
                break
            orientations = "".join(epoch_orientations)
            data_history.append({"epoch": epoch, "last_step": step, "examples": len(epoch_indices),
                                 "every_training_window_once": sorted(epoch_indices) == sorted(train_list),
                                 "reverse_complement_fraction": orientations.count("R") / max(len(orientations), 1),
                                 "sha256": digest([epoch_indices, orientations]), "indices": epoch_indices, "orientations": orientations})
            epoch_indices, epoch_orientations = [], []
            if data_history_path is not None:
                with gzip.open(data_history_path, "wt", encoding="utf-8") as handle:
                    for record in data_history:
                        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            if state_path is not None and step < steps:
                save_state(step)

    model.load_state_dict(best_state)
    if best_state_path is not None:
        torch.save({"model": best_state, "best_step": best_step, "validation_loss": best_loss, "seed": seed}, best_state_path)
    test_metrics, test_loss = evaluate(model, test_cache, list(range(len(test_cache.records))), eval_batch_size, device, amp_dtype)
    return {
        "best_step": best_step, "best_epoch": (best_step - 1) // steps_per_epoch + 1, "steps": steps, "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup_steps, "validation_loss": best_loss, "validation_metrics": best_metrics, "test_loss": test_loss,
        "test_metrics": test_metrics, "history": history, "nonfinite_steps": nonfinite_steps, "resumed_at_steps": resumed_at,
        "data_history": [{k: v for k, v in record.items() if k not in ("indices", "orientations")} for record in data_history],
        "effective_autocast_dtype": str(amp_dtype).removeprefix("torch."), "diverged_at_step": diverged_at_step,
    }
