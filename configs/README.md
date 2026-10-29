# Configurations

The settings of every model and fine-tuning recipe in the paper. The scripts that use them are in `scripts/`
([`scripts/README.md`](../scripts/README.md)).

## Pretraining (`pretrain/`)

One file per released model, passed to `scripts/pretrain.py --config`.

| File | Model | What differs from `locus-2m.yaml` |
|---|---|---|
| `locus-2m.yaml` | LoCUS-2M | |
| `locus-0.5m.yaml` | LoCUS-0.5M | width 64, 2 attention heads, feed-forward 256 |
| `twin-fixed-2m.yaml` | fixed-segment twin | segments of 10 bp starting every 5 bp instead of LCP |
| `twin-bpe-2m.yaml` | BPE twin | BPE tokens (`data/tokenizers/bpe-4096-1gb-seed42.json`) instead of LCP |

Each file has three sections: `model` (width, attention heads, feed-forward size, the radii of the two base attention
layers, the number of segment layers), `data` (the genome windows, the input length, the segmenter) and `training` (the
run name and batch size). Every other setting, such as the number of steps, the learning rate schedule, the masking and
the seed, takes its default from `locus/config.py`; the defaults are the paper's settings. To change one setting without
editing a file, use `--set`, for example `--set training.device=cpu`.

## Fine-tuning (`finetune/`)

Recipes passed to `scripts/finetune.py --recipe`. `scripts/finetune_baseline.py` uses `finetune/<baseline>.yaml` unless
you pass another. Each recipe is used for every task.

| File | Used for | Recipe |
|---|---|---|
| `locus-2m.yaml`, `locus-0.5m.yaml` | LoCUS-2M, LoCUS-0.5M and the twins (the two files are identical) | 20 epochs, batch 128, learning rate 1e-3 with 1% warm-up and cosine decay to 0.1 of the peak, weight decay 0.1, dropout 0.1, bf16 |
| `caduceus-ps-1.9m.yaml`, `hyenadna-tiny-0.44m.yaml` | the two smaller baselines | the LoCUS-2M recipe in float32, each batch of 128 split into 4 micro-batches |
| `hyenadna-small-3.3m.yaml` | HyenaDNA-small, its own recipe | as above, with learning rate 3e-4 and 10% warm-up |
| `hyenadna-small-3.3m-standard.yaml` | HyenaDNA-small, the recipe of the other baselines | the same as `caduceus-ps-1.9m.yaml` |

Every recipe validates on one of 10 folds of the training set (`validation_folds: 10`, `seed: 42`); the fold is chosen
with `--fold`. Micro-batches only save memory: the update is that of the full batch.
