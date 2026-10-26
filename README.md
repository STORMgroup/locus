# LoCUS

LoCUS is a family of small DNA language models whose tokens are segments from locally consistent parsing (LCP). The LCP
segments of a sequence are determined by the sequence around them, so an edit changes only the segments near it.

This repository contains everything needed to use the pretrained models and to reproduce every result of the paper:
pretraining, fine-tuning and evaluation of LoCUS and of the baseline models, and the scripts that build the result
tables.

## Quick start

```bash
git clone <repository> locus && cd locus
bash install.sh --models        # environment, LCP library, tests, and the six pretrained models
source .venv/bin/activate
```

```python
import locus

model = locus.load("locus-2m")                           # pretrained model from the Hugging Face Hub
vectors = model.embed(["ACGTAGGCTA" * 50])                # [sequences, 128]: mean over bases, both strands averaged
per_base = model.embed(["ACGTAGGCTA" * 50], per_base=True)  # list of [length, 128] arrays
```

Fine-tune a released model on one task and fold:

```bash
python benchmarks/nt18/prepare.py
python scripts/finetune.py --pretrained locus-2m --recipe configs/finetune/locus-2m.yaml \
    --data-root data/benchmarks/nt18 --task H3K27ac_revised --fold 0 \
    --output results/locus-2m/nt18/H3K27ac_revised/fold-00.json
```

## Reproduce the paper

```bash
bash install.sh --all                                             # also the baselines' CUDA kernels and PBSIM3
source .venv/bin/activate
python scripts/reproduce.py all --nt18-folds 10 --gpus 0,1,2,3    # every experiment; NT18 on all 10 folds, as in the paper
python scripts/compare_results.py                                 # differences from the paper's numbers
```

`scripts/reproduce.py` downloads and prepares the data each experiment needs, runs its jobs on the listed GPUs, and
writes the tables to `results/tables.md` (every number also in `results/tables.json`). A job whose result exists is
skipped and an interrupted fine-tune resumes from its last finished epoch, so the same command continues after an
interruption. `--list` shows how many jobs of each plan are done, `--dry-run` prints the commands without running them,
and a single plan can be run by name:

| Plan | Experiment | Jobs (paper) | GPU time (RTX A5000, one job per GPU) |
|---|---|---|---|
| `nt18` | NT18 revised, LoCUS-0.5M / 2M / 7M, 18 tasks x 10 folds | 540 | about 13 / 17 / 25 min per fold |
| `random-init` | pretraining ablation: the LoCUS-2M architecture fine-tuned on NT18 from random initialization | 180 | about 17 min per fold |
| `baselines` | NT18 revised, Caduceus-PS 1.9M, HyenaDNA-tiny 0.44M, HyenaDNA-small 3.3M, same protocol | 540 | about 71 / 12 / 29 min per fold |
| `genomic-benchmarks` | Genomic Benchmarks, LoCUS-0.5M / 2M / 7M, 9 tasks x 3 folds | 81 | |
| `tokenizer-study` | the three 2M twins on NT18, scored on the modified test sets (errors, controls, PBSIM3 reads) | 1,080 | about 17 min per fold, 6 min per scoring (7 sets x 10 error draws) |
| `heldout` | LoCUS-2M and its random-initialization control on NT18 tested on chromosomes outside the pretraining chromosomes (official test set scored alongside) | 108 | about 17 min per fold |
| `context` | context-length ablation, long-range enhancer task at 4 kb and 16 kb input | 18 | |
| `long-range` | frozen embeddings, eQTL and bulk RNA tasks of the long-range benchmark | 19 | |
| `parsing` | parse-level edit locality and context independence of the three segmenters (CPU) | 1 | 2 CPU minutes |
| `bend` | BEND zero-shot variant effect prediction | 6 | |
| `efficiency` | fine-tuning throughput of LoCUS and the baselines | 1 | minutes |
| `timing` | wall clock of full fine-tunes (fold 0 of every NT18 task), always one job per GPU | 108 | |
| `pretrain` | the six pretraining runs (not part of `all`) | 6 | 11-19 h per model |

The paper reports NT18 over all 10 folds of the split (`--nt18-folds 10`, the job counts above). Without the option, the
NT18 plans (`nt18`, `random-init`, `baselines`, `tokenizer-study`) run folds 0-2 only (162, 54, 162 and 324 jobs), which
takes about a third of the time. The `heldout` plan always runs folds 0-2. The paper's fine-tunes shared their GPU with up
to two others; `--jobs-per-gpu 3` does the same (see Reproducibility for what this changes). With three jobs per GPU on
up to 12 GPUs, the paper's 10-fold NT18 plans took about two days.

`reference/paper_tables.md` and `.json` hold the paper's numbers in the same format; `scripts/compare_results.py`
reports, per table, the mean and largest difference of a rerun from them. Fine-tuning on GPU is not bit-for-bit
repeatable (see Reproducibility), so the 18-task means of a rerun differ from the paper's by about 0.01, and single small
tasks such as promoter_tata by more.

To reproduce the pretrained models as well:

```bash
python scripts/reproduce.py pretrain --gpus 0,1,2,3          # runs/<model>/checkpoints/final.pt
python scripts/reproduce.py all --models-from runs --results results-own --gpus 0,1,2,3
```

## Model

Each model has four stages:
1. **Base encoder:** every base is embedded and contextualized by two layers of windowed self-attention, with radii 32
   and 128 bases.
2. **Segment pooling:** the input is cut into LCP level-2 segments, which overlap and are about 10 bp long. The base
   vectors of each segment are pooled into one vector, and a depthwise convolution mixes each segment with its
   neighbours.
3. **Segment transformer:** six transformer layers run over the segment sequence, with rotary position embeddings at the
   segment centres in base pairs.
4. **Per-base output:** each base's vector is the mean of the segments covering it.

Pretraining masks whole segments and predicts their bases from the covering segments.

| Model | Parameters | Width | Context | Segments |
|---|---|---|---|---|
| LoCUS-0.5M | 471,556 | 64 | 4 kb | LCP level 2 |
| LoCUS-2M | 1,737,604 | 128 | 4 kb | LCP level 2 |
| LoCUS-7M | 6,653,572 | 256 | 4 kb | LCP level 2 |
| LoCUS-2M-16k | 1,737,604 | 128 | 16 kb | LCP level 2 (context-length ablation) |
| twin-fixed-2m | 1,737,604 | 128 | 4 kb | fixed 10 bp segments every 5 bp (tokenizer study) |
| twin-bpe-2m | 1,737,604 | 128 | 4 kb | BPE tokens, vocabulary 4,096 (tokenizer study) |

All models are pretrained on GRCh38 chromosomes 3-22; chromosome 2 is held out for validation and chromosome 1 for
testing. Each run takes 359,466 steps. The 4 kb models see 16 crops of 256 to 4,096 bp per step (12.5B bases). The
16 kb model sees 4 crops of 256 to 16,384 bp per step (12.0B bases).

## Pretrained models

The six models of the table above are on the Hugging Face Hub as `<huggingface-organization>/<model>` (for example
`<huggingface-organization>/locus-2m`). Each repository holds `config.json` and `model.safetensors`.

- **Use by name:** `locus.load(...)`, `scripts/finetune.py --pretrained ...` and the benchmark scripts take the model
  name and download the files on first use.
- **Download in advance:** `python scripts/download_models.py`.
- **Offline:** set `LOCUS_HUB_DIR` to a directory with one sub-directory per model.
- **Your own checkpoints:** a `checkpoints/final.pt` written by `scripts/pretrain.py` can be passed wherever a model name
  is accepted.
- **Publishing (authors):** `scripts/upload_to_hub.py` converts pretraining checkpoints to this layout and, with
  `--upload`, pushes them to the Hub.

## Baseline models

The baselines are public pretrained models, fine-tuned with exactly the protocol of the LoCUS models (same folds, seeds,
strand handling and selection; `locus/finetune/baselines.py`, `scripts/finetune_baseline.py`). Their Hugging Face
repositories are pinned to the revisions used in the paper.

| Baseline | Hugging Face repository | Revision | Recipe |
|---|---|---|---|
| `caduceus-ps-1.9m` | kuleshov-group/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 | 2de3650 | LoCUS-2M recipe, float32 |
| `hyenadna-tiny-0.44m` | LongSafari/hyenadna-tiny-1k-seqlen-hf | e8c1eff | LoCUS-2M recipe, float32 |
| `hyenadna-small-3.3m` | LongSafari/hyenadna-small-32k-seqlen-hf | 8fe770c | LoCUS-7M recipe, float32 |

```bash
bash install.sh --baselines                       # mamba-ssm and causal-conv1d (Caduceus), and the three models
python scripts/finetune_baseline.py --baseline hyenadna-tiny-0.44m --data-root data/benchmarks/nt18 \
    --task H3K27ac_revised --fold 0 --output results/hyenadna-tiny-0.44m/nt18/H3K27ac_revised/fold-00.json
```

The `baselines` plan adds `--save-best`, which keeps the selected fine-tuned weights (`fold-XX.best.pt`) next to each result;
`benchmarks/noisy_reads/score.py` scores them on the modified test sets of the tokenizer study, as for the twins.

Caduceus needs a CUDA GPU. mamba-ssm and causal-conv1d (`requirements-baselines.txt`) publish prebuilt wheels for
Python 3.10 and newer; on Python 3.9 they are compiled (about 50 minutes), which needs the CUDA toolkit (`nvcc`).
causal-conv1d 1.7.0 uses type hints that need Python 3.10, so on Python 3.9 `install.sh` builds it from its source with
`third_party/causal-conv1d-1.7.0-python3.9.patch`, which rewrites only those type hints (as in the paper's environment).

## Installation

`bash install.sh` creates `.venv`, installs the pinned packages, builds the native code and runs the unit tests. Options:
`--models` (download the LoCUS models), `--baselines` (baseline kernels and models), `--pbsim3` (build PBSIM3 for the
simulated reads of the tokenizer study), `--all`. torch 2.6.0 from PyPI is built for CUDA 12.4; for the CUDA 11.8 build
of the paper, run `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 bash install.sh`. The steps by hand:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # Python 3.9; the paper used PyTorch 2.6.0 with CUDA 11.8
pip install -e . --no-deps
make                                 # builds build/lcptools (the LCP library) and build/split_fasta
pytest                               # unit tests, CPU only, about one minute
```

`third_party/lcptools` is [lcptools](https://github.com/BilkentCompGen/lcptools) at commit 9dcc05f, with the changes in
`third_party/lcptools/local-changes.patch`. Those changes add the core type to `struct core`, which the Python bindings in
`locus/segmentation.py` rely on. They also fix freeing the cores in `lps_deepen1_dct_iters`. `make lcptools` builds the library with
the settings of the paper (32-bit labels and positions, dictionary encoding); the build gives the same library as the
paper's, byte for byte, also when an environment module sets `LD_RUN_PATH`. `make pbsim3` clones and builds
[PBSIM3](https://github.com/yukiteruono/pbsim3) at commit 35ce0de in `build/pbsim3`.

## Data

`scripts/reproduce.py` prepares all data itself. The individual steps:

```bash
bash scripts/prepare_genome.sh                     # GRCh38.p14 -> data/GRCh38.p14.acgt.w4096.s4096 and .w16384.s16384
python benchmarks/nt18/prepare.py                  # revised Nucleotide Transformer benchmark -> data/benchmarks/nt18
python benchmarks/genomic_benchmarks/prepare.py    # Genomic Benchmarks -> data/benchmarks/genomic_benchmarks
```

The genome is downloaded from NCBI and checked against its MD5 checksum, then cut into consecutive 4,096 bp (or
16,384 bp) windows that contain only A, C, G and T (`--download-only` skips the windows, which only pretraining needs).
Benchmark datasets are pinned to the dataset revisions used in the paper; the BEND variant files are checked against
their SHA-256. The BPE tokenizer of twin-bpe-2m is included as `data/tokenizers/bpe-4096-1gb-seed42.json`;
`scripts/train_bpe.py` documents how it was trained.

## Pretraining

```bash
python scripts/pretrain.py --config configs/pretrain/locus-2m.yaml --output-dir runs/locus-2m
python scripts/pretrain.py --config configs/pretrain/locus-2m.yaml --output-dir runs/locus-2m --resume   # continue
```

Training uses one GPU, AdamW (learning rate 6e-4, 3,600 warm-up steps, then linear decay to zero), and fp16 mixed
precision. Windows are read in genome order through a shuffle buffer of 512 crops. Every random draw of an example is
seeded by the window and the pass, so a resumed run continues with exactly the examples an uninterrupted run would see.
The output directory holds `train_metrics.jsonl` and checkpoints; `checkpoints/final.pt` is the pretrained model.

## Fine-tuning

```bash
python scripts/finetune.py --pretrained locus-2m --recipe configs/finetune/locus-2m.yaml \
    --data-root data/benchmarks/nt18 --task H3K27ac_revised --fold 0 --micro-batches 2 \
    --output results/locus-2m/nt18/H3K27ac_revised/fold-00.json
```

`--pretrained` takes a released model name (`locus-2m`), a directory in the Hub layout, or a checkpoint of
`scripts/pretrain.py`. `--random-init` uses only its architecture and segmenter and starts from random weights (the
pretraining ablation).

**Recipe** (every task uses the same recipe for a given model size):
- 20 epochs, batch 128, AdamW with weight decay 0.1.
- Cosine decay to 0.1 of the peak learning rate. LoCUS-0.5M and LoCUS-2M: learning rate 1e-3 with 1% warm-up. LoCUS-7M:
  3e-4 with 10% warm-up.
- Dropout 0.1, bf16.
- Each training window is read on a random strand; at evaluation the logits of both strands are averaged.

**Folds and selection:**
- Fold k validates on fold k of a stratified 10-fold split of the training set (seed 42) and trains with seed 42 + k.
  The paper uses all 10 folds (k = 0, ..., 9) on NT18 and folds 0-2 on the other fine-tuned benchmarks and in the
  held-out-chromosome check.
- The reported checkpoint is the epoch with the lowest validation loss. The test set is scored after every epoch but
  never used for selection.

`--micro-batches N` splits each batch for gradient accumulation; the update is the same, it only uses less memory. The
paper's runs used 1 (LoCUS-0.5M), 2 (2M models) and 4 (LoCUS-7M), four times as many for 16 kb inputs and for the two
Genomic Benchmarks tasks with long sequences; `scripts/reproduce.py` uses the same values.

## Benchmarks

| Benchmark | Data | Run |
|---|---|---|
| Nucleotide Transformer, revised (18 tasks, MCC) | `benchmarks/nt18/prepare.py` | `scripts/finetune.py`, folds 0-9 |
| NT18 tested on chromosomes outside pretraining | `benchmarks/nt18/heldout_split.py` | `scripts/finetune.py` with the official test set as an extra test set, folds 0-2 |
| Genomic Benchmarks (9 tasks, accuracy) | `benchmarks/genomic_benchmarks/prepare.py` | `scripts/finetune.py`, folds 0-2 |
| Long-range benchmark, eQTL and bulk RNA (frozen embeddings) | `benchmarks/long_range/frozen.py prepare` | `benchmarks/long_range/frozen.py embed / fit` |
| Context-length ablation (long-range enhancer task, 4 kb and 16 kb input) | `benchmarks/long_range/prepare_enhancer.py` | `scripts/finetune.py` with `configs/finetune/long-range-enhancer-ep20.yaml` (and `-ep20-constant.yaml` for LoCUS-2M-16k with a constant learning rate) |
| Tokenizer study: sequencing errors and PBSIM3 reads | `benchmarks/noisy_reads/make_errors.py`, `make_controls.py`, `simulate_pbsim.py` | `scripts/finetune.py --save-best`, then `benchmarks/noisy_reads/score.py` |
| Parsing: edit locality and context independence of LCP, fixed and BPE segments | genome windows (`scripts/prepare_genome.sh`) | `benchmarks/parsing/locality.py` |
| BEND zero-shot variant effect prediction | `benchmarks/bend/zero_shot_vep.py prepare` | `benchmarks/bend/zero_shot_vep.py embed / analyze` |
| Fine-tuning throughput (LoCUS and the baselines, inputs prepared before timing) | | `benchmarks/efficiency/finetune_speed.py` |

Each script's exact arguments for every run of the paper are printed by `python scripts/reproduce.py <plan> --dry-run`.

**Tokenizer study:**
- The three 2M twins (LoCUS-2M, twin-fixed-2m and twin-bpe-2m) are fine-tuned on clean data and selected on clean
  validation data; the selected weights are kept (`--save-best`).
- `benchmarks/noisy_reads/score.py` scores the selected weights on the modified test sets: nanopore-like errors at 1%
  and 3%, the controls (substitutions only, errors outside the label window) and PBSIM3 reads. The tables report the
  errors outside the label window for the three splice tasks only (no edit within 100 bp of the splice junction, which
  sits at the read centre); the other tasks have no label region that could be protected reliably.
- Every modified test set is drawn 10 times with independent seeds (`<set>`, `<set>-d1` to `<set>-d9`), and the score of
  a fold on a set is the mean over the 10 draws: one draw of errors on a small test set (promoter_tata has 212 sequences)
  can by chance hit or spare the bases that carry the label, and all folds of a task are scored on the same sets.
- Results go to `results/tokenizer_study/<lcp|fixed|bpe>/nt18/<task>/`. The NT18 columns of LoCUS-2M, twin-fixed-2m and
  twin-bpe-2m in the main table are these runs.
- `benchmarks/parsing/locality.py` compares the three segmenters without a model: how many segments, and how far from
  the edit, one substitution, insertion or deletion changes, and whether a stretch is parsed the same way in two
  different contexts.

**Held-out chromosomes:** the NT18 test sets lie on chromosomes 20 and 21, which are among the pretraining chromosomes
(3-22). `benchmarks/nt18/heldout_split.py` moves every training example on a chromosome outside pretraining (1, 2, X, Y)
into a new test set. LoCUS-2M and its random-initialization control are fine-tuned on the remaining examples, and the
held-out and the official test set are both scored at the same selected epoch (plan `heldout`).

LoCUS-2M-16k is evaluated only where the context length is the question: the context-length ablation and the frozen
eQTL task at 4 kb and 16 kb input.

## Result tables

```bash
python scripts/collect_results.py --results results --output results/tables
```

This writes every table of the paper that `results/` has results for to `results/tables.md`, and every number to
`results/tables.json`. Fine-tuning cells are the mean and standard deviation over the folds that were run (10 for NT18
in the paper, 3 for the other benchmarks); a cell with fewer finished folds shows their mean and count instead.

The paired drop tests of the tokenizer study use a bootstrap over tasks (4,000 resamples). The folds of a task are kept
together because they share its test set.

## Reproducibility

- **Tested on:** Red Hat Enterprise Linux 9.8 (x86-64, Intel Xeon Silver 4216), gcc 11.5, Python 3.9 with PyTorch 2.6.0
  built for CUDA 11.8, and NVIDIA GPUs. Pretraining ran on RTX A5000 GPUs, except LoCUS-7M on an RTX 6000 Ada; the
  fine-tunes ran on RTX A5000, RTX A6000 and RTX 6000 Ada GPUs, whichever was free (most on RTX A5000).
  `install.sh --baselines` was also tested on Python 3.10 without
  the CUDA toolkit (prebuilt wheels; the same Caduceus fine-tuning result as on Python 3.9). Other GPUs (other NVIDIA
  models, AMD) and other CPUs give slightly different floating-point results, as repeated GPU runs do.
- This repository is a reorganized version of the research code that produced the paper's results. Every part
  (segmentation, data pipeline, pretraining, fine-tuning, benchmarks, table collection) was checked against the research
  code: bit-identical results on CPU, and on GPU in PyTorch's deterministic mode; the pretrained models are the paper's
  weights.
- On CPU, pretraining and fine-tuning are bit-for-bit deterministic single-threaded (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`).
  With several threads, a few runs differ in the last digits from run to run.
- On GPU, the mean over covering segments (`index_add_`) uses floating-point atomics, so repeated runs differ slightly
  unless `torch.use_deterministic_algorithms(True)` is set.
- The paper's fine-tunes ran up to three at a time on one GPU (`--jobs-per-gpu 3`). Sharing a GPU does not change the
  computation, but a run alone on a GPU is not bit-identical to a shared one, just as two shared runs are not: repeated
  fine-tunes with the same seed differed by about 0.013 MCC per task and fold on average (alone vs shared and shared vs
  shared alike), more on small tasks such as promoter_tata, and by about 0.01 on 18-task means. `reproduce.py` sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for its jobs, as the paper's runs did, so that three fine-tunes fit
  on one 24 GB GPU; it changes only how memory is allocated.

## Repository layout

```
locus/            model, data pipeline, pretraining and fine-tuning code
configs/          pretraining (configs/pretrain) and fine-tuning (configs/finetune) configurations
scripts/          installation helpers, pretraining, fine-tuning, reproduction and table collection
benchmarks/       preparation and evaluation code of each benchmark
reference/        the paper's result tables, for comparison with a rerun
tools/            genome window splitter (C)
third_party/      lcptools and the htslib 1.24 FASTA reader headers (kseq.h)
tests/            unit tests
```

## Citation

The citation will be added with the publication.

## Changes to external code

`third_party/README.md` lists every external component, model and dataset with its exact version and every change made
to it: a local patch of lcptools (a core-type field and memory fixes), a Python 3.9 type-hint patch of causal-conv1d,
and the ways the baselines are fine-tuned with the LoCUS protocol instead of their authors' recipes. Everything else is
used unmodified.

## License

MIT (see `LICENSE`). `third_party/` keeps the licenses of the included libraries.
