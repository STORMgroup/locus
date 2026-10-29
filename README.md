# LoCUS

LoCUS is a family of small DNA language models. Instead of k-mers or BPE tokens, LoCUS reads DNA as segments produced by
locally consistent parsing (LCP). Each segment depends only on the sequence around it, so a substitution, insertion or
deletion changes just the few segments next to it.

This repository holds the code behind the paper: pretraining, fine-tuning, the benchmarks, the baseline models, and the
scripts that rebuild the result tables.

## Quick start

```bash
git clone https://github.com/STORMgroup/locus.git && cd locus
bash install.sh --models        # environment, LCP library, tests, and the four pretrained models
source .venv/bin/activate
```

```python
import locus

model = locus.load("locus-2m")                           # downloaded from the Hugging Face Hub on first use
vectors = model.embed(["ACGTAGGCTA" * 50])                # [sequences, 128]: mean over bases, both strands averaged
per_base = model.embed(["ACGTAGGCTA" * 50], per_base=True)  # list of [length, 128] arrays
print(vectors.shape, per_base[0].shape)                   # (1, 128) (500, 128)
```

To fine-tune a released model on one task and fold:

```bash
python benchmarks/nt18/prepare.py
python scripts/finetune.py --pretrained locus-2m --recipe configs/finetune/locus-2m.yaml \
    --data-root data/benchmarks/nt18 --task H3K27ac_revised --fold 0 \
    --output results/locus-2m/nt18/H3K27ac_revised/fold-00.json
```

## Reproducing the paper

```bash
bash install.sh --all                                             # adds the baselines' CUDA kernels and PBSIM3
source .venv/bin/activate
python scripts/reproduce.py all --gpus 0,1,2,3                    # all experiments; NT18 on 10 folds, as in the paper
python scripts/compare_results.py                                 # how far your numbers are from the paper's
```

`scripts/reproduce.py` downloads the data each experiment needs, runs the jobs on the GPUs you list, and writes the
tables to `results/tables.md` (all numbers are also in `results/tables.json`). Finished jobs are skipped and interrupted
fine-tunes resume from their last epoch, so after a crash you can simply run the same command again. `--list` shows
progress, `--dry-run` prints the commands without running them, and a plan name runs one experiment:

| Plan | What it runs | Jobs | GPU time (RTX A5000, one job per GPU) |
|---|---|---|---|
| `nt18` | LoCUS-0.5M and LoCUS-2M on the revised NT benchmark (18 tasks, 10 folds) | 360 | about 13 / 17 min per fold |
| `random-init` | pretraining ablation: LoCUS-2M fine-tuned from random weights | 180 | about 17 min per fold |
| `baselines` | Caduceus-PS 1.9M, HyenaDNA-tiny 0.44M and HyenaDNA-small 3.3M on NT18, same protocol; HyenaDNA-small with two recipes | 720 | about 71 / 12 / 29 min per fold |
| `genomic-benchmarks` | LoCUS-0.5M and LoCUS-2M on the Genomic Benchmarks (9 tasks, 3 folds) | 54 | |
| `tokenizer-study` | the three 2M twins (LCP, BPE, fixed segments) on NT18, then scored on the test sets with errors | 1,080 | about 17 min per fold, 6 min per scoring |
| `gue-splice` | the same study on GUE splice sites, on all test sequences and on those sharing no splice site with NT18 | 90 | |
| `real-reads` | the twins scored on real HG002 nanopore reads of the NT18 test windows (needs network; extraction about 25 min) | 1,080 (540 shared with `tokenizer-study`) | |
| `single-nucleotide` | the baselines scored on the same error sets and real reads | 2,160 (720 shared with `baselines`) | |
| `intervals` | confidence intervals from resampling test windows (CPU) | 3 | minutes |
| `heldout` | LoCUS-2M and its random-init control tested on chromosomes that pretraining never saw | 108 | about 17 min per fold |
| `long-range` | frozen embeddings on the eQTL and bulk RNA tasks of the long-range benchmark | 9 | |
| `parsing` | how LCP, BPE and fixed segments react to one edit, and statistics of the LCP parse (CPU) | 2 | about 2 and 10 CPU minutes |
| `stability` | how far one edit spreads through the frozen 2M twins | 1 | about 40 min |
| `bend` | BEND zero-shot variant effect prediction | 5 | |
| `efficiency` | fine-tuning throughput and segmentation time (CPU for the latter) | 2 | minutes |
| `timing` | wall clock of full fine-tunes, fold 0 of each NT18 task, one job per GPU | 90 | |
| `pretrain` | the four pretraining runs (not part of `all`) | 4 | 11-19 h per model |

A few things to know:
- **Folds.** The NT18 plans and `gue-splice` run all 10 folds, as in the paper. `--nt18-folds 3` runs folds 0-2 in about
  a third of the time. `heldout` and the Genomic Benchmarks always use folds 0-2.
- **Sharing GPUs.** The paper ran up to three fine-tunes on one GPU; `--jobs-per-gpu 3` does the same. With three jobs
  per GPU on 12 GPUs, the 10-fold NT18 plans take about two days.
- **Comparing with the paper.** `reference/paper_tables.md` and `.json` hold the paper's numbers, and
  `scripts/compare_results.py` reports how far a rerun is from them. Fine-tuning on a GPU is not bit-for-bit repeatable
  (see Reproducibility), so expect 18-task means within about 0.01 of the paper, and larger differences on small tasks such
  as promoter_tata.

To pretrain the models yourself as well:

```bash
python scripts/reproduce.py pretrain --gpus 0,1,2,3          # writes runs/<model>/checkpoints/final.pt
python scripts/reproduce.py all --models-from runs --results results-own --gpus 0,1,2,3
```

## Model

A LoCUS model works in four steps:
1. **Base encoder.** Each base is embedded, then two layers of windowed self-attention (radius 32 and 128 bases) give it
   context.
2. **Segment pooling.** The sequence is cut into LCP level-2 segments, about 10 bp long and overlapping. The base vectors
   of each segment are pooled into one vector, and a small convolution lets each segment see its neighbours.
3. **Segment transformer.** Six transformer layers run over the segments, with rotary position embeddings at the segment
   centres.
4. **Per-base output.** Each base gets the mean of the segments that cover it.

Pretraining masks whole segments and predicts their bases.

| Model | Parameters | Width | Context | Segments |
|---|---|---|---|---|
| LoCUS-0.5M | 471,556 | 64 | 4 kb | LCP level 2 |
| LoCUS-2M | 1,737,604 | 128 | 4 kb | LCP level 2 |
| twin-fixed-2m | 1,737,604 | 128 | 4 kb | fixed 10 bp segments every 5 bp (tokenizer study) |
| twin-bpe-2m | 1,737,604 | 128 | 4 kb | BPE tokens, vocabulary 4,096 (tokenizer study) |

All four are pretrained on GRCh38 chromosomes 3-22. Chromosomes 1, 2, X and Y are left out, and there is no validation
during pretraining: the final checkpoint is the model. Each run is 359,466 steps of 16 crops of 256 to 4,096 bp, about
12.5 billion bases.

## Pretrained models

The four models are in one Hugging Face Hub repository,
[`STORMResearch/locus`](https://huggingface.co/STORMResearch/locus), one folder per model (`locus-0.5m/`, `locus-2m/`,
`twin-fixed-2m/`, `twin-bpe-2m/`). Each folder has `config.json` and `model.safetensors`.

- **By name:** `locus.load(...)`, `scripts/finetune.py --pretrained ...` and the benchmark scripts accept the model name
  and download the files the first time.
- **Ahead of time:** `python scripts/download_models.py`.
- **Offline:** point `LOCUS_HUB_DIR` to a directory with the same folders.
- **Your own checkpoints:** anywhere a model name works, you can also pass a `checkpoints/final.pt` from
  `scripts/pretrain.py`.
- **Publishing (authors only):** `scripts/upload_to_hub.py` converts pretraining checkpoints to this layout and, with
  `--upload`, pushes each one to its folder of the repository.

## Baseline models

The baselines are public pretrained models, fine-tuned with the same protocol as LoCUS: same folds, seeds, strand
handling and selection (`locus/finetune/baselines.py`, `scripts/finetune_baseline.py`). Their Hugging Face repositories
are pinned to the revisions used in the paper.

| Baseline | Hugging Face repository | Revision | Recipe |
|---|---|---|---|
| `caduceus-ps-1.9m` | kuleshov-group/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 | 2de3650 | LoCUS-2M recipe, float32 |
| `hyenadna-tiny-0.44m` | LongSafari/hyenadna-tiny-1k-seqlen-hf | e8c1eff | LoCUS-2M recipe, float32 |
| `hyenadna-small-3.3m` | LongSafari/hyenadna-small-32k-seqlen-hf | 8fe770c | learning rate 3e-4, 10% warm-up, float32 |
| `hyenadna-small-3.3m-standard` | the same model and revision | 8fe770c | LoCUS-2M recipe, float32 |

```bash
bash install.sh --baselines                       # mamba-ssm and causal-conv1d (Caduceus), and the three models
python scripts/finetune_baseline.py --baseline hyenadna-tiny-0.44m --data-root data/benchmarks/nt18 \
    --task H3K27ac_revised --fold 0 --output results/hyenadna-tiny-0.44m/nt18/H3K27ac_revised/fold-00.json
```

HyenaDNA-small is reported with two recipes: its own (learning rate 3e-4 with 10% warm-up, which our larger models needed)
and, as `hyenadna-small-3.3m-standard`, the recipe of the other baselines (`--recipe
configs/finetune/hyenadna-small-3.3m-standard.yaml`). With the higher learning rate it is clearly worse on the splice-site
tasks.

The `baselines` plan adds `--save-best`, which keeps the selected weights (`fold-XX.best.pt`) next to each result, so
`benchmarks/noisy_reads/score.py` can score them on the error sets just like the twins.

Caduceus needs a CUDA GPU. Its kernels, mamba-ssm and causal-conv1d (`requirements-baselines.txt`), come as prebuilt
wheels for Python 3.10 and newer. On Python 3.9 they are compiled instead, which takes about 50 minutes and needs the
CUDA toolkit (`nvcc`). causal-conv1d 1.7.0 also uses type hints that Python 3.9 does not accept, so `install.sh` builds it
with `third_party/causal-conv1d-1.7.0-python3.9.patch`, which changes only those hints (as in the paper's environment).

## Installation

What you need:
- Linux on x86-64, Python 3.9 or newer (the paper used 3.9), gcc and make.
- An NVIDIA GPU for fine-tuning. One fine-tune needs at most about 6 GB of GPU memory, so three fit on a 24 GB GPU.
- About 6 GB of disk for data (GRCh38 0.9 GB and its windows 1.4 GB, hg19 0.9 GB for the bulk RNA task, the benchmarks
  and their modified test sets about 1 GB) and about 15 GB for results, most of it the fine-tuned weights kept for the
  tokenizer study. Data preparation needs at most about 5 GB of RAM. These sizes come from the paper's runs.

`bash install.sh` creates `.venv`, installs the pinned packages, builds the native code and runs the unit tests. Options:
`--models` (download the LoCUS models), `--baselines` (baseline kernels and models), `--pbsim3` (build PBSIM3 for the
simulated reads) and `--all`. torch 2.6.0 from PyPI is built for CUDA 12.4; to get the CUDA 11.8 build the paper used,
run `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 bash install.sh`.

The same steps by hand:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # the paper used Python 3.9 and PyTorch 2.6.0 with CUDA 11.8
pip install -e . --no-deps
make                                 # builds build/lcptools (the LCP library) and build/split_fasta
pytest                               # unit tests, CPU only, about one minute
```

`third_party/lcptools` is [lcptools](https://github.com/BilkentCompGen/lcptools) at commit 9dcc05f with a small local
patch (`third_party/lcptools/local-changes.patch`). The patch adds the core type to `struct core`, which the Python
bindings in `locus/segmentation.py` need, and fixes how cores are released in `lps_deepen1_dct_iters`. `make lcptools`
builds the library with the paper's settings (32-bit labels and positions, dictionary encoding) and gives a byte-identical
library, even when an environment module sets `LD_RUN_PATH`. `make pbsim3` builds
[PBSIM3](https://github.com/yukiteruono/pbsim3) at commit 35ce0de in `build/pbsim3`.

## Data

`scripts/reproduce.py` prepares the data it needs. To run the steps yourself:

```bash
bash scripts/prepare_genome.sh                     # GRCh38.p14 -> data/GRCh38.p14.acgt.w4096.s4096
python benchmarks/nt18/prepare.py                  # revised Nucleotide Transformer benchmark -> data/benchmarks/nt18
python benchmarks/genomic_benchmarks/prepare.py    # Genomic Benchmarks -> data/benchmarks/genomic_benchmarks
python benchmarks/gue/prepare.py                   # GUE splice sites -> data/benchmarks/gue (needs data/benchmarks/nt18)
bash scripts/get_real_reads.sh                     # real nanopore reads of the NT18 test windows -> data/benchmarks/real_reads
```

- **Genome.** GRCh38.p14 is downloaded from NCBI, checked against its MD5 checksum, and cut into consecutive 4,096 bp
  windows that contain only A, C, G and T. Only pretraining needs the windows; `--download-only` skips them.
- **hg19.** The bulk RNA task of the long-range benchmark uses hg19. `benchmarks/long_range/frozen.py prepare --task
  bulk_rna` downloads UCSC's `hg19.fa.gz` (0.9 GB, MD5-checked) to `data/hg19.fa.gz` if it is missing.
- **Benchmarks.** The datasets are pinned to the revisions used in the paper, and the BEND variant files are checked
  against their SHA-256.
- **BPE tokenizer.** The tokenizer of twin-bpe-2m is included as `data/tokenizers/bpe-4096-1gb-seed42.json`;
  `scripts/train_bpe.py` shows how it was trained.

## Pretraining

```bash
python scripts/pretrain.py --config configs/pretrain/locus-2m.yaml --output-dir runs/locus-2m
python scripts/pretrain.py --config configs/pretrain/locus-2m.yaml --output-dir runs/locus-2m --resume   # continue
```

Pretraining uses one GPU, AdamW (learning rate 6e-4, 3,600 warm-up steps, then linear decay to zero) and fp16 mixed
precision. Windows are read in genome order through a shuffle buffer of 512 crops. Every random choice is seeded by the
window and the pass, so a resumed run sees the same examples as an uninterrupted one. The output directory holds
`train_metrics.jsonl` and the checkpoints; `checkpoints/final.pt` is the pretrained model.

## Fine-tuning

```bash
python scripts/finetune.py --pretrained locus-2m --recipe configs/finetune/locus-2m.yaml \
    --data-root data/benchmarks/nt18 --task H3K27ac_revised --fold 0 --micro-batches 2 \
    --output results/locus-2m/nt18/H3K27ac_revised/fold-00.json
```

`--pretrained` takes a released model name (`locus-2m`), a directory in the Hub layout, or a checkpoint from
`scripts/pretrain.py`. `--random-init` keeps only the architecture and segmenter and starts from random weights (the
pretraining ablation).

**Recipe.** All tasks use the same recipe:
- 20 epochs, batch 128, AdamW with weight decay 0.1.
- Learning rate 1e-3 with 1% warm-up, then cosine decay to 0.1 of the peak.
- Dropout 0.1, bf16.
- Each training sequence is read on a random strand; at evaluation the logits of both strands are averaged.

**Folds and selection.**
- Fold k validates on fold k of a stratified 10-fold split of the training set (seed 42) and trains with seed 42 + k. The
  paper uses all 10 folds on NT18 and GUE splice, and folds 0-2 on the Genomic Benchmarks and the held-out-chromosome
  check.
- We keep the epoch with the lowest validation loss. The test set is scored after every epoch but never used to choose
  the epoch.

`--micro-batches N` splits each batch for gradient accumulation. The update stays the same; it only saves memory. The
paper used 1 for LoCUS-0.5M and 2 for the 2M models (four times as many for the two Genomic Benchmarks tasks with long
sequences), and `scripts/reproduce.py` uses the same values.

## Benchmarks

| Benchmark | Data | Run |
|---|---|---|
| Nucleotide Transformer, revised (18 tasks, MCC) | `benchmarks/nt18/prepare.py` | `scripts/finetune.py`, folds 0-9 |
| NT18 tested on chromosomes outside pretraining | `benchmarks/nt18/heldout_split.py` | `scripts/finetune.py` with the official test set as an extra test set, folds 0-2 |
| Genomic Benchmarks (9 tasks, accuracy) | `benchmarks/genomic_benchmarks/prepare.py` | `scripts/finetune.py`, folds 0-2 |
| Long-range benchmark, eQTL and bulk RNA (frozen embeddings) | `benchmarks/long_range/frozen.py prepare` | `benchmarks/long_range/frozen.py embed / fit` |
| Tokenizer study: sequencing errors and PBSIM3 reads | `benchmarks/noisy_reads/make_errors.py`, `make_controls.py`, `simulate_pbsim.py` | `scripts/finetune.py --save-best`, then `benchmarks/noisy_reads/score.py` |
| Tokenizer study on GUE splice sites (400 bp, 3 classes, MCC) | `benchmarks/gue/prepare.py`, the same scripts with `--tasks gue_splice`, `benchmarks/gue/independent_sets.py` | `scripts/finetune.py --save-best`, folds 0-9, then `benchmarks/noisy_reads/score.py` |
| Tokenizer study on real nanopore reads (HG002, R10.4.1, sup and hac basecalls) | `scripts/get_real_reads.sh` | `benchmarks/noisy_reads/score.py --output-dir` on the tokenizer study's NT18 folds |
| Single-nucleotide models on the error sets and real reads | the sets of the two rows above | `scripts/finetune_baseline.py --save-best`, then `benchmarks/noisy_reads/score.py --output-dir --save-predictions` |
| Intervals from resampling test windows | the saved predictions (`score.py --save-predictions`) | `benchmarks/noisy_reads/window_intervals.py` |
| Parsing: how one edit changes LCP, fixed and BPE segments | genome windows (`scripts/prepare_genome.sh`) | `benchmarks/parsing/locality.py` |
| LCP parse statistics: core types, segment lengths, coverage, worst case of one edit | genome windows | `benchmarks/parsing/lcp_statistics.py` |
| Embedding stability after one edit (frozen 2M twins) | genome windows | `benchmarks/embedding/stability.py` |
| BEND zero-shot variant effect prediction | `benchmarks/bend/zero_shot_vep.py prepare` | `benchmarks/bend/zero_shot_vep.py embed / analyze` |
| Fine-tuning throughput (LoCUS and the baselines, inputs prepared before timing) | | `benchmarks/efficiency/finetune_speed.py` |
| Segmentation time of the NT18 data per segmenter (CPU) | `benchmarks/nt18/prepare.py` | `benchmarks/efficiency/segmentation_time.py` |

`python scripts/reproduce.py <plan> --dry-run` prints the exact arguments of each run in the paper.

### The tokenizer study

The study asks how much the choice of tokenizer matters when the test sequences contain sequencing errors.
- **Three twins.** LoCUS-2M, twin-fixed-2m and twin-bpe-2m differ only in how they cut the sequence. They are fine-tuned
  and selected on clean data, and their selected weights are kept (`--save-best`). Their NT18 results are also the
  LoCUS-2M, twin-fixed-2m and twin-bpe-2m columns of the main table (`results/tokenizer_study/<lcp|fixed|bpe>/nt18/<task>/`).
- **Test sets with errors.** `benchmarks/noisy_reads/score.py` scores the kept weights on modified copies of each test
  set: nanopore-like errors at an event rate of 1% and 3% per base (about 1.2% and 3.5% of bases edited), two controls
  (substitutions only, and errors outside the label window) and PBSIM3 reads. The label-window control is reported for
  the three splice tasks only: there, no edit comes within 100 bp of the splice junction at the centre of the sequence.
  The other tasks have no label region that could be protected reliably.
- **Ten draws.** Each modified set is drawn 10 times with different seeds (`<set>`, `<set>-d1` to `<set>-d9`), and a
  fold's score is the mean over the draws. A single draw on a small test set (promoter_tata has 212 sequences) can, by
  chance, hit or miss the bases that carry the label.
- **GUE splice sites.** The same study on a second splice benchmark (`benchmarks/gue/prepare.py`: DNABERT-2's GUE,
  pinned Hugging Face copy; the junction is between positions 199 and 200 of each 400 bp sequence). The twins are
  fine-tuned on its training set with the same recipe and 10-fold validation split; GUE's own dev set is not used. 564 of
  its 4,562 test sequences share their splice site with an NT18 splice record, so `benchmarks/gue/independent_sets.py`
  also writes every test set without them, and the tables report both. Results: `results/tokenizer_study/<twin>/gue/gue_splice/`
  and `results/gue_independent/`.
- **Real reads.** The NT18 test windows (all on chromosomes 20 and 21) are cut out of real HG002 R10.4.1 nanopore reads,
  in two basecalls of the same molecules (sup and hac): 10 reads per window, with the reference sequences of the same
  windows as the clean comparison. Only primary alignments with MAPQ of at least 20 are used, so windows in repeats
  without 10 uniquely mapped reads (mostly negatives) are left out. Reads with an insertion of 50 bp or more inside the
  window, or a deletion of 50 bp or more overlapping it, are not used. `scripts/get_real_reads.sh` runs the extraction
  (network, about 25 min) and checks that every set is identical to the paper's (`benchmarks/noisy_reads/real_reads_sha256.tsv`).
  Results: `results/real_reads/`.
- **Without a model.** `benchmarks/parsing/locality.py` compares the three segmenters directly: how many segments one
  substitution, insertion or deletion changes, how far from the edit, and whether the same stretch is parsed the same
  way in two different contexts. `benchmarks/parsing/lcp_statistics.py` describes the LCP parse itself: level-1 core
  types and spacing; length, spacing and coverage of the level-2 segments (how many segments cover each base, and how
  many bases at the ends none covers); and the worst case of one edit over 10,000 chromosome-1 and 10,000 random
  sequences.
- **Frozen embeddings.** `benchmarks/embedding/stability.py` applies one edit at the centre of 10,000 chromosome-1
  sequences and measures, for the three frozen twins, the changed segments, the change of each per-base embedding by
  distance from the edit (relative to the change between unrelated bases of the same model), and the change of the read
  embedding.
- **Comparisons.** The paired drop tests compare LCP's drops with those of BPE and fixed segments for each NT18 group
  (histone marks, promoters, splice sites, enhancers) and over all 18 tasks, for the error sets and the substitution-only
  controls.

**Held-out chromosomes.** The NT18 test sets lie on chromosomes 20 and 21, which are among the pretraining chromosomes
(3-22). `benchmarks/nt18/heldout_split.py` moves every training example on a chromosome that pretraining did not use (1,
2, X, Y) into a new test set. LoCUS-2M and its random-init control are fine-tuned on the remaining examples, and both test
sets are scored at the same selected epoch (plan `heldout`).

## Result tables

```bash
python scripts/collect_results.py --results results --output results/tables
```

This writes each table of the paper that `results/` has results for to `results/tables.md`, and all numbers to
`results/tables.json`.
- Fine-tuning cells show the mean and standard deviation over the folds: 10 folds for NT18, the tokenizer study, GUE
  splice and the real reads (`--nt18-folds`), 3 for the other benchmarks. A cell with missing folds shows the mean and the
  number of folds instead, and the script warns about missing folds in the NT18, GUE splice and real-read tables.
- The table "Tokenizer study: modified test sets" explains the set names (ont1, sub3, pbsim97, ...).
- The paired drop tests give each drop as the mean over task-folds and the standard deviation over folds, with a 95%
  interval from a bootstrap over tasks (4,000 resamples). The folds of a task stay together because they share its test
  set. The held-out-chromosome table adds the pretraining gain (LoCUS-2M minus random init) on both test sets, with the
  same kind of interval for their difference.
- The single-nucleotide table puts LCP's drops next to those of the baselines (Caduceus-PS 1.9M, HyenaDNA-tiny 0.44M and
  HyenaDNA-small 3.3M with both recipes), whose tokens are single nucleotides, on the same task-folds.
- The interval tables (plan `intervals`) resample test windows instead of tasks: each replicate draws the windows of a
  task group with replacement, with the same weights for every model, fold, set and draw. The reference tables have these
  intervals for the splice and promoter tasks, whose twin predictions were saved for the paper.

## Reproducibility

- **Tested on** Red Hat Enterprise Linux 9.8 (x86-64, Intel Xeon Silver 4216), gcc 11.5, Python 3.9 and PyTorch 2.6.0
  built for CUDA 11.8, with NVIDIA RTX A5000 GPUs. `install.sh --baselines` was also tested on Python 3.10 without the
  CUDA toolkit, using the prebuilt wheels, and gave the same Caduceus fine-tuning result. Other GPUs and CPUs give slightly
  different floating-point results, as repeated GPU runs do.
- **Same code as the paper.** This repository is a reorganized version of the research code behind the paper. Each part
  (segmentation, data pipeline, pretraining, fine-tuning, benchmarks, tables) was checked against the research code:
  bit-identical results on CPU, and on GPU in PyTorch's deterministic mode. The pretrained models are the paper's weights.
- **CPU.** Pretraining and fine-tuning are bit-for-bit repeatable with one thread (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`).
  With several threads, a few runs differ in the last digits.
- **GPU.** The mean over covering segments (`index_add_`) uses floating-point atomics, so repeated runs differ slightly
  unless `torch.use_deterministic_algorithms(True)` is set.
- **Shared GPUs.** The paper's fine-tunes ran up to three at a time on one GPU (`--jobs-per-gpu 3`). Sharing does not
  change the computation, but a run alone on a GPU is not bit-identical to a shared one, just as two shared runs are not.
  Repeated fine-tunes with the same seed differed by about 0.013 MCC per task and fold on average (alone vs shared, and
  shared vs shared), more on small tasks such as promoter_tata, and by about 0.01 on 18-task means. `reproduce.py` sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for its jobs, as the paper's runs did, so that three fine-tunes fit
  on one 24 GB GPU; this only changes how memory is allocated.
- **Test sets with errors.** The error sets and controls are generated from fixed seeds, and PBSIM3 runs with a fixed
  seed at its pinned commit. Regenerated from the NT18 download, they are byte-identical to the paper's. PBSIM3 draws its
  random numbers from the C library, so this holds on Linux (glibc). The real reads are checked against the paper's
  checksums by `scripts/get_real_reads.sh`.

## Repository layout

```
locus/            model, data pipeline, pretraining and fine-tuning code
configs/          pretraining (configs/pretrain) and fine-tuning (configs/finetune) configurations
scripts/          installation helpers, pretraining, fine-tuning, reproduction and table collection
benchmarks/       data preparation and evaluation for each benchmark
reference/        the paper's result tables, to compare a rerun against
tools/            genome window splitter (C)
third_party/      lcptools and the htslib 1.24 FASTA reader header (kseq.h)
tests/            unit tests
```

## Citation

The citation will be added when the paper is published.

## Changes to external code

`third_party/README.md` lists each external component, model and dataset with its version and any change we made: a
local patch of lcptools (a core-type field and memory fixes), a Python 3.9 type-hint patch of causal-conv1d, and how the
baselines are fine-tuned with the LoCUS protocol instead of their authors' recipes. Everything else is used unmodified.

## License

GNU General Public License v3.0 (see `LICENSE`). `third_party/` keeps the licenses of the included libraries. The
genomes, benchmarks, nanopore reads and baseline models are not part of this repository: the scripts download them from
their sources, and they stay under their own licenses and terms of use.
