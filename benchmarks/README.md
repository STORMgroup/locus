# Reproducing the results

This guide explains how to regenerate every result table of the paper: what to install, which data are downloaded, what
each experiment runs, how long it takes, and how the tables are made. All commands are run from the repository root.

[Requirements](#requirements) · [Running the experiments](#running-the-experiments) · [Data](#data) ·
[Pretraining](#pretraining) · [Fine-tuning](#fine-tuning) · [Baselines](#baselines) · [Benchmarks](#benchmarks) ·
[The tokenizer study](#the-tokenizer-study) · [The pretraining-split ablation](#the-pretraining-split-ablation) ·
[Result tables](#result-tables) · [Reproducibility](#reproducibility)

## Requirements

```bash
bash install.sh --all
source .venv/bin/activate
```

`--all` installs everything `--models` does (see the main README), plus the baselines' CUDA kernels and models and
[PBSIM3](https://github.com/yukiteruono/pbsim3) at commit 35ce0de for the simulated reads. You also need:
- NVIDIA GPUs. One fine-tune needs at most about 6 GB of GPU memory, so three fit on a 24 GB GPU.
- About 5 GB of disk for data (GRCh38 0.9 GB and its windows 1.4 GB, the benchmarks and their modified test sets about
  1 GB) and about 15 GB for results, most of it the fine-tuned weights kept for the
  tokenizer study. Data preparation needs at most about 5 GB of RAM.
- Network access while the data and models are downloaded.

## Running the experiments

```bash
python scripts/reproduce.py all --gpus 0,1,2,3 --jobs-per-gpu 3   # every downstream experiment (not pretrain or the split)
python scripts/reproduce.py nt18 --gpus 0                          # one experiment (a "plan")
python scripts/reproduce.py --list                                 # the plans and how many of their jobs are done
python scripts/reproduce.py nt18 --dry-run --verbose               # the exact command of every job
python scripts/compare_results.py                                  # your tables against the paper's
```

`scripts/reproduce.py` downloads the data each plan needs, runs its jobs on the GPUs you list, and writes the tables to
`results/tables.md` (all numbers also to `results/tables.json`). Finished jobs are skipped and interrupted fine-tunes
resume from their last epoch, so after a crash you can run the same command again.

| Plan | What it runs | Jobs | GPU time (RTX A5000, one job per GPU) |
|---|---|---|---|
| `nt18` | LoCUS-0.5M and LoCUS-2M on the revised NT benchmark (18 tasks, 10 folds) | 360 | about 13 / 17 min per fold |
| `random-init` | pretraining ablation: LoCUS-2M fine-tuned from random weights | 180 | about 17 min per fold |
| `baselines` | Caduceus-PS 0.47M and 1.9M (the sizes of LoCUS-0.5M and LoCUS-2M) on NT18, same protocol and precision (bfloat16) as LoCUS, 10 folds | 360 | about 21 / 40 min per fold |
| `baselines-fp32` | the same runs in float32 (supplementary precision check) | 360 | about 29 / 71 min per fold |
| `baselines-authors` | Caduceus-PS 0.47M and 1.9M (the sizes of LoCUS-0.5M and LoCUS-2M) on NT18 with Caduceus's own fine-tuning settings (supplementary table), 10 folds | 360 | |
| `baselines-authors-bf16` | the folds of `baselines-authors` whose fp16 training overflowed, again in bfloat16 (after `baselines-authors`) | 64 in the paper (2 + 62) | |
| `genomic-benchmarks` | LoCUS-0.5M and LoCUS-2M on the Genomic Benchmarks (9 tasks, 10 folds) | 180 | |
| `tokenizer-study` | the three 2M twins (LCP, BPE, fixed segments) on NT18, then scored on the test sets with errors | 1,080 (180 shared with `nt18`: the LoCUS-2M runs) | about 17 min per fold, 6 min per scoring |
| `gue-splice` | the same study on GUE splice sites, on all test sequences and on those sharing no splice site with NT18 | 90 | |
| `real-reads` | the twins scored on real HG002 nanopore reads of the NT18 test windows (needs network; extraction about 25 min) | 1,080 (540 shared with `tokenizer-study`) | |
| `single-nucleotide` | Caduceus-PS 1.9M scored on the same error sets and real reads | 540 (180 shared with `baselines`) | |
| `single-nucleotide-fp32` | the same for its float32 runs (supplementary precision check) | 540 (180 shared with `baselines-fp32`) | |
| `heldout` | LoCUS-2M and its random-init control tested on chromosomes that pretraining never saw | 360 | about 17 min per fold |
| `parsing` | how LCP, BPE and fixed segments react to one edit, statistics of the LCP parse, and what the boundaries reveal about masked bases (CPU) | 3 | about 2, 10 and 40 CPU minutes |
| `stability` | how far one edit spreads through the frozen 2M twins | 1 | about 40 min |
| `efficiency` | fine-tuning throughput and segmentation time (CPU for the latter) | 2 | minutes |
| `timing` | wall clock of full fine-tunes, fold 0 of each NT18 task, one job per GPU; Caduceus-PS with the recipes of `baselines` (bfloat16) and `baselines-fp32` (float32) | 108 | |
| `pretrain` | the four pretraining runs (not part of `all`) | 4 | 9-19 h per model |
| `split-pretrain` | the pretraining-split ablation: the same four models pretrained without chromosomes 20 and 21 (not part of `all`) | 4 | 9-19 h per model |
| `split-nt18` | those four models on NT18, 10 folds: the released split models, or your own `split-pretrain` with `--models-from runs` (not part of `all`) | 720 | as `nt18` |
| `split-score` | the three 2M ones scored on the error sets and real reads (not part of `all`) | 1,620 (540 shared with `split-nt18`) | 6 min per scoring |

- **Folds.** Every fine-tuning plan runs all 10 folds, as in the paper. `--nt18-folds 3` runs folds 0-2 of the NT18 plans
  and `gue-splice` in about a third of the time, and `--other-folds 3` does the same for the Genomic Benchmarks and
  `heldout`.
- **Sharing GPUs.** The paper ran up to three fine-tunes on one GPU, as `--jobs-per-gpu 3` does. With three jobs per GPU
  on 12 GPUs, the 10-fold NT18 plans take about two days.
- **Comparing with the paper.** `reference/paper_tables.md` and `.json` hold the paper's numbers, and
  `scripts/compare_results.py` reports how far a rerun is from them. Fine-tuning on a GPU is not bit-for-bit repeatable
  (see [Reproducibility](#reproducibility)), so expect 18-task means within about 0.01 of the paper, and larger
  differences on small tasks such as promoter_tata.
- **The split ablation** has its own four models, released next to the others; its tables compare with the released
  models' runs of `nt18`, `tokenizer-study` and `real-reads`, so run those too:
  `python scripts/reproduce.py split-nt18 split-score --gpus 0,1,2,3` (see
  [The pretraining-split ablation](#the-pretraining-split-ablation)), after `split-pretrain` for your own pretraining.
- **Your own pretraining.** To pretrain the models as well and run every experiment on them:

  ```bash
  python scripts/reproduce.py pretrain --gpus 0,1,2,3          # writes runs/<model>/checkpoints/final.pt
  python scripts/reproduce.py all --models-from runs --results results-own --gpus 0,1,2,3
  ```

## Data

`scripts/reproduce.py` prepares the data it needs. Every dataset is downloaded from its public source and prepared by a
script; the only derived files in this repository are the two BPE tokenizers (below). To run the steps yourself:

```bash
bash scripts/prepare_genome.sh                     # GRCh38.p14 -> data/GRCh38.p14.acgt.w4096.s4096
python benchmarks/nt18/prepare.py                  # revised Nucleotide Transformer benchmark -> data/benchmarks/nt18
python benchmarks/genomic_benchmarks/prepare.py    # Genomic Benchmarks -> data/benchmarks/genomic_benchmarks
python benchmarks/gue/prepare.py                   # GUE splice sites -> data/benchmarks/gue (needs data/benchmarks/nt18)
bash scripts/get_real_reads.sh                     # real nanopore reads of the NT18 test windows -> data/benchmarks/real_reads
```

- **Genome.** GRCh38.p14 is downloaded from NCBI, checked against its MD5 checksum, and cut into consecutive 4,096 bp
  windows that contain only A, C, G and T. Only pretraining needs the windows; `--download-only` skips them.
- **Benchmarks.** The datasets are pinned to the revisions used in the paper.
- **BPE tokenizers.** The tokenizer of twin-bpe-2m is included as `data/tokenizers/bpe-4096-1gb-seed42.json`, and that of
  its split-ablation counterpart (trained the same way on chromosomes 1-19 and 22) as
  `data/tokenizers/bpe-4096-1gb-seed42-no-chr20-21.json`, because retraining one needs about 64 GB of RAM;
  `scripts/train_bpe.py` shows how they were trained (`--config` for the second).

## Pretraining

```bash
python scripts/pretrain.py --config configs/pretrain/locus-2m.yaml --output-dir runs/locus-2m
python scripts/pretrain.py --config configs/pretrain/locus-2m.yaml --output-dir runs/locus-2m --resume   # continue
```

| Model | Parameters | Width | Context | Segments |
|---|---|---|---|---|
| LoCUS-0.5M | 471,556 | 64 | 4 kb | LCP level 2 |
| LoCUS-2M | 1,737,604 | 128 | 4 kb | LCP level 2 |
| twin-fixed-2m | 1,737,604 | 128 | 4 kb | fixed 10 bp segments every 5 bp (tokenizer study) |
| twin-bpe-2m | 1,737,604 | 128 | 4 kb | BPE tokens, vocabulary 4,096 (tokenizer study) |

- **Data.** All four are pretrained on GRCh38 chromosomes 3-22. Chromosomes 1, 2, X and Y are left out, and there is no
  validation during pretraining: the final checkpoint is the model. Each run is 359,466 steps of 16 crops of 256 to
  4,096 bp, about 12.5 billion bases.
- **Objective.** Whole segments are masked and their bases predicted. The segment transformer uses rotary position
  embeddings at the segment centres.
- **Optimization.** One GPU, AdamW (learning rate 6e-4, 3,600 warm-up steps, then linear decay to zero) and fp16 mixed
  precision.
- **Order.** Windows are read in genome order through a shuffle buffer of 512 crops. Every random choice is seeded by the
  window and the pass, so a resumed run sees the same examples as an uninterrupted one.
- **Output.** The output directory holds `train_metrics.jsonl` and the checkpoints; `checkpoints/final.pt` is the
  pretrained model. `scripts/upload_to_hub.py` (authors only) converts it to the Hugging Face layout.

## Fine-tuning

```bash
python scripts/finetune.py --pretrained locus-2m --recipe configs/finetune/locus-2m.yaml \
    --data-root data/benchmarks/nt18 --task H3K27ac_revised --fold 0 --micro-batches 2 \
    --output results/examples/locus-2m/H3K27ac_revised/fold-00.json
```

`--pretrained` takes a released model name (`locus-2m`), a directory in the Hub layout, or a checkpoint from
`scripts/pretrain.py`. `--random-init` keeps only the architecture and segmenter and starts from random weights (the
pretraining ablation).

- **Recipe.** All tasks use the same recipe: 20 epochs, batch 128, AdamW with weight decay 0.1, learning rate 1e-3 with
  1% warm-up and cosine decay to 0.1 of the peak, dropout 0.1, bf16. Each training sequence is read on a random strand;
  at evaluation the logits of both strands are averaged.
- **Folds.** Fold k validates on fold k of a stratified 10-fold split of the training set (seed 42) and trains with seed
  42 + k. The paper uses all 10 folds.
- **Selection.** We keep the epoch with the lowest validation loss. The test set is scored after every epoch but never
  used to choose the epoch.
- **Memory.** `--micro-batches N` splits each batch for gradient accumulation. The update stays the same; it only saves
  memory. The paper used 1 for LoCUS-0.5M and 2 for the 2M models (four times as many for the two Genomic Benchmarks
  tasks with long sequences), and `scripts/reproduce.py` uses the same values.

## Baselines

The baselines are public pretrained models, fine-tuned with the same protocol as LoCUS: same folds, seeds, strand
handling and selection (`locus/finetune/baselines.py`, `scripts/finetune_baseline.py`). Their Hugging Face repositories
are pinned to the revisions used in the paper.

| Baseline | Hugging Face repository | Revision | Recipe |
|---|---|---|---|
| `caduceus-ps-1.9m` | kuleshov-group/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 | 2de3650 | LoCUS-2M recipe, bfloat16 |
| `caduceus-ps-0.47m` | kuleshov-group/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3 | 51d88e3 | LoCUS-2M recipe, bfloat16 |

```bash
bash install.sh --baselines                       # mamba-ssm and causal-conv1d (Caduceus), and the two models
python scripts/finetune_baseline.py --baseline caduceus-ps-1.9m --data-root data/benchmarks/nt18 \
    --task H3K27ac_revised --fold 0 --output results/examples/caduceus-ps-1.9m-bf16/H3K27ac_revised/fold-00.json
```

- **Precision.** The baselines fine-tune in bfloat16 autocast like LoCUS (`configs/finetune/caduceus-ps-<size>-bf16.yaml`,
  the default of `scripts/finetune_baseline.py`), each batch of 128 accumulated over micro-batches of 32. The
  `baselines-fp32` plan runs the same recipe, folds, seeds and training-data order in float32
  (`configs/finetune/caduceus-ps-<size>.yaml`); the supplementary table compares the two (bfloat16 - float32 on the
  18-task mean: +0.001 for 0.47M, +0.004 for 1.9M in the paper).
- **Caduceus's own settings (supplementary).** The `baselines-authors` plan fine-tunes Caduceus-PS at the sizes of
  LoCUS-0.5M (0.47M parameters) and LoCUS-2M (1.9M) on NT18 with Caduceus's own fine-tuning settings for the NT tasks
  (`configs/finetune/caduceus-ps-<size>-authors.yaml`; details in `third_party/README.md`): per task the learning rate and
  batch size of its paper's Table 6, its weight-decay groups, no dropout, the forward strand in training and evaluation,
  and fp16 mixed precision. The data, folds, seeds, 20 epochs and epoch selection stay those of every other model. It
  runs fold by fold, so an interrupted run leaves complete folds. The folds whose fp16 training overflowed (non-finite
  steps or validation losses) are run again in bfloat16 by the `baselines-authors-bf16` plan (2 of 180 for 0.47M and 62
  of 180 for 1.9M in the paper), and the table uses those runs for them.
- **Kept weights.** The `baselines` plan adds `--save-best`, which keeps the selected weights (`fold-XX.best.pt`) next to
  each result, so `benchmarks/noisy_reads/score.py` can score Caduceus-PS 1.9M on the error sets just like the twins
  (plan `single-nucleotide`; its float32 runs of `baselines-fp32`: plan `single-nucleotide-fp32`).
- **Caduceus** needs a CUDA GPU. Its kernels, mamba-ssm and causal-conv1d (`requirements-baselines.txt`), come as
  prebuilt wheels for Python 3.10 and newer. On Python 3.9 they are compiled instead, which takes about 50 minutes and
  needs the CUDA toolkit (`nvcc`). causal-conv1d 1.7.0 also uses type hints that Python 3.9 does not accept, so
  `install.sh` builds it with `third_party/causal-conv1d-1.7.0-python3.9.patch`, which changes only those hints (as in
  the paper's environment).

## Benchmarks

Each folder here prepares the data of one benchmark or runs one analysis:

| Benchmark | Data | Run |
|---|---|---|
| Nucleotide Transformer, revised (18 tasks, MCC) | `nt18/prepare.py` | `scripts/finetune.py`, folds 0-9 |
| NT18 tested on chromosomes outside pretraining | `nt18/heldout_split.py` | `scripts/finetune.py` with the official test set as an extra test set, folds 0-9 |
| Genomic Benchmarks (9 tasks, accuracy) | `genomic_benchmarks/prepare.py` | `scripts/finetune.py`, folds 0-9 |
| Tokenizer study: sequencing errors and PBSIM3 reads | `noisy_reads/make_errors.py`, `make_controls.py`, `simulate_pbsim.py` | `scripts/finetune.py --save-best`, then `noisy_reads/score.py` |
| Tokenizer study on GUE splice sites (400 bp, 3 classes, MCC) | `gue/prepare.py`, the same scripts with `--tasks gue_splice`, `gue/independent_sets.py` | `scripts/finetune.py --save-best`, folds 0-9, then `noisy_reads/score.py` |
| Tokenizer study on real nanopore reads (HG002, R10.4.1, sup and hac basecalls) | `scripts/get_real_reads.sh` | `noisy_reads/score.py --output-dir` on the tokenizer study's NT18 folds |
| Single-nucleotide models on the error sets and real reads | the sets of the two rows above | `scripts/finetune_baseline.py --save-best`, then `noisy_reads/score.py --output-dir` |
| Parsing: how one edit changes LCP, fixed and BPE segments | genome windows (`scripts/prepare_genome.sh`) | `parsing/locality.py` |
| LCP parse statistics: core types, segment lengths, coverage, worst case of one edit | genome windows | `parsing/lcp_statistics.py` |
| Masked bases recovered from the segment boundaries (pretrained models, random and chromosome-1 sequences) | genome windows | `parsing/boundary_information.py` |
| Embedding stability after one edit (frozen 2M twins) | genome windows | `embedding/stability.py` |
| Fine-tuning throughput (LoCUS and Caduceus-PS, inputs prepared before timing) | | `efficiency/finetune_speed.py` |
| Segmentation time of the NT18 data per segmenter (CPU) | `nt18/prepare.py` | `efficiency/segmentation_time.py` |

Paths without a folder are in `benchmarks/`. `python scripts/reproduce.py <plan> --dry-run --verbose` prints the exact
arguments of each run in the paper.

**Held-out chromosomes.** The NT18 test sets lie on chromosomes 20 and 21, which are among the pretraining chromosomes
(3-22). `nt18/heldout_split.py` moves every training example on a chromosome that pretraining did not use (1, 2, X, Y)
into a new test set. LoCUS-2M and its random-init control are fine-tuned on the remaining examples, and both test sets
are scored at the same selected epoch (plan `heldout`). The pretraining-split ablation (below) removes the test chromosomes
from pretraining instead.

## The tokenizer study

The study asks how much the choice of tokenizer matters when the test sequences contain sequencing errors.
- **Three twins.** LoCUS-2M, twin-fixed-2m and twin-bpe-2m differ only in how they cut the sequence. They are fine-tuned
  and selected on clean data, and their selected weights are kept (`--save-best`). Their NT18 results are also the
  LoCUS-2M, twin-fixed-2m and twin-bpe-2m columns of the main table (`results/tokenizer_study/<lcp|fixed|bpe>/nt18/<task>/`).
- **Test sets with errors.** `noisy_reads/score.py` scores the kept weights on modified copies of each test set:
  nanopore-like errors at an event rate of 1% and 3% per base (about 1.2% and 3.5% of bases edited), two controls
  (substitutions only, and errors outside the label window) and PBSIM3 reads. The label-window control is reported for
  the three splice tasks only: there, no edit comes within 100 bp of the splice junction at the centre of the sequence.
  The other tasks have no label region that could be protected reliably.
- **Ten draws.** Each modified set is drawn 10 times with different seeds (`<set>`, `<set>-d1` to `<set>-d9`), and a
  fold's score is the mean over the draws. A single draw on a small test set (promoter_tata has 212 sequences) can, by
  chance, hit or miss the bases that carry the label.
- **GUE splice sites.** The same study on a second splice benchmark (`gue/prepare.py`: DNABERT-2's GUE, pinned Hugging
  Face copy; the junction is between positions 199 and 200 of each 400 bp sequence). The twins are fine-tuned on its
  training set with the same recipe and 10-fold validation split; GUE's own dev set is not used. 564 of its 4,562 test
  sequences share their splice site with an NT18 splice record, so `gue/independent_sets.py` also writes every test set
  without them, and the tables report both. Results: `results/tokenizer_study/<twin>/gue/gue_splice/` and
  `results/gue_independent/`.
- **Real reads.** The NT18 test windows (all on chromosomes 20 and 21) are cut out of real HG002 R10.4.1 nanopore reads,
  in two basecalls of the same molecules (sup and hac): 10 reads per window, with the reference sequences of the same
  windows as the clean comparison. Only primary alignments with MAPQ of at least 20 are used, so windows in repeats
  without 10 uniquely mapped reads (mostly negatives) are left out. Reads with an insertion of 50 bp or more inside the
  window, or a deletion of 50 bp or more overlapping it, are not used. `scripts/get_real_reads.sh` runs the extraction
  (network, about 25 min) and checks that every set is identical to the paper's (`noisy_reads/real_reads_sha256.tsv`).
  Results: `results/real_reads/`.
- **Without a model.** `parsing/locality.py` compares the three segmenters directly: how many segments one substitution,
  insertion or deletion changes, how far from the edit, and whether the same stretch is parsed the same way in two
  different contexts. `parsing/lcp_statistics.py` describes the LCP parse itself: level-1 core types and spacing; length,
  spacing and coverage of the level-2 segments (how many segments cover each base, and how many bases at the ends no
  level-1 core or segment covers); and the worst case of one edit over 10,000 chromosome-1 and 10,000 random sequences.
- **Boundaries and masked bases.** Pretraining segments the clean crop before hiding whole segments, so the boundaries of a
  hidden segment, which LCP derives from its bases, carry information about them. `parsing/boundary_information.py`
  builds crops as in pretraining from 1,000 uniform random sequences, where the context carries no information, and
  1,000 chromosome-1 windows, and measures each pretrained model's accuracy on the hidden bases (chance 25%); the fixed
  twin, whose boundaries carry no information, is the control.
- **Frozen embeddings.** `embedding/stability.py` applies one edit at the centre of 10,000 chromosome-1 sequences and
  measures, for the three frozen twins, the changed segments, the change of each per-base embedding by distance from the
  edit (relative to the change between unrelated bases of the same model), and the change of the read embedding.
- **Comparisons.** The paired drop tests compare LCP's drops with those of BPE and fixed segments for each NT18 group
  (histone marks, promoters, splice sites, enhancers) and over all 18 tasks, for the error sets and the substitution-only
  controls.

## The pretraining-split ablation

NT18 tests on chromosomes 20 and 21, which are among the pretraining chromosomes of the released models (3-22). The split
ablation pretrains the four released models again on chromosomes 1-19 and 22, with everything else unchanged
(`configs/pretrain/<model>-no-chr20-21.yaml`; the BPE twin also gets a vocabulary trained on those chromosomes), and
compares them with the released models on the same task-folds.

```bash
python scripts/reproduce.py split-nt18 split-score --gpus 0,1,2,3 --jobs-per-gpu 3   # from the released split models
# or pretrain them yourself first (runs/<model>-no-chr20-21/checkpoints/final.pt, 9-19 h each):
python scripts/reproduce.py split-pretrain --gpus 0,1,2,3
python scripts/reproduce.py split-nt18 split-score --models-from runs --gpus 0,1,2,3 --jobs-per-gpu 3
```

The released split models are `locus-0.5m-no-chr20-21`, `locus-2m-no-chr20-21`, `twin-fixed-2m-no-chr20-21` and
`twin-bpe-2m-no-chr20-21` (`python scripts/download_models.py --split`).

- **Fine-tuning.** Each model is fine-tuned on NT18 (10 folds) with the recipe, micro-batches, folds and seeds of its
  released counterpart, and its selected weights are kept. A fine-tune from your own pretraining starts only from a
  `final.pt` of the full schedule and the model's own configuration. Results: `results/<model>-no-chr20-21/nt18/`.
- **Comparison.** The tables give the paired difference (model without chromosomes 20 and 21 minus the released model's run
  of the same task and fold) of the test MCC and of the lowest validation loss. The validation folds never contain
  chromosomes 20 and 21 (about 18% of their records lie on chromosomes 1 and 2, which only the new models saw), so a test
  difference matched by a validation difference is a difference between the pretraining runs, and a test difference alone
  would point to the test chromosomes.
- **Scoring.** The three 2M models are scored on the tokenizer study's error sets and real reads (`split-score`), and their
  drops are compared with those of the released twins in the same way.

## Result tables

```bash
python scripts/collect_results.py --results results --output results/tables
```

This writes each table of the paper that `results/` has results for to `results/tables.md`, and all numbers to
`results/tables.json`.
- **Cells.** Fine-tuning cells show the mean and standard deviation over the folds: 10 folds for NT18, the tokenizer
  study, GUE splice and the real reads (`--nt18-folds`) and 10 for the Genomic Benchmarks and the held-out check
  (`--other-folds`). A cell with missing folds shows the
  mean and the number of folds instead, and the script warns about missing folds in the NT18, GUE splice and real-read
  tables.
- **Set names.** The table "Tokenizer study: modified test sets" explains the set names (ont1, sub3, pbsim97, ...).
- **Paired drop tests.** Each drop is the mean over task-folds, with the standard deviation over folds and a 95%
  interval from a bootstrap over tasks (4,000 resamples). The folds of a task stay together because they share its test
  set. The held-out-chromosome table adds the pretraining gain (LoCUS-2M minus random init) on both test sets, with the
  same kind of interval for their difference.
- **Single-nucleotide table.** LCP's drops next to those of Caduceus-PS 1.9M, whose tokens are single nucleotides, on the
  same task-folds.
- **Split-ablation tables.** NT18 for the models without chromosomes 20 and 21 next to the released models, and the paired
  differences of test MCC, validation loss and drops, with 95% intervals from resampling tasks (4,000 resamples).

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
  shared vs shared), and by about 0.01 on 18-task means. Small test sets differ more: on promoter_tata (212 test
  sequences, where 4 sequences change the MCC by about 0.04) same-seed runs of LoCUS-2M on fold 0 ranged from 0.77 to
  0.90, and on H3K9me3 runs differed by 0.03 to 0.06. The fine-tune example of the main README (H3K27ac, fold 0) gave
  0.448 to 0.467 against the paper's 0.487. Compare means over folds and tasks, not single folds. `reproduce.py` sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for its jobs, as the paper's runs did, so that three fine-tunes fit
  on one 24 GB GPU; this only changes how memory is allocated.
- **Test sets with errors.** The error sets and controls are generated from fixed seeds, and PBSIM3 runs with a fixed
  seed at its pinned commit. Regenerated from the NT18 download, they are byte-identical to the paper's. PBSIM3 draws its
  random numbers from the C library, so this holds on Linux (glibc). `benchmarks/noisy_reads/test_sets_sha256.tsv` lists
  the SHA-256 of the paper's 1,330 generated sets (NT18 and GUE splice sites, 10 draws each); check yours with
  `cd data/benchmarks && sha256sum -c ../../benchmarks/noisy_reads/test_sets_sha256.tsv`. The real reads are checked
  against the paper's checksums by `scripts/get_real_reads.sh`.
- **The LCP library.** `third_party/lcptools` is [lcptools](https://github.com/BilkentCompGen/lcptools) at commit
  9dcc05f with a small local patch (`third_party/lcptools/local-changes.patch`). The patch adds the core type to `struct
  core`, which the Python bindings in `locus/segmentation.py` need, and fixes how cores are released in
  `lps_deepen1_dct_iters`. `make lcptools` builds the library with the paper's settings (32-bit labels and positions,
  dictionary encoding) and gives a byte-identical library, even when an environment module sets `LD_RUN_PATH`.
- **External code.** `third_party/README.md` lists each external component, model and dataset with its version and any
  change we made.

## Repository layout

```
locus/            model, data pipeline, pretraining and fine-tuning code
configs/          pretraining (configs/pretrain) and fine-tuning (configs/finetune) configurations (see configs/README.md)
scripts/          data download, pretraining, fine-tuning, reproduction and table collection (see scripts/README.md)
benchmarks/       data preparation and evaluation for each benchmark (this folder)
reference/        the paper's result tables, to compare a rerun against
tools/            genome window splitter (C)
third_party/      lcptools, the htslib 1.24 FASTA reader headers and the causal-conv1d Python 3.9 patch (see third_party/README.md)
tests/            unit tests
```
