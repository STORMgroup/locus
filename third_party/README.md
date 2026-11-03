# External code and models, and every change made to them

This file lists every piece of external code, every external model and every external dataset the paper's results use,
the exact version, and each change made to it. Anything not listed as changed is used unmodified.

## Code in this directory

| Component | Version | License | Changes |
|---|---|---|---|
| [lcptools](https://github.com/BilkentCompGen/lcptools) | commit 9dcc05f | BSD 3-Clause (`lcptools/LICENSE`) | `lcptools/local-changes.patch`, see below |
| [htslib](https://github.com/samtools/htslib) headers `kseq.h`, `kstring.h`, `kroundup.h`, `hts_defs.h` | release 1.24 | MIT (`htslib/LICENSE`) | none |
| [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) (installed by `install.sh --baselines`, Python 3.9 only) | 1.7.0 | BSD 3-Clause (`causal-conv1d-1.7.0-LICENSE`, the upstream LICENSE and AUTHORS files) | `causal-conv1d-1.7.0-python3.9.patch`, see below |

**lcptools** (`lcptools/local-changes.patch`, applied to `core.c`, `core.h` and `lps.c`):
- Each core records which branch of the parser produced it, in a new field `type` of `struct core` (values
  `LCP_CORE_LMIN`, `LCP_CORE_LMAX`, `LCP_CORE_RINT`, `LCP_CORE_SSEQ`, and `LCP_CORE_UNKNOWN` for cores created directly or
  read from the legacy binary format). The Python bindings (`locus/segmentation.py`) rely on this struct layout. The field
  does not change which cores are produced.
- `lps_deepen1_dct_iters` (one level deeper with dictionary encoding) frees the cores of the previous level that are no
  longer used, also when it returns early; the original loop meant to remove them did not free them. A failed `realloc`
  keeps the old array. These are memory fixes; the parse is unchanged.
- Missing newlines at the ends of the three files are added.
- Only the library sources are included here; upstream's tests, documentation and data are not. The checkout used for
  the paper also added two assertions on the new field to upstream's tests.

**causal-conv1d 1.7.0** (`causal-conv1d-1.7.0-python3.9.patch`, applied to `causal_conv1d/cpp_functions.py`): the type
hints `torch.Tensor | None` need Python 3.10, so the package does not import on Python 3.9. The patch rewrites them as
`Optional[torch.Tensor]` and adds `from __future__ import annotations`; no code changes. The paper's environment
(Python 3.9) had the same change, made by hand after installation; `install.sh` applies it to the pinned source
(SHA-256 checked) on Python 3.9 only. On Python 3.10 and newer the unmodified package is installed.

## Tools built by `make`

| Tool | Version | Changes |
|---|---|---|
| [PBSIM3](https://github.com/yukiteruono/pbsim3) (`make pbsim3`, simulated nanopore reads) | commit 35ce0de | none to the sources; building regenerates its autotools files (`configure`, `config.h`), in the paper's build as here. The release build reproduces the paper's PBSIM3 test sets of all 18 tasks exactly |

## Python packages

All other packages are installed unmodified at the versions in `requirements.txt` and `requirements-baselines.txt`
(mamba-ssm 2.3.1 is built from its PyPI source on Python 3.9). In the paper's environment, every installed file of the 61
packages was checked against its installation record: causal-conv1d's `cpp_functions.py` (above) is the only modified file.

## Pretrained baseline models

Loaded from their Hugging Face repositories at the revisions pinned in `locus/finetune/baselines.py`; the files used for
the paper are identical to those revisions, including the models' own code (`trust_remote_code`).

| Baseline | Repository | Revision |
|---|---|---|
| Caduceus-PS 1.9M | kuleshov-group/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 | 2de36509b76b4395a97bfa39acafeb6f17b537c5 |
| Caduceus-PS 0.47M | kuleshov-group/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3 | 51d88e326c0fadb0dcc63efdd7345119ce672ce5 |

The model code is not changed, but the baselines are fine-tuned with the LoCUS protocol rather than with their authors'
recipes (`locus/finetune/baselines.py`, `scripts/finetune_baseline.py`, `configs/finetune/`):
- **Input:** the bases only, as the models' character tokens (A, C, G, T, N = 7-11). The models' tokenizers would append
  a separator token, which is not added.
- **Output and head:** the mean of the final hidden states over the bases, then dropout and a linear classifier (the
  LoCUS head), instead of the models' own classification heads. Caduceus-PS is reverse-complement equivariant; its two
  output halves are averaged per base, which with mean pooling and a linear head equals Caduceus's averaging of the two
  halves' logits.
- **Dropout:** 0.1 in every dropout layer of the encoder and before the classifier. Caduceus-PS has no dropout layers, so
  only the dropout before the classifier applies (Caduceus's own fine-tuning uses none).
- **Optimizer:** the LoCUS-2M recipe for both sizes (learning rate 1e-3, 1% warm-up, batch 128, 20 epochs, cosine decay to
  0.1x), AdamW with weight decay 0.1 on every parameter with two or more dimensions. This includes Mamba's `A_log`, which Mamba marks to be excluded from weight
  decay (`_no_weight_decay`); `D`, also marked, has one dimension and is not decayed.
- **Precision:** float32 (LoCUS fine-tunes in bf16): bf16 training at this learning rate failed for Caduceus-PS.
- **Strands, folds and selection:** as for LoCUS (random strand in training, logits of both strands averaged at
  evaluation, the same validation folds and seeds, the epoch with the lowest validation loss).

**Supplementary runs with Caduceus's own settings** (`scripts/reproduce.py baselines-authors`, NT18, 10 folds,
`configs/finetune/caduceus-ps-<size>-authors.yaml`). Caduceus-PS at the sizes of LoCUS-0.5M (0.47M parameters) and LoCUS-2M
(1.9M), with the same input and head, fine-tuned with the choices of Caduceus's own Nucleotide Transformer fine-tuning
(github.com/kuleshov-group/caduceus: `configs/experiment/hg38/nucleotide_transformer.yaml`, `train.py`,
`src/utils/optim_groups.py`) wherever those differ from the LoCUS protocol:
- **Learning rate and batch size:** per task, the values of its paper's Table 6 (learning rate 1e-3, 2e-3 for enhancer types;
  batch 128, or 512 for H3K4me2, H3K4me3, enhancers, enhancer types, promoter_tata and splice_sites_all), matched by name to the
  tasks of the original NT benchmark; the five histone marks of the revised benchmark without a namesake there (H2AFZ,
  H3K27ac, H3K27me3, H3K9me3, H4K20me1) take batch 128, the choice for 8 of its 10 histone tasks (`task_settings` in the
  recipe). Table 6 was chosen for the 1.9M model; the 0.47M model takes the same values.
- **Weight-decay groups:** no decay on biases, embeddings, normalization layers and Mamba's `A_log` and `D`
  (`add_optimizer_hooks`; Hugging Face's loading drops the `_no_weight_decay` marks, so `A_log` and `D` are found by module).
- **Dropout:** none, as in Caduceus's fine-tuning.
- **Precision:** fp16 mixed precision with loss scaling, as in Caduceus's code. The folds whose fp16 training overflowed
  (non-finite steps or validation losses) are run again in bfloat16, Caduceus's config's choice on GPUs that support it
  (`scripts/reproduce.py baselines-authors-bf16`, `configs/finetune/caduceus-ps-<size>-authors-bf16.yaml`), and the table
  uses those runs for them.
- **Strands:** the forward strand in training (`rc_aug: false`) and in validation and test (Caduceus's code reads one
  strand); the model with its head is reverse-complement invariant, so strand-averaged evaluation would give the same
  logits up to rounding.
- **Unchanged:** 20 epochs, 1% warm-up, cosine decay to 0.1x, clipping at 1.0 and weight decay 0.1 (as in its NT config),
  our data, folds and seeds, and the epoch with the lowest validation loss
  (Caduceus's code selects by validation MCC).

## Data

No benchmark sequence or label is changed. The preparation scripts download pinned revisions and record what they do:
- GRCh38.p14 (NCBI RefSeq, MD5 checked): cut into windows that contain only A, C, G and T (`scripts/prepare_genome.sh`).
- NT18 revised and Genomic Benchmarks: sequences and labels as published; preparation stops on any symbol other than
  A, C, G, T or N (NT18), or would replace it by N and count it (Genomic Benchmarks; the count is 0 for every split).
- Frozen eQTL / bulk RNA tasks of the long-range benchmark: the benchmark's elements cut from the genome as the
  benchmark's loader does (4 kb windows); an element whose 4 kb or 16 kb window runs off the chromosome or contains an N (the item set of the paper's runs, fixed when longer
  inputs were also tried)
  is dropped (counts in the prepared `rows.json.gz`).
- GUE splice sites (DNABERT-2's GUE, `GUE/splice/reconstructed`): read from the Hugging Face copy leannmlindsey/GUE at
  commit 240df51a2cd2c0fb1d88efffaf86e8d1e477f932, whose files are identical to those of the GUE archive released with
  DNABERT-2; sequences upper-cased, labels unchanged; GUE's dev split is not used.
- Real nanopore reads: Oxford Nanopore open data (giab_2023.05, HG002, flowcell PAO89685, sup and hac basecalls), streamed
  from the public CRAM files; chromosome 21 of their alignment reference (GRCh38 no-alt analysis set, hard-masked at
  chr21:10,864,560-12,915,808) is fetched by its MD5 from the ENA CRAM reference registry. Only the read bases over the
  NT18 test windows are kept (`benchmarks/noisy_reads/real_reads.py`).
- The modified NT18 and GUE test sets of the tokenizer study (errors, controls, PBSIM3 reads, real reads) are generated by
  `benchmarks/noisy_reads`; only test sets are modified, and the training data is the clean benchmark.
