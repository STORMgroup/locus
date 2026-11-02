## Nucleotide Transformer benchmark, revised (test MCC)

| group | task | locus-0.5m | locus-2m | twin-fixed-2m | twin-bpe-2m | locus-2m-random-init | caduceus-ps-0.47m | caduceus-ps-1.9m |
|---|---|---|---|---|---|---|---|---|
| histone | H3K4me3 | 0.609 ± 0.012 | 0.615 ± 0.014 | 0.607 ± 0.009 | 0.622 ± 0.011 | 0.582 ± 0.008 | 0.607 ± 0.012 | 0.617 ± 0.009 |
| histone | H3K27ac | 0.463 ± 0.009 | 0.469 ± 0.009 | 0.454 ± 0.011 | 0.469 ± 0.008 | 0.328 ± 0.026 | 0.436 ± 0.014 | 0.458 ± 0.014 |
| histone | H3K36me3 | 0.525 ± 0.006 | 0.527 ± 0.012 | 0.536 ± 0.008 | 0.535 ± 0.006 | 0.458 ± 0.011 | 0.518 ± 0.010 | 0.527 ± 0.013 |
| histone | H3K9me3 | 0.393 ± 0.023 | 0.385 ± 0.015 | 0.390 ± 0.018 | 0.381 ± 0.025 | 0.301 ± 0.021 | 0.386 ± 0.025 | 0.395 ± 0.014 |
| histone | H2AFZ | 0.455 ± 0.009 | 0.460 ± 0.013 | 0.437 ± 0.006 | 0.451 ± 0.006 | 0.371 ± 0.007 | 0.439 ± 0.011 | 0.437 ± 0.008 |
| histone | H3K27me3 | 0.542 ± 0.009 | 0.543 ± 0.007 | 0.545 ± 0.008 | 0.552 ± 0.004 | 0.517 ± 0.010 | 0.548 ± 0.005 | 0.555 ± 0.012 |
| histone | H3K4me1 | 0.432 ± 0.009 | 0.445 ± 0.010 | 0.447 ± 0.009 | 0.448 ± 0.011 | 0.354 ± 0.008 | 0.420 ± 0.012 | 0.430 ± 0.013 |
| histone | H3K4me2 | 0.501 ± 0.007 | 0.505 ± 0.011 | 0.504 ± 0.010 | 0.508 ± 0.011 | 0.429 ± 0.017 | 0.479 ± 0.015 | 0.493 ± 0.014 |
| histone | H3K9ac | 0.508 ± 0.017 | 0.512 ± 0.010 | 0.512 ± 0.022 | 0.516 ± 0.015 | 0.425 ± 0.017 | 0.496 ± 0.014 | 0.507 ± 0.010 |
| histone | H4K20me1 | 0.591 ± 0.008 | 0.590 ± 0.012 | 0.584 ± 0.007 | 0.586 ± 0.008 | 0.522 ± 0.009 | 0.580 ± 0.013 | 0.579 ± 0.007 |
| promoter | promoter_all | 0.721 ± 0.004 | 0.716 ± 0.008 | 0.739 ± 0.014 | 0.719 ± 0.010 | 0.696 ± 0.009 | 0.710 ± 0.007 | 0.709 ± 0.009 |
| promoter | promoter_no_tata | 0.746 ± 0.009 | 0.734 ± 0.011 | 0.746 ± 0.007 | 0.734 ± 0.007 | 0.727 ± 0.011 | 0.735 ± 0.007 | 0.728 ± 0.014 |
| promoter | promoter_tata | 0.821 ± 0.049 | 0.843 ± 0.022 | 0.924 ± 0.020 | 0.837 ± 0.019 | 0.777 ± 0.050 | 0.765 ± 0.086 | 0.794 ± 0.029 |
| splice | splice_sites_all | 0.925 ± 0.007 | 0.936 ± 0.006 | 0.963 ± 0.003 | 0.949 ± 0.005 | 0.547 ± 0.038 | 0.902 ± 0.009 | 0.902 ± 0.006 |
| splice | splice_sites_acceptors | 0.917 ± 0.007 | 0.923 ± 0.006 | 0.957 ± 0.003 | 0.939 ± 0.006 | 0.671 ± 0.031 | 0.872 ± 0.010 | 0.869 ± 0.013 |
| splice | splice_sites_donors | 0.930 ± 0.007 | 0.942 ± 0.004 | 0.965 ± 0.003 | 0.958 ± 0.006 | 0.667 ± 0.095 | 0.887 ± 0.013 | 0.889 ± 0.015 |
| enhancer | enhancers | 0.505 ± 0.007 | 0.514 ± 0.005 | 0.497 ± 0.004 | 0.500 ± 0.006 | 0.449 ± 0.005 | 0.495 ± 0.010 | 0.505 ± 0.009 |
| enhancer | enhancers_types | 0.477 ± 0.007 | 0.466 ± 0.009 | 0.455 ± 0.006 | 0.468 ± 0.011 | 0.409 ± 0.008 | 0.455 ± 0.007 | 0.459 ± 0.012 |
| **mean** | histone | 0.502 | 0.505 | 0.502 | 0.507 | 0.429 | 0.491 | 0.500 |
| **mean** | promoter | 0.763 | 0.764 | 0.803 | 0.764 | 0.733 | 0.737 | 0.744 |
| **mean** | splice | 0.924 | 0.934 | 0.962 | 0.949 | 0.628 | 0.887 | 0.887 |
| **mean** | enhancer | 0.491 | 0.490 | 0.476 | 0.484 | 0.429 | 0.475 | 0.482 |
| **mean** | all 18 | 0.614 | 0.618 | 0.626 | 0.621 | 0.513 | 0.596 | 0.603 |

twin-fixed-2m and twin-bpe-2m: LoCUS-2M with fixed 10 bp segments every 5 bp or BPE tokens (tokenizer study), same recipe and seeds. locus-2m-random-init: the LoCUS-2M architecture fine-tuned from random initialization (no pretraining), same recipe and seeds. Its selected epoch is the last one in 30 of 180 runs (20 of 30 on the splice tasks). Baselines (caduceus-ps-0.47m, caduceus-ps-1.9m): fine-tuned with the same protocol (scripts/finetune_baseline.py).

## Nucleotide Transformer benchmark, Caduceus-PS with its own fine-tuning settings (test MCC)

| group | task | locus-0.5m | locus-2m | caduceus-ps-0.47m | caduceus-ps-1.9m | caduceus-ps-0.47m-authors | caduceus-ps-1.9m-authors |
|---|---|---|---|---|---|---|---|
| histone | H3K4me3 | 0.609 ± 0.012 | 0.615 ± 0.014 | 0.607 ± 0.012 | 0.617 ± 0.009 | 0.612 ± 0.011 | 0.622 ± 0.011 |
| histone | H3K27ac | 0.463 ± 0.009 | 0.469 ± 0.009 | 0.436 ± 0.014 | 0.458 ± 0.014 | 0.438 ± 0.022 | 0.459 ± 0.013 |
| histone | H3K36me3 | 0.525 ± 0.006 | 0.527 ± 0.012 | 0.518 ± 0.010 | 0.527 ± 0.013 | 0.525 ± 0.009 | 0.531 ± 0.015 |
| histone | H3K9me3 | 0.393 ± 0.023 | 0.385 ± 0.015 | 0.386 ± 0.025 | 0.395 ± 0.014 | 0.370 ± 0.023 | 0.397 ± 0.020 |
| histone | H2AFZ | 0.455 ± 0.009 | 0.460 ± 0.013 | 0.439 ± 0.011 | 0.437 ± 0.008 | 0.435 ± 0.010 | 0.436 ± 0.009 |
| histone | H3K27me3 | 0.542 ± 0.009 | 0.543 ± 0.007 | 0.548 ± 0.005 | 0.555 ± 0.012 | 0.549 ± 0.013 | 0.550 ± 0.015 |
| histone | H3K4me1 | 0.432 ± 0.009 | 0.445 ± 0.010 | 0.420 ± 0.012 | 0.430 ± 0.013 | 0.421 ± 0.013 | 0.432 ± 0.013 |
| histone | H3K4me2 | 0.501 ± 0.007 | 0.505 ± 0.011 | 0.479 ± 0.015 | 0.493 ± 0.014 | 0.468 ± 0.010 | 0.476 ± 0.006 |
| histone | H3K9ac | 0.508 ± 0.017 | 0.512 ± 0.010 | 0.496 ± 0.014 | 0.507 ± 0.010 | 0.503 ± 0.012 | 0.500 ± 0.014 |
| histone | H4K20me1 | 0.591 ± 0.008 | 0.590 ± 0.012 | 0.580 ± 0.013 | 0.579 ± 0.007 | 0.579 ± 0.010 | 0.587 ± 0.010 |
| promoter | promoter_all | 0.721 ± 0.004 | 0.716 ± 0.008 | 0.710 ± 0.007 | 0.709 ± 0.009 | 0.712 ± 0.013 | 0.714 ± 0.008 |
| promoter | promoter_no_tata | 0.746 ± 0.009 | 0.734 ± 0.011 | 0.735 ± 0.007 | 0.728 ± 0.014 | 0.736 ± 0.013 | 0.730 ± 0.012 |
| promoter | promoter_tata | 0.821 ± 0.049 | 0.843 ± 0.022 | 0.765 ± 0.086 | 0.794 ± 0.029 | 0.730 ± 0.096 | 0.740 ± 0.045 |
| splice | splice_sites_all | 0.925 ± 0.007 | 0.936 ± 0.006 | 0.902 ± 0.009 | 0.902 ± 0.006 | 0.860 ± 0.020 | 0.858 ± 0.024 |
| splice | splice_sites_acceptors | 0.917 ± 0.007 | 0.923 ± 0.006 | 0.872 ± 0.010 | 0.869 ± 0.013 | 0.877 ± 0.008 | 0.873 ± 0.009 |
| splice | splice_sites_donors | 0.930 ± 0.007 | 0.942 ± 0.004 | 0.887 ± 0.013 | 0.889 ± 0.015 | 0.890 ± 0.013 | 0.890 ± 0.010 |
| enhancer | enhancers | 0.505 ± 0.007 | 0.514 ± 0.005 | 0.495 ± 0.010 | 0.505 ± 0.009 | 0.484 ± 0.008 | 0.496 ± 0.010 |
| enhancer | enhancers_types | 0.477 ± 0.007 | 0.466 ± 0.009 | 0.455 ± 0.007 | 0.459 ± 0.012 | 0.450 ± 0.011 | 0.460 ± 0.009 |
| **mean** | histone | 0.502 | 0.505 | 0.491 | 0.500 | 0.490 | 0.499 |
| **mean** | promoter | 0.763 | 0.764 | 0.737 | 0.744 | 0.726 | 0.728 |
| **mean** | splice | 0.924 | 0.934 | 0.887 | 0.887 | 0.876 | 0.874 |
| **mean** | enhancer | 0.491 | 0.490 | 0.475 | 0.482 | 0.467 | 0.478 |
| **mean** | all 18 | 0.614 | 0.618 | 0.596 | 0.603 | 0.591 | 0.597 |

caduceus-ps-<size>-authors: Caduceus-PS fine-tuned with Caduceus's own settings for the NT tasks (configs/finetune/caduceus-ps-<size>-authors.yaml: per task the learning rate and batch size of its paper's Table 6, its weight-decay groups, no dropout, the forward strand in training and evaluation); caduceus-ps-1.9m: the protocol shared with LoCUS. The folds, seeds and epoch selection (lowest validation loss) are the same for every column. 64 folds whose fp16 training overflowed (non-finite steps or validation losses) are the bfloat16 reruns (caduceus-ps-<size>-authors-bf16.yaml).

## NT18 on chromosomes outside pretraining (test MCC)

| group | task | LoCUS-2M: held-out chromosomes | LoCUS-2M: official test | LoCUS-2M: held-out - official | random init: held-out chromosomes | random init: official test | random init: held-out - official |
|---|---|---|---|---|---|---|---|
| histone | H3K4me3 | 0.656 ± 0.009 | 0.607 ± 0.009 | 0.049 ± 0.015 | 0.622 ± 0.004 | 0.579 ± 0.014 | 0.043 ± 0.017 |
| histone | H3K27ac | 0.465 ± 0.008 | 0.449 ± 0.014 | 0.016 ± 0.012 | 0.354 ± 0.021 | 0.328 ± 0.015 | 0.026 ± 0.020 |
| histone | H3K36me3 | 0.513 ± 0.011 | 0.523 ± 0.006 | -0.010 ± 0.012 | 0.439 ± 0.008 | 0.450 ± 0.007 | -0.011 ± 0.011 |
| histone | H3K9me3 | 0.315 ± 0.009 | 0.381 ± 0.013 | -0.065 ± 0.015 | 0.238 ± 0.011 | 0.280 ± 0.021 | -0.042 ± 0.016 |
| histone | H2AFZ | 0.430 ± 0.007 | 0.445 ± 0.015 | -0.015 ± 0.015 | 0.365 ± 0.008 | 0.370 ± 0.008 | -0.005 ± 0.011 |
| histone | H3K27me3 | 0.558 ± 0.006 | 0.549 ± 0.007 | 0.009 ± 0.011 | 0.531 ± 0.008 | 0.510 ± 0.009 | 0.021 ± 0.007 |
| histone | H3K4me1 | 0.438 ± 0.006 | 0.436 ± 0.009 | 0.003 ± 0.008 | 0.346 ± 0.009 | 0.350 ± 0.008 | -0.005 ± 0.007 |
| histone | H3K4me2 | 0.515 ± 0.008 | 0.486 ± 0.011 | 0.029 ± 0.009 | 0.427 ± 0.013 | 0.418 ± 0.015 | 0.009 ± 0.007 |
| histone | H3K9ac | 0.485 ± 0.008 | 0.498 ± 0.013 | -0.013 ± 0.008 | 0.407 ± 0.013 | 0.406 ± 0.025 | 0.002 ± 0.028 |
| histone | H4K20me1 | 0.610 ± 0.007 | 0.576 ± 0.008 | 0.035 ± 0.011 | 0.554 ± 0.008 | 0.514 ± 0.015 | 0.040 ± 0.015 |
| promoter | promoter_all | 0.714 ± 0.009 | 0.715 ± 0.012 | -0.001 ± 0.008 | 0.704 ± 0.007 | 0.695 ± 0.009 | 0.009 ± 0.012 |
| promoter | promoter_no_tata | 0.730 ± 0.007 | 0.731 ± 0.011 | -0.001 ± 0.015 | 0.716 ± 0.005 | 0.719 ± 0.006 | -0.004 ± 0.008 |
| promoter | promoter_tata | 0.781 ± 0.028 | 0.795 ± 0.048 | -0.015 ± 0.026 | 0.649 ± 0.032 | 0.741 ± 0.035 | -0.092 ± 0.022 |
| splice | splice_sites_all | 0.939 ± 0.003 | 0.931 ± 0.005 | 0.009 ± 0.003 | 0.542 ± 0.052 | 0.554 ± 0.042 | -0.012 ± 0.014 |
| splice | splice_sites_acceptors | 0.928 ± 0.004 | 0.921 ± 0.005 | 0.007 ± 0.002 | 0.690 ± 0.035 | 0.691 ± 0.029 | -0.001 ± 0.008 |
| splice | splice_sites_donors | 0.947 ± 0.003 | 0.938 ± 0.004 | 0.009 ± 0.003 | 0.613 ± 0.019 | 0.583 ± 0.021 | 0.030 ± 0.008 |
| enhancer | enhancers | 0.491 ± 0.006 | 0.505 ± 0.005 | -0.014 ± 0.009 | 0.430 ± 0.006 | 0.444 ± 0.007 | -0.014 ± 0.009 |
| enhancer | enhancers_types | 0.454 ± 0.011 | 0.468 ± 0.008 | -0.014 ± 0.010 | 0.398 ± 0.004 | 0.405 ± 0.005 | -0.007 ± 0.008 |
| **mean** | histone | 0.499 | 0.495 | +0.004 | 0.428 | 0.421 | +0.008 |
| **mean** | promoter | 0.741 | 0.747 | -0.006 | 0.690 | 0.719 | -0.029 |
| **mean** | splice | 0.938 | 0.930 | +0.008 | 0.615 | 0.609 | +0.006 |
| **mean** | enhancer | 0.472 | 0.486 | -0.014 | 0.414 | 0.424 | -0.010 |
| **mean** | all 18 | 0.609 | 0.608 | +0.001 | 0.501 | 0.502 | -0.001 |

Fine-tuned on the official training set without its examples on chromosomes outside the pretraining chromosomes (3-22), tested on those examples (seen neither in pretraining nor in fine-tuning); the official test set (chromosomes 20 and 21, part of the pretraining chromosomes) is scored at the same selected epoch. Random init: the same architecture without pretrained weights, so neither test set was seen before fine-tuning; its held-out - official gap measures how much harder the held-out chromosomes are. 10 folds.

## NT18 on chromosomes outside pretraining: pretraining gain (LoCUS-2M - random init, test MCC)

| tasks | number | gain, held-out chromosomes | gain, official test | official - held-out | 95% CI | tasks with official - held-out > 0 |
|---|---|---|---|---|---|---|
| all 18 | 18 | +0.108 | +0.106 | -0.002 | [-0.013, +0.007] | 10/18 |
| histone | 10 | +0.070 | +0.074 | +0.004 | [-0.003, +0.012] | 6/10 |
| promoter | 3 | +0.052 | +0.028 | -0.023 | [-0.077, +0.010] | 1/3 |
| splice | 3 | +0.323 | +0.321 | -0.003 | [-0.021, +0.021] | 1/3 |
| enhancer | 2 | +0.058 | +0.062 | +0.004 | [+0.000, +0.007] | 2/2 |

Gain per task: mean over the folds of LoCUS-2M minus that of random init, on the same test set; > 0 in the official - held-out column would mean that seeing the official test chromosomes in pretraining helps. Interval: bootstrap over tasks (4,000 resamples).

## Genomic Benchmarks (test accuracy)

| task | locus-0.5m | locus-2m |
|---|---|---|
| demo_coding_vs_intergenomic_seqs | 0.924 ± 0.002 | 0.929 ± 0.002 |
| demo_human_or_worm | 0.969 ± 0.001 | 0.969 ± 0.001 |
| dummy_mouse_enhancers_ensembl | 0.790 ± 0.023 | 0.805 ± 0.022 |
| human_enhancers_cohn | 0.741 ± 0.003 | 0.737 ± 0.003 |
| human_enhancers_ensembl | 0.877 ± 0.002 | 0.896 ± 0.002 |
| human_ensembl_regulatory | 0.938 ± 0.001 | 0.941 ± 0.001 |
| human_nontata_promoters | 0.920 ± 0.004 | 0.948 ± 0.005 |
| human_ocr_ensembl | 0.807 ± 0.003 | 0.834 ± 0.002 |
| drosophila_enhancers_stark | 0.731 ± 0.009 | 0.743 ± 0.030 |
| **mean, 8 tasks (no drosophila)** | 0.871 | 0.882 |
| **mean, all 9** | 0.855 | 0.867 |

Mean ± SD over 10 folds.

## Long-range benchmark, frozen embeddings, 4 kb input: causal eQTL and bulk RNA expression

| model | eQTL AUROC | eQTL accuracy | bulk RNA Spearman (genes) | bulk RNA R2 |
|---|---|---|---|---|
| locus-0.5m | 0.708 | 0.651 | 0.726 | 0.437 |
| locus-2m | 0.703 | 0.654 | 0.734 | 0.447 |
| no sequence: tissue + TSS distance | 0.645 | 0.617 | - | - |

Means over the 5 chromosome folds (benchmarks/long_range/frozen.py).

## Dataset creation: segmentation time of the NT18 fine-tuning data (one CPU core)

| segmenter | 18 tasks (seconds) | mean per task (seconds) | largest task (seconds) | million bases per second | ms per 600 bp sequence (one strand) |
|---|---|---|---|---|---|
| lcp | 177.6 | 9.9 | 14.6 | 4.56 | 0.132 |
| bpe | 277.8 | 15.4 | 23.0 | 2.92 | 0.198 |
| fixed | 72.3 | 4.0 | 6.0 | 11.21 | 0.053 |
| single-base | 8.9 | 0.5 | 0.6 | 90.65 | 0.008 |

Every training and test record of each task, both strands (benchmarks/efficiency/segmentation_time.py): lcp = LCP level 2 (LoCUS), bpe = BPE with 4,096 tokens (twin-bpe-2m), fixed = 10 bp every 5 bp (twin-fixed-2m), single-base = base encoding (Caduceus). x86_64.

## Fine-tuning throughput (one GPU, same task, batch and precision for every model)

| model | parameters (fine-tuned model) | sequences per second | sequences per second, segmenting each sequence first | ms per step | minutes per epoch | peak GPU memory (GB) |
|---|---|---|---|---|---|---|
| LoCUS-0.5M | 434,435 | 1594.0 | 1317.5 | 40.2 | 0.31 | 1.04 |
| LoCUS-2M | 1,655,171 | 1128.5 | 982.6 | 56.7 | 0.44 | 2.03 |
| Caduceus-PS 1.9M | 1,935,363 | 232.4 | 232.0 | 275.4 | 2.15 | 3.35 |
| Caduceus-PS 0.47M | 471,059 | 484.8 | 483.0 | 132.0 | 1.03 | 1.74 |

NVIDIA RTX A5000, task splice_sites_all_revised, batch 64, bf16 autocast, 10 warm-up and 60 timed steps. Inputs are segmented before timing; the second column adds the segmentation time of one sequence (one strand, one CPU core, segmentation table) to each sequence, the worst case of segmenting on the fly without overlap; fine-tuning segments each record once and reuses it in every epoch.

## Full fine-tuning wall clock: fold 0 of every NT18 task, one run per GPU

| model | tasks timed (of 18) | tasks timed for every model | fine-tuning, mean per task (minutes) | segmentation (dataset creation), mean per task (minutes) | total, mean per task (minutes) | total on those tasks (hours) | peak GPU memory allocated (GB) | GPU |
|---|---|---|---|---|---|---|---|---|
| locus-0.5m | 18 | 18 | 13.04 | 0.16 | 13.20 | 3.96 | 5.05 | NVIDIA RTX A5000 |
| locus-2m | 18 | 18 | 17.22 | 0.16 | 17.39 | 5.22 | 5.01 | NVIDIA RTX A5000 |
| caduceus-ps-0.47m | 18 | 18 | 28.49 | 0.00 | 28.49 | 8.55 | 2.43 | NVIDIA RTX A5000 |
| caduceus-ps-1.9m | 18 | 18 | 71.30 | 0.00 | 71.30 | 21.39 | 5.12 | NVIDIA RTX A5000 |
| caduceus-ps-0.47m-bf16 | 18 | 18 | 20.72 | 0.00 | 20.72 | 6.22 | 1.46 | NVIDIA RTX A5000 |
| caduceus-ps-1.9m-bf16 | 18 | 18 | 40.27 | 0.00 | 40.27 | 12.08 | 2.76 | NVIDIA RTX A5000 |

Fine-tuning: training and every per-epoch evaluation (resources.wall_seconds of the fold result, without its segmentation time). Segmentation: the time to segment the task's training and test records on both strands, once per run (resources.segmentation_seconds, or, for runs that segmented before the clock started, the measurement of the segmentation table on the same data; "included": inside the fine-tuning time, not measured on its own); the single-base baselines segment nothing (their bases are encoded per batch, inside the fine-tuning time). LoCUS runs in bf16 and Caduceus-PS in float32, as in the accuracy runs; caduceus-ps-<size>-bf16: the same Caduceus-PS recipe in bf16, timed only, so the models can also be compared at the same precision.

## Tokenizer study: modified test sets

| set | definition |
|---|---|
| ont1, ont3 | nanopore-like errors (benchmarks/noisy_reads/noisy_sets.py): every base starts an error event with probability 1% or 3%; an event is a substitution (40%), a deletion (35%) or an insertion before the base (25%); deletions and insertions are 1, 2 or 3 bases long (probability 0.75, 0.1875, 0.0625), and an inserted stretch repeats the current base with probability 0.5, otherwise random bases |
| sub1, sub3 | control sets (benchmarks/noisy_reads/make_controls.py): an independent draw of the nanopore-like error schedule at 1% or 3%, every event a single-base substitution (no shift of the segment grid) |
| ontw1, ontw3 | the full nanopore-like model on the same schedule as sub1 / sub3, without any edit inside a protected window; reported for the splice tasks, whose window keeps every edit at least 100 bp from the junction ([200, 400) of the 600 bp NT18 records, [100, 300) of the 400 bp GUE records) |
| pbsim97 | PBSIM3 reads (ERRHMM-ONT-HQ error model, requested mean read accuracy 97%, realized 95.9% by PBSIM3's own report, each test sequence read once) |
| <set>-d1 .. -d9 | further independent error draws of a set; the score of a fold on a set is the mean over its 10 draws |
| realref, realsup, realhac | real HG002 nanopore reads of the test windows (benchmarks/noisy_reads/real_reads.py): the reference sequences of the windows with 10 reads in both basecalls, and those reads in the sup and hac basecalls (10 draws each) |

Only the test set is modified; every model is fine-tuned on the clean training data. Drop = score on the set - score on the clean test set, on the same fold.

## Tokenizer study: test MCC, clean, 1% / 3% errors and PBSIM3 reads

| group | task | LCP clean | LCP ont1 | LCP ont3 | LCP pbsim97 | BPE clean | BPE ont1 | BPE ont3 | BPE pbsim97 | fixed clean | fixed ont1 | fixed ont3 | fixed pbsim97 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| histone | H3K4me3 | 0.615 ± 0.014 | 0.606 ± 0.011 | 0.582 ± 0.013 | 0.563 ± 0.010 | 0.622 ± 0.011 | 0.614 ± 0.010 | 0.586 ± 0.015 | 0.573 ± 0.013 | 0.607 ± 0.009 | 0.602 ± 0.007 | 0.582 ± 0.014 | 0.562 ± 0.012 |
| histone | H3K27ac | 0.469 ± 0.009 | 0.462 ± 0.008 | 0.444 ± 0.006 | 0.448 ± 0.006 | 0.469 ± 0.008 | 0.465 ± 0.007 | 0.448 ± 0.005 | 0.447 ± 0.007 | 0.454 ± 0.011 | 0.446 ± 0.009 | 0.437 ± 0.005 | 0.432 ± 0.007 |
| histone | H3K36me3 | 0.527 ± 0.012 | 0.523 ± 0.012 | 0.514 ± 0.012 | 0.511 ± 0.011 | 0.535 ± 0.006 | 0.531 ± 0.006 | 0.521 ± 0.006 | 0.520 ± 0.005 | 0.536 ± 0.008 | 0.529 ± 0.005 | 0.523 ± 0.007 | 0.521 ± 0.006 |
| histone | H3K9me3 | 0.385 ± 0.015 | 0.382 ± 0.014 | 0.372 ± 0.011 | 0.362 ± 0.015 | 0.381 ± 0.025 | 0.380 ± 0.025 | 0.372 ± 0.023 | 0.366 ± 0.024 | 0.390 ± 0.018 | 0.382 ± 0.015 | 0.367 ± 0.011 | 0.367 ± 0.014 |
| histone | H2AFZ | 0.460 ± 0.013 | 0.455 ± 0.009 | 0.437 ± 0.009 | 0.433 ± 0.009 | 0.451 ± 0.006 | 0.445 ± 0.005 | 0.434 ± 0.007 | 0.427 ± 0.006 | 0.437 ± 0.006 | 0.431 ± 0.004 | 0.421 ± 0.005 | 0.417 ± 0.006 |
| histone | H3K27me3 | 0.543 ± 0.007 | 0.534 ± 0.005 | 0.521 ± 0.007 | 0.515 ± 0.007 | 0.552 ± 0.004 | 0.545 ± 0.004 | 0.532 ± 0.004 | 0.525 ± 0.005 | 0.545 ± 0.008 | 0.538 ± 0.007 | 0.526 ± 0.008 | 0.522 ± 0.007 |
| histone | H3K4me1 | 0.445 ± 0.010 | 0.440 ± 0.008 | 0.430 ± 0.005 | 0.428 ± 0.006 | 0.448 ± 0.011 | 0.442 ± 0.007 | 0.433 ± 0.007 | 0.430 ± 0.007 | 0.447 ± 0.009 | 0.442 ± 0.008 | 0.434 ± 0.008 | 0.428 ± 0.007 |
| histone | H3K4me2 | 0.505 ± 0.011 | 0.497 ± 0.007 | 0.478 ± 0.008 | 0.473 ± 0.008 | 0.508 ± 0.011 | 0.501 ± 0.010 | 0.484 ± 0.010 | 0.479 ± 0.011 | 0.504 ± 0.010 | 0.491 ± 0.007 | 0.475 ± 0.009 | 0.471 ± 0.009 |
| histone | H3K9ac | 0.512 ± 0.010 | 0.507 ± 0.011 | 0.498 ± 0.008 | 0.493 ± 0.008 | 0.516 ± 0.015 | 0.512 ± 0.011 | 0.501 ± 0.012 | 0.492 ± 0.012 | 0.512 ± 0.022 | 0.511 ± 0.016 | 0.506 ± 0.015 | 0.498 ± 0.016 |
| histone | H4K20me1 | 0.590 ± 0.012 | 0.585 ± 0.010 | 0.578 ± 0.009 | 0.577 ± 0.010 | 0.586 ± 0.008 | 0.587 ± 0.007 | 0.578 ± 0.005 | 0.578 ± 0.005 | 0.584 ± 0.007 | 0.581 ± 0.005 | 0.576 ± 0.006 | 0.573 ± 0.005 |
| promoter | promoter_all | 0.716 ± 0.008 | 0.714 ± 0.006 | 0.709 ± 0.006 | 0.707 ± 0.005 | 0.719 ± 0.010 | 0.716 ± 0.009 | 0.710 ± 0.010 | 0.709 ± 0.009 | 0.739 ± 0.014 | 0.729 ± 0.007 | 0.721 ± 0.008 | 0.723 ± 0.008 |
| promoter | promoter_no_tata | 0.734 ± 0.011 | 0.734 ± 0.009 | 0.733 ± 0.006 | 0.728 ± 0.006 | 0.734 ± 0.007 | 0.735 ± 0.005 | 0.732 ± 0.006 | 0.727 ± 0.007 | 0.746 ± 0.007 | 0.741 ± 0.004 | 0.730 ± 0.004 | 0.732 ± 0.005 |
| promoter | promoter_tata | 0.843 ± 0.022 | 0.825 ± 0.018 | 0.821 ± 0.020 | 0.807 ± 0.019 | 0.837 ± 0.019 | 0.822 ± 0.017 | 0.815 ± 0.014 | 0.802 ± 0.016 | 0.924 ± 0.020 | 0.899 ± 0.017 | 0.880 ± 0.016 | 0.877 ± 0.013 |
| splice | splice_sites_all | 0.936 ± 0.006 | 0.914 ± 0.003 | 0.866 ± 0.005 | 0.843 ± 0.005 | 0.949 ± 0.005 | 0.909 ± 0.014 | 0.819 ± 0.036 | 0.802 ± 0.024 | 0.963 ± 0.003 | 0.818 ± 0.024 | 0.678 ± 0.041 | 0.716 ± 0.026 |
| splice | splice_sites_acceptors | 0.923 ± 0.006 | 0.903 ± 0.003 | 0.861 ± 0.006 | 0.834 ± 0.007 | 0.939 ± 0.006 | 0.898 ± 0.004 | 0.801 ± 0.013 | 0.777 ± 0.012 | 0.957 ± 0.003 | 0.669 ± 0.053 | 0.522 ± 0.035 | 0.612 ± 0.029 |
| splice | splice_sites_donors | 0.942 ± 0.004 | 0.908 ± 0.003 | 0.846 ± 0.008 | 0.819 ± 0.009 | 0.958 ± 0.006 | 0.872 ± 0.031 | 0.758 ± 0.039 | 0.754 ± 0.027 | 0.965 ± 0.003 | 0.742 ± 0.050 | 0.587 ± 0.048 | 0.651 ± 0.031 |
| enhancer | enhancers | 0.514 ± 0.005 | 0.509 ± 0.005 | 0.500 ± 0.004 | 0.500 ± 0.005 | 0.500 ± 0.006 | 0.500 ± 0.005 | 0.498 ± 0.005 | 0.490 ± 0.005 | 0.497 ± 0.004 | 0.496 ± 0.005 | 0.492 ± 0.004 | 0.488 ± 0.005 |
| enhancer | enhancers_types | 0.466 ± 0.009 | 0.465 ± 0.008 | 0.457 ± 0.007 | 0.454 ± 0.007 | 0.468 ± 0.011 | 0.465 ± 0.009 | 0.459 ± 0.009 | 0.453 ± 0.009 | 0.455 ± 0.006 | 0.455 ± 0.007 | 0.452 ± 0.006 | 0.445 ± 0.005 |
| **mean** | histone | 0.505 | 0.499 | 0.485 | 0.481 | 0.507 | 0.502 | 0.489 | 0.484 | 0.502 | 0.495 | 0.485 | 0.479 |
| **mean** | promoter | 0.764 | 0.758 | 0.755 | 0.748 | 0.764 | 0.758 | 0.752 | 0.746 | 0.803 | 0.789 | 0.777 | 0.777 |
| **mean** | splice | 0.934 | 0.908 | 0.858 | 0.832 | 0.949 | 0.893 | 0.793 | 0.778 | 0.962 | 0.743 | 0.596 | 0.660 |
| **mean** | enhancer | 0.490 | 0.487 | 0.479 | 0.477 | 0.484 | 0.482 | 0.478 | 0.471 | 0.476 | 0.476 | 0.472 | 0.466 |
| **mean** | all 18 | 0.618 | 0.609 | 0.591 | 0.583 | 0.621 | 0.608 | 0.582 | 0.575 | 0.626 | 0.583 | 0.550 | 0.558 |

Noisy sets: each score is the mean over 10 independent error draws of the set per fold.

## Tokenizer study: test MCC, controls at 3%: errors anywhere / substitutions only

| group | task | LCP clean | LCP ont3 | LCP sub3 | BPE clean | BPE ont3 | BPE sub3 | fixed clean | fixed ont3 | fixed sub3 |
|---|---|---|---|---|---|---|---|---|---|---|
| histone | H3K4me3 | 0.615 ± 0.014 | 0.582 ± 0.013 | 0.561 ± 0.016 | 0.622 ± 0.011 | 0.586 ± 0.015 | 0.566 ± 0.015 | 0.607 ± 0.009 | 0.582 ± 0.014 | 0.564 ± 0.017 |
| histone | H3K27ac | 0.469 ± 0.009 | 0.444 ± 0.006 | 0.444 ± 0.007 | 0.469 ± 0.008 | 0.448 ± 0.005 | 0.446 ± 0.007 | 0.454 ± 0.011 | 0.437 ± 0.005 | 0.434 ± 0.006 |
| histone | H3K36me3 | 0.527 ± 0.012 | 0.514 ± 0.012 | 0.505 ± 0.012 | 0.535 ± 0.006 | 0.521 ± 0.006 | 0.514 ± 0.008 | 0.536 ± 0.008 | 0.523 ± 0.007 | 0.512 ± 0.008 |
| histone | H3K9me3 | 0.385 ± 0.015 | 0.372 ± 0.011 | 0.365 ± 0.012 | 0.381 ± 0.025 | 0.372 ± 0.023 | 0.362 ± 0.021 | 0.390 ± 0.018 | 0.367 ± 0.011 | 0.363 ± 0.013 |
| histone | H2AFZ | 0.460 ± 0.013 | 0.437 ± 0.009 | 0.422 ± 0.010 | 0.451 ± 0.006 | 0.434 ± 0.007 | 0.422 ± 0.006 | 0.437 ± 0.006 | 0.421 ± 0.005 | 0.412 ± 0.005 |
| histone | H3K27me3 | 0.543 ± 0.007 | 0.521 ± 0.007 | 0.504 ± 0.009 | 0.552 ± 0.004 | 0.532 ± 0.004 | 0.517 ± 0.007 | 0.545 ± 0.008 | 0.526 ± 0.008 | 0.512 ± 0.008 |
| histone | H3K4me1 | 0.445 ± 0.010 | 0.430 ± 0.005 | 0.425 ± 0.007 | 0.448 ± 0.011 | 0.433 ± 0.007 | 0.425 ± 0.007 | 0.447 ± 0.009 | 0.434 ± 0.008 | 0.428 ± 0.007 |
| histone | H3K4me2 | 0.505 ± 0.011 | 0.478 ± 0.008 | 0.465 ± 0.007 | 0.508 ± 0.011 | 0.484 ± 0.010 | 0.471 ± 0.014 | 0.504 ± 0.010 | 0.475 ± 0.009 | 0.467 ± 0.011 |
| histone | H3K9ac | 0.512 ± 0.010 | 0.498 ± 0.008 | 0.492 ± 0.007 | 0.516 ± 0.015 | 0.501 ± 0.012 | 0.495 ± 0.011 | 0.512 ± 0.022 | 0.506 ± 0.015 | 0.500 ± 0.015 |
| histone | H4K20me1 | 0.590 ± 0.012 | 0.578 ± 0.009 | 0.570 ± 0.010 | 0.586 ± 0.008 | 0.578 ± 0.005 | 0.572 ± 0.004 | 0.584 ± 0.007 | 0.576 ± 0.006 | 0.567 ± 0.007 |
| promoter | promoter_all | 0.716 ± 0.008 | 0.709 ± 0.006 | 0.705 ± 0.005 | 0.719 ± 0.010 | 0.710 ± 0.010 | 0.706 ± 0.007 | 0.739 ± 0.014 | 0.721 ± 0.008 | 0.723 ± 0.012 |
| promoter | promoter_no_tata | 0.734 ± 0.011 | 0.733 ± 0.006 | 0.731 ± 0.005 | 0.734 ± 0.007 | 0.732 ± 0.006 | 0.731 ± 0.006 | 0.746 ± 0.007 | 0.730 ± 0.004 | 0.736 ± 0.005 |
| promoter | promoter_tata | 0.843 ± 0.022 | 0.821 ± 0.020 | 0.795 ± 0.020 | 0.837 ± 0.019 | 0.815 ± 0.014 | 0.796 ± 0.017 | 0.924 ± 0.020 | 0.880 ± 0.016 | 0.874 ± 0.013 |
| splice | splice_sites_all | 0.936 ± 0.006 | 0.866 ± 0.005 | 0.881 ± 0.004 | 0.949 ± 0.005 | 0.819 ± 0.036 | 0.898 ± 0.004 | 0.963 ± 0.003 | 0.678 ± 0.041 | 0.912 ± 0.002 |
| splice | splice_sites_acceptors | 0.923 ± 0.006 | 0.861 ± 0.006 | 0.874 ± 0.005 | 0.939 ± 0.006 | 0.801 ± 0.013 | 0.896 ± 0.003 | 0.957 ± 0.003 | 0.522 ± 0.035 | 0.901 ± 0.003 |
| splice | splice_sites_donors | 0.942 ± 0.004 | 0.846 ± 0.008 | 0.873 ± 0.005 | 0.958 ± 0.006 | 0.758 ± 0.039 | 0.892 ± 0.006 | 0.965 ± 0.003 | 0.587 ± 0.048 | 0.909 ± 0.003 |
| enhancer | enhancers | 0.514 ± 0.005 | 0.500 ± 0.004 | 0.497 ± 0.003 | 0.500 ± 0.006 | 0.498 ± 0.005 | 0.496 ± 0.005 | 0.497 ± 0.004 | 0.492 ± 0.004 | 0.490 ± 0.003 |
| enhancer | enhancers_types | 0.466 ± 0.009 | 0.457 ± 0.007 | 0.455 ± 0.007 | 0.468 ± 0.011 | 0.459 ± 0.009 | 0.457 ± 0.007 | 0.455 ± 0.006 | 0.452 ± 0.006 | 0.449 ± 0.006 |
| **mean** | histone | 0.505 | 0.485 | 0.475 | 0.507 | 0.489 | 0.479 | 0.502 | 0.485 | 0.476 |
| **mean** | promoter | 0.764 | 0.755 | 0.744 | 0.764 | 0.752 | 0.744 | 0.803 | 0.777 | 0.778 |
| **mean** | splice | 0.934 | 0.858 | 0.876 | 0.949 | 0.793 | 0.895 | 0.962 | 0.596 | 0.907 |
| **mean** | enhancer | 0.490 | 0.479 | 0.476 | 0.484 | 0.478 | 0.477 | 0.476 | 0.472 | 0.470 |
| **mean** | all 18 | 0.618 | 0.591 | 0.587 | 0.621 | 0.582 | 0.592 | 0.626 | 0.550 | 0.597 |

Noisy sets: each score is the mean over 10 independent error draws of the set per fold.

## Tokenizer study: test MCC, controls at 1%: errors anywhere / substitutions only

| group | task | LCP clean | LCP ont1 | LCP sub1 | BPE clean | BPE ont1 | BPE sub1 | fixed clean | fixed ont1 | fixed sub1 |
|---|---|---|---|---|---|---|---|---|---|---|
| histone | H3K4me3 | 0.615 ± 0.014 | 0.606 ± 0.011 | 0.601 ± 0.012 | 0.622 ± 0.011 | 0.614 ± 0.010 | 0.609 ± 0.012 | 0.607 ± 0.009 | 0.602 ± 0.007 | 0.597 ± 0.009 |
| histone | H3K27ac | 0.469 ± 0.009 | 0.462 ± 0.008 | 0.461 ± 0.008 | 0.469 ± 0.008 | 0.465 ± 0.007 | 0.464 ± 0.006 | 0.454 ± 0.011 | 0.446 ± 0.009 | 0.451 ± 0.009 |
| histone | H3K36me3 | 0.527 ± 0.012 | 0.523 ± 0.012 | 0.522 ± 0.012 | 0.535 ± 0.006 | 0.531 ± 0.006 | 0.530 ± 0.005 | 0.536 ± 0.008 | 0.529 ± 0.005 | 0.528 ± 0.005 |
| histone | H3K9me3 | 0.385 ± 0.015 | 0.382 ± 0.014 | 0.378 ± 0.015 | 0.381 ± 0.025 | 0.380 ± 0.025 | 0.378 ± 0.024 | 0.390 ± 0.018 | 0.382 ± 0.015 | 0.379 ± 0.016 |
| histone | H2AFZ | 0.460 ± 0.013 | 0.455 ± 0.009 | 0.448 ± 0.010 | 0.451 ± 0.006 | 0.445 ± 0.005 | 0.442 ± 0.006 | 0.437 ± 0.006 | 0.431 ± 0.004 | 0.429 ± 0.004 |
| histone | H3K27me3 | 0.543 ± 0.007 | 0.534 ± 0.005 | 0.530 ± 0.006 | 0.552 ± 0.004 | 0.545 ± 0.004 | 0.542 ± 0.004 | 0.545 ± 0.008 | 0.538 ± 0.007 | 0.535 ± 0.008 |
| histone | H3K4me1 | 0.445 ± 0.010 | 0.440 ± 0.008 | 0.437 ± 0.007 | 0.448 ± 0.011 | 0.442 ± 0.007 | 0.438 ± 0.008 | 0.447 ± 0.009 | 0.442 ± 0.008 | 0.439 ± 0.008 |
| histone | H3K4me2 | 0.505 ± 0.011 | 0.497 ± 0.007 | 0.495 ± 0.007 | 0.508 ± 0.011 | 0.501 ± 0.010 | 0.500 ± 0.011 | 0.504 ± 0.010 | 0.491 ± 0.007 | 0.492 ± 0.008 |
| histone | H3K9ac | 0.512 ± 0.010 | 0.507 ± 0.011 | 0.506 ± 0.009 | 0.516 ± 0.015 | 0.512 ± 0.011 | 0.508 ± 0.012 | 0.512 ± 0.022 | 0.511 ± 0.016 | 0.512 ± 0.015 |
| histone | H4K20me1 | 0.590 ± 0.012 | 0.585 ± 0.010 | 0.583 ± 0.010 | 0.586 ± 0.008 | 0.587 ± 0.007 | 0.584 ± 0.007 | 0.584 ± 0.007 | 0.581 ± 0.005 | 0.578 ± 0.005 |
| promoter | promoter_all | 0.716 ± 0.008 | 0.714 ± 0.006 | 0.713 ± 0.007 | 0.719 ± 0.010 | 0.716 ± 0.009 | 0.714 ± 0.010 | 0.739 ± 0.014 | 0.729 ± 0.007 | 0.734 ± 0.011 |
| promoter | promoter_no_tata | 0.734 ± 0.011 | 0.734 ± 0.009 | 0.736 ± 0.008 | 0.734 ± 0.007 | 0.735 ± 0.005 | 0.735 ± 0.006 | 0.746 ± 0.007 | 0.741 ± 0.004 | 0.745 ± 0.005 |
| promoter | promoter_tata | 0.843 ± 0.022 | 0.825 ± 0.018 | 0.829 ± 0.021 | 0.837 ± 0.019 | 0.822 ± 0.017 | 0.826 ± 0.014 | 0.924 ± 0.020 | 0.899 ± 0.017 | 0.905 ± 0.016 |
| splice | splice_sites_all | 0.936 ± 0.006 | 0.914 ± 0.003 | 0.919 ± 0.004 | 0.949 ± 0.005 | 0.909 ± 0.014 | 0.934 ± 0.003 | 0.963 ± 0.003 | 0.818 ± 0.024 | 0.947 ± 0.002 |
| splice | splice_sites_acceptors | 0.923 ± 0.006 | 0.903 ± 0.003 | 0.907 ± 0.005 | 0.939 ± 0.006 | 0.898 ± 0.004 | 0.926 ± 0.005 | 0.957 ± 0.003 | 0.669 ± 0.053 | 0.939 ± 0.003 |
| splice | splice_sites_donors | 0.942 ± 0.004 | 0.908 ± 0.003 | 0.918 ± 0.004 | 0.958 ± 0.006 | 0.872 ± 0.031 | 0.936 ± 0.006 | 0.965 ± 0.003 | 0.742 ± 0.050 | 0.946 ± 0.002 |
| enhancer | enhancers | 0.514 ± 0.005 | 0.509 ± 0.005 | 0.507 ± 0.004 | 0.500 ± 0.006 | 0.500 ± 0.005 | 0.499 ± 0.004 | 0.497 ± 0.004 | 0.496 ± 0.005 | 0.493 ± 0.004 |
| enhancer | enhancers_types | 0.466 ± 0.009 | 0.465 ± 0.008 | 0.463 ± 0.009 | 0.468 ± 0.011 | 0.465 ± 0.009 | 0.466 ± 0.009 | 0.455 ± 0.006 | 0.455 ± 0.007 | 0.454 ± 0.006 |
| **mean** | histone | 0.505 | 0.499 | 0.496 | 0.507 | 0.502 | 0.499 | 0.502 | 0.495 | 0.494 |
| **mean** | promoter | 0.764 | 0.758 | 0.759 | 0.764 | 0.758 | 0.758 | 0.803 | 0.789 | 0.795 |
| **mean** | splice | 0.934 | 0.908 | 0.915 | 0.949 | 0.893 | 0.932 | 0.962 | 0.743 | 0.944 |
| **mean** | enhancer | 0.490 | 0.487 | 0.485 | 0.484 | 0.482 | 0.482 | 0.476 | 0.476 | 0.474 |
| **mean** | all 18 | 0.618 | 0.609 | 0.609 | 0.621 | 0.608 | 0.613 | 0.626 | 0.583 | 0.617 |

Noisy sets: each score is the mean over 10 independent error draws of the set per fold.

## Tokenizer study: test MCC, errors outside the splice junction region (no edit within 100 bp of the junction)

| group | task | LCP clean | LCP ontw1 | LCP ontw3 | BPE clean | BPE ontw1 | BPE ontw3 | fixed clean | fixed ontw1 | fixed ontw3 |
|---|---|---|---|---|---|---|---|---|---|---|
| splice | splice_sites_acceptors | 0.923 ± 0.006 | 0.918 ± 0.004 | 0.907 ± 0.003 | 0.939 ± 0.006 | 0.923 ± 0.003 | 0.877 ± 0.007 | 0.957 ± 0.003 | 0.730 ± 0.056 | 0.599 ± 0.043 |
| splice | splice_sites_all | 0.936 ± 0.006 | 0.933 ± 0.004 | 0.922 ± 0.004 | 0.949 ± 0.005 | 0.935 ± 0.008 | 0.894 ± 0.026 | 0.963 ± 0.003 | 0.869 ± 0.020 | 0.769 ± 0.035 |
| splice | splice_sites_donors | 0.942 ± 0.004 | 0.933 ± 0.003 | 0.914 ± 0.006 | 0.958 ± 0.006 | 0.912 ± 0.023 | 0.838 ± 0.044 | 0.965 ± 0.003 | 0.803 ± 0.050 | 0.671 ± 0.051 |
| **mean** | splice sites | 0.934 | 0.928 | 0.914 | 0.949 | 0.923 | 0.870 | 0.962 | 0.801 | 0.679 |

Noisy sets: each score is the mean over 10 independent error draws of the set per fold.

## Tokenizer study: paired drop tests by NT18 group (drop = set - clean on the same fold; > 0: LCP loses less)

| test set | tasks | task-folds | LCP drop | BPE drop | fixed drop | LCP - BPE | LCP - BPE 95% CI | LCP drops less than BPE | LCP - fixed | LCP - fixed 95% CI | LCP drops less than fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1% | all 18 | 180 | -0.009 ± 0.001 | -0.013 ± 0.002 | -0.042 ± 0.002 | +0.004 ± 0.003 | [-0.001, +0.011] | 91/180 | +0.033 ± 0.003 | [+0.002, +0.072] | 114/180 |
| 1% | histone | 100 | -0.006 ± 0.002 | -0.005 ± 0.002 | -0.006 ± 0.004 | -0.001 ± 0.003 | [-0.003, -0.000] | 41/100 | -0.000 ± 0.005 | [-0.002, +0.002] | 52/100 |
| 1% | promoter | 30 | -0.007 ± 0.004 | -0.006 ± 0.004 | -0.014 ± 0.005 | -0.001 ± 0.005 | n/a (< 5 tasks) | 14/30 | +0.007 ± 0.007 | n/a (< 5 tasks) | 21/30 |
| 1% | splice | 30 | -0.025 ± 0.003 | -0.056 ± 0.013 | -0.219 ± 0.011 | +0.030 ± 0.011 | n/a (< 5 tasks) | 29/30 | +0.193 ± 0.012 | n/a (< 5 tasks) | 30/30 |
| 1% | enhancer | 20 | -0.003 ± 0.002 | -0.002 ± 0.002 | -0.001 ± 0.003 | -0.001 ± 0.002 | n/a (< 5 tasks) | 7/20 | -0.002 ± 0.004 | n/a (< 5 tasks) | 11/20 |
| 3% | all 18 | 180 | -0.027 ± 0.002 | -0.039 ± 0.004 | -0.075 ± 0.005 | +0.012 ± 0.005 | [-0.001, +0.028] | 94/180 | +0.049 ± 0.004 | [+0.001, +0.105] | 98/180 |
| 3% | histone | 100 | -0.020 ± 0.004 | -0.018 ± 0.002 | -0.017 ± 0.005 | -0.002 ± 0.004 | [-0.003, +0.000] | 41/100 | -0.003 ± 0.006 | [-0.006, +0.001] | 43/100 |
| 3% | promoter | 30 | -0.010 ± 0.003 | -0.011 ± 0.004 | -0.026 ± 0.007 | +0.001 ± 0.006 | n/a (< 5 tasks) | 18/30 | +0.016 ± 0.007 | n/a (< 5 tasks) | 24/30 |
| 3% | splice | 30 | -0.076 ± 0.006 | -0.156 ± 0.022 | -0.366 ± 0.015 | +0.080 ± 0.018 | n/a (< 5 tasks) | 30/30 | +0.290 ± 0.018 | n/a (< 5 tasks) | 30/30 |
| 3% | enhancer | 20 | -0.011 ± 0.002 | -0.006 ± 0.004 | -0.004 ± 0.003 | -0.005 ± 0.004 | n/a (< 5 tasks) | 5/20 | -0.007 ± 0.003 | n/a (< 5 tasks) | 1/20 |
| PBSIM3 | all 18 | 180 | -0.035 ± 0.002 | -0.046 ± 0.003 | -0.068 ± 0.003 | +0.011 ± 0.004 | [-0.001, +0.025] | 92/180 | +0.033 ± 0.003 | [+0.000, +0.073] | 97/180 |
| PBSIM3 | histone | 100 | -0.025 ± 0.003 | -0.023 ± 0.002 | -0.022 ± 0.004 | -0.001 ± 0.003 | [-0.003, +0.001] | 38/100 | -0.002 ± 0.004 | [-0.004, -0.000] | 41/100 |
| PBSIM3 | promoter | 30 | -0.017 ± 0.003 | -0.018 ± 0.004 | -0.026 ± 0.004 | +0.001 ± 0.006 | n/a (< 5 tasks) | 15/30 | +0.009 ± 0.005 | n/a (< 5 tasks) | 21/30 |
| PBSIM3 | splice | 30 | -0.102 ± 0.007 | -0.171 ± 0.016 | -0.302 ± 0.011 | +0.069 ± 0.013 | n/a (< 5 tasks) | 30/30 | +0.200 ± 0.015 | n/a (< 5 tasks) | 30/30 |
| PBSIM3 | enhancer | 20 | -0.013 ± 0.002 | -0.013 ± 0.003 | -0.010 ± 0.003 | +0.000 ± 0.003 | n/a (< 5 tasks) | 9/20 | -0.003 ± 0.003 | n/a (< 5 tasks) | 5/20 |
| 3% subst. only | all 18 | 180 | -0.031 ± 0.002 | -0.029 ± 0.003 | -0.028 ± 0.003 | -0.003 ± 0.004 | [-0.005, -0.001] | 71/180 | -0.003 ± 0.004 | [-0.006, +0.000] | 72/180 |
| 3% subst. only | histone | 100 | -0.030 ± 0.004 | -0.028 ± 0.004 | -0.026 ± 0.005 | -0.002 ± 0.005 | [-0.004, +0.000] | 41/100 | -0.004 ± 0.005 | [-0.008, -0.001] | 40/100 |
| 3% subst. only | promoter | 30 | -0.021 ± 0.004 | -0.020 ± 0.003 | -0.025 ± 0.004 | -0.001 ± 0.006 | n/a (< 5 tasks) | 14/30 | +0.005 ± 0.006 | n/a (< 5 tasks) | 17/30 |
| 3% subst. only | splice | 30 | -0.057 ± 0.003 | -0.053 ± 0.004 | -0.055 ± 0.001 | -0.004 ± 0.005 | n/a (< 5 tasks) | 10/30 | -0.003 ± 0.003 | n/a (< 5 tasks) | 14/30 |
| 3% subst. only | enhancer | 20 | -0.014 ± 0.003 | -0.007 ± 0.003 | -0.007 ± 0.003 | -0.006 ± 0.005 | n/a (< 5 tasks) | 6/20 | -0.007 ± 0.003 | n/a (< 5 tasks) | 1/20 |
| 1% subst. only | all 18 | 180 | -0.010 ± 0.001 | -0.008 ± 0.001 | -0.009 ± 0.002 | -0.001 ± 0.002 | [-0.002, -0.000] | 71/180 | -0.001 ± 0.002 | [-0.002, +0.001] | 81/180 |
| 1% subst. only | histone | 100 | -0.009 ± 0.002 | -0.008 ± 0.001 | -0.007 ± 0.003 | -0.001 ± 0.002 | [-0.003, +0.000] | 45/100 | -0.002 ± 0.004 | [-0.004, +0.000] | 45/100 |
| 1% subst. only | promoter | 30 | -0.005 ± 0.002 | -0.005 ± 0.004 | -0.009 ± 0.002 | +0.000 ± 0.005 | n/a (< 5 tasks) | 14/30 | +0.003 ± 0.003 | n/a (< 5 tasks) | 17/30 |
| 1% subst. only | splice | 30 | -0.019 ± 0.001 | -0.017 ± 0.002 | -0.018 ± 0.001 | -0.002 ± 0.002 | n/a (< 5 tasks) | 6/30 | -0.001 ± 0.001 | n/a (< 5 tasks) | 12/30 |
| 1% subst. only | enhancer | 20 | -0.004 ± 0.002 | -0.002 ± 0.003 | -0.002 ± 0.002 | -0.002 ± 0.003 | n/a (< 5 tasks) | 6/20 | -0.002 ± 0.002 | n/a (< 5 tasks) | 7/20 |

Drops and differences: mean over task-folds ± SD over folds of the per-fold mean over the row's tasks. Intervals: bootstrap over tasks (4,000 resamples, one seed per row), for rows with at least 5 tasks only (with 2 or 3 tasks it would be the range of the task means); the folds of a task share its test set and are kept together. Substitutions only: the control sets' own error schedule, every event one substitution (no shift of the segment grid). Noisy sets: each score is the mean over 10 independent error draws of the set per fold.

## GUE splice sites (4,562 test sequences of 400 bp): test MCC and drop from clean, 2M twins

| model | folds | clean test MCC | drop, 1% | drop, 3% | drop, PBSIM3 | drop, 3% subst. only | drop, 3%, no edit within 100 bp of the junction |
|---|---|---|---|---|---|---|---|
| LCP | 10 | 0.864 ± 0.004 | -0.025 ± 0.003 | -0.084 ± 0.009 | -0.103 ± 0.010 | -0.079 ± 0.006 | -0.011 ± 0.004 |
| BPE | 10 | 0.875 ± 0.005 | -0.037 ± 0.005 | -0.131 ± 0.013 | -0.145 ± 0.011 | -0.075 ± 0.002 | -0.033 ± 0.008 |
| fixed | 10 | 0.881 ± 0.008 | -0.179 ± 0.036 | -0.354 ± 0.039 | -0.253 ± 0.021 | -0.074 ± 0.005 | -0.217 ± 0.040 |
| LCP - BPE (drop) | 10 |  | +0.012 ± 0.005 (10/10) | +0.047 ± 0.011 (10/10) | +0.042 ± 0.011 (10/10) | -0.004 ± 0.006 (3/10) | +0.023 ± 0.007 (10/10) |
| LCP - fixed (drop) | 10 |  | +0.154 ± 0.037 (10/10) | +0.270 ± 0.043 (10/10) | +0.150 ± 0.025 (10/10) | -0.006 ± 0.008 (4/10) | +0.207 ± 0.041 (10/10) |

Drop = test MCC on the set - clean test MCC, same fold; mean ± SD over folds. LCP - x: LCP's drop minus x's (> 0: LCP loses less), mean ± SD over folds, (folds where LCP loses less / folds). Noisy sets: mean over 10 error draws per fold.

## GUE splice sites, test sequences that share no splice site with NT18 (3,998): test MCC and drop from clean, 2M twins

| model | folds | clean test MCC | drop, 1% | drop, 3% | drop, PBSIM3 | drop, 3% subst. only | drop, 3%, no edit within 100 bp of the junction |
|---|---|---|---|---|---|---|---|
| LCP | 10 | 0.853 ± 0.005 | -0.024 ± 0.004 | -0.081 ± 0.010 | -0.099 ± 0.011 | -0.078 ± 0.007 | -0.009 ± 0.004 |
| BPE | 10 | 0.863 ± 0.006 | -0.035 ± 0.006 | -0.127 ± 0.013 | -0.140 ± 0.011 | -0.074 ± 0.003 | -0.031 ± 0.008 |
| fixed | 10 | 0.869 ± 0.009 | -0.171 ± 0.035 | -0.344 ± 0.039 | -0.245 ± 0.021 | -0.072 ± 0.005 | -0.210 ± 0.040 |
| LCP - BPE (drop) | 10 |  | +0.011 ± 0.005 (10/10) | +0.046 ± 0.011 (10/10) | +0.041 ± 0.011 (10/10) | -0.004 ± 0.008 (4/10) | +0.022 ± 0.007 (10/10) |
| LCP - fixed (drop) | 10 |  | +0.147 ± 0.036 (10/10) | +0.263 ± 0.043 (10/10) | +0.146 ± 0.026 (10/10) | -0.006 ± 0.009 (4/10) | +0.201 ± 0.040 (10/10) |

Drop = test MCC on the set - clean test MCC, same fold; mean ± SD over folds. LCP - x: LCP's drop minus x's (> 0: LCP loses less), mean ± SD over folds, (folds where LCP loses less / folds). Noisy sets: mean over 10 error draws per fold. The 564 test sequences whose 32 bases around the junction occur in an NT18 splice record are removed (same weights).

## Real nanopore reads (HG002, R10.4.1) of the NT18 test windows: test MCC and drop from the reference sequences, 2M twins

| tasks | task-folds | model | reference sequences (realref) | drop, sup reads | drop, hac reads | LCP - model, sup (95% CI over tasks) (LCP loses less) | LCP - model, hac (95% CI over tasks) (LCP loses less) |
|---|---|---|---|---|---|---|---|
| histone | 100 | LCP | 0.492 | -0.005 | -0.007 |  |  |
|  |  | BPE | 0.494 | -0.005 | -0.007 | -0.000 [-0.001, +0.000] (50/100) | -0.000 [-0.002, +0.001] (51/100) |
|  |  | fixed | 0.488 | -0.004 | -0.007 | -0.001 [-0.002, +0.001] (53/100) | -0.001 [-0.002, +0.001] (51/100) |
| promoter | 30 | LCP | 0.764 | -0.008 | -0.015 |  |  |
|  |  | BPE | 0.761 | -0.007 | -0.014 | -0.001 n/a (< 5 tasks) (14/30) | -0.001 n/a (< 5 tasks) (17/30) |
|  |  | fixed | 0.802 | -0.021 | -0.031 | +0.012 n/a (< 5 tasks) (22/30) | +0.016 n/a (< 5 tasks) (23/30) |
| splice | 30 | LCP | 0.933 | -0.041 | -0.060 |  |  |
|  |  | BPE | 0.948 | -0.081 | -0.123 | +0.040 n/a (< 5 tasks) (30/30) | +0.063 n/a (< 5 tasks) (30/30) |
|  |  | fixed | 0.962 | -0.217 | -0.295 | +0.176 n/a (< 5 tasks) (30/30) | +0.236 n/a (< 5 tasks) (30/30) |
| enhancer | 20 | LCP | 0.479 | -0.003 | -0.006 |  |  |
|  |  | BPE | 0.472 | -0.003 | -0.005 | +0.001 n/a (< 5 tasks) (10/20) | -0.001 n/a (< 5 tasks) (6/20) |
|  |  | fixed | 0.464 | -0.003 | -0.004 | +0.000 n/a (< 5 tasks) (11/20) | -0.002 n/a (< 5 tasks) (6/20) |
| all 18 | 180 | LCP | 0.609 | -0.011 | -0.017 |  |  |
|  |  | BPE | 0.612 | -0.018 | -0.027 | +0.006 [-0.000, +0.015] (104/180) | +0.010 [-0.000, +0.022] (104/180) |
|  |  | fixed | 0.617 | -0.042 | -0.059 | +0.031 [+0.002, +0.065] (116/180) | +0.041 [+0.004, +0.088] (110/180) |

Windows with 10 reads in both basecalls (sup and hac, the same molecules); the windows left out are mostly repetitive negatives, where reads do not map uniquely (counts in summary.json); each fold's score on reads is the mean over the 10 reads per window, its drop is relative to the reference sequences of the same windows. Divergence of the reads from the reference: data/benchmarks/real_reads/summary.json. LCP - model: LCP's drop minus the model's (> 0: LCP loses less); interval: bootstrap over tasks (4,000 resamples), for rows with at least 5 tasks only (with 2 or 3 tasks it would be the range of the task means).

## LCP and single-nucleotide models: NT18, simulated errors (drop = set - clean test set)

| test set | tasks | task-folds | LCP drop | Caduceus drop | LCP - Caduceus | LCP - Caduceus 95% CI | LCP loses less than Caduceus |
|---|---|---|---|---|---|---|---|
| 1% | all 18 | 180 | -0.009 | -0.008 | -0.001 | [-0.003, +0.001] | 80/180 |
| 1% | histone | 100 | -0.006 | -0.006 | -0.000 | [-0.002, +0.001] | 47/100 |
| 1% | promoter | 30 | -0.007 | -0.008 | +0.001 | n/a (< 5 tasks) | 16/30 |
| 1% | splice | 30 | -0.025 | -0.019 | -0.007 | n/a (< 5 tasks) | 6/30 |
| 1% | enhancer | 20 | -0.003 | -0.004 | +0.001 | n/a (< 5 tasks) | 11/20 |
| 3% | all 18 | 180 | -0.027 | -0.024 | -0.003 | [-0.008, +0.002] | 80/180 |
| 3% | histone | 100 | -0.020 | -0.017 | -0.003 | [-0.006, +0.001] | 45/100 |
| 3% | promoter | 30 | -0.010 | -0.020 | +0.010 | n/a (< 5 tasks) | 22/30 |
| 3% | splice | 30 | -0.076 | -0.058 | -0.018 | n/a (< 5 tasks) | 5/30 |
| 3% | enhancer | 20 | -0.011 | -0.009 | -0.002 | n/a (< 5 tasks) | 8/20 |
| PBSIM3 | all 18 | 180 | -0.035 | -0.031 | -0.004 | [-0.012, +0.002] | 79/180 |
| PBSIM3 | histone | 100 | -0.025 | -0.022 | -0.003 | [-0.006, -0.000] | 38/100 |
| PBSIM3 | promoter | 30 | -0.017 | -0.027 | +0.010 | n/a (< 5 tasks) | 24/30 |
| PBSIM3 | splice | 30 | -0.102 | -0.076 | -0.025 | n/a (< 5 tasks) | 6/30 |
| PBSIM3 | enhancer | 20 | -0.013 | -0.013 | +0.000 | n/a (< 5 tasks) | 11/20 |
| 3% subst. only | all 18 | 180 | -0.031 | -0.028 | -0.003 | [-0.006, -0.000] | 72/180 |
| 3% subst. only | histone | 100 | -0.030 | -0.026 | -0.004 | [-0.008, -0.000] | 37/100 |
| 3% subst. only | promoter | 30 | -0.021 | -0.017 | -0.003 | n/a (< 5 tasks) | 12/30 |
| 3% subst. only | splice | 30 | -0.057 | -0.057 | -0.000 | n/a (< 5 tasks) | 16/30 |
| 3% subst. only | enhancer | 20 | -0.014 | -0.011 | -0.002 | n/a (< 5 tasks) | 7/20 |
| 1% subst. only | all 18 | 180 | -0.010 | -0.008 | -0.001 | [-0.002, -0.001] | 81/180 |
| 1% subst. only | histone | 100 | -0.009 | -0.007 | -0.002 | [-0.003, -0.001] | 45/100 |
| 1% subst. only | promoter | 30 | -0.005 | -0.004 | -0.002 | n/a (< 5 tasks) | 14/30 |
| 1% subst. only | splice | 30 | -0.019 | -0.018 | -0.001 | n/a (< 5 tasks) | 13/30 |
| 1% subst. only | enhancer | 20 | -0.004 | -0.003 | -0.001 | n/a (< 5 tasks) | 9/20 |

Single-nucleotide models: Caduceus (caduceus-ps-1.9m), fine-tuned like the other baselines; their tokens are nucleotides, so a drop is the loss from the read errors without any merging into tokens. Drops: mean over the task-folds scored for every model (each noisy set or basecall: mean over its 10 draws); n/a: the model has not yet been scored on every task-fold of the row. LCP - x > 0: LCP loses less. Intervals: bootstrap over tasks (4,000 resamples), for rows with at least 5 tasks only.

## LCP and single-nucleotide models: NT18 test windows as real HG002 nanopore reads (drop = reads - reference sequences of the same windows)

| test set | tasks | task-folds | LCP drop | Caduceus drop | LCP - Caduceus | LCP - Caduceus 95% CI | LCP loses less than Caduceus |
|---|---|---|---|---|---|---|---|
| sup reads | all 18 | 180 | -0.011 | -0.009 | -0.002 | [-0.005, +0.000] | 70/180 |
| sup reads | histone | 100 | -0.005 | -0.004 | -0.001 | [-0.003, +0.001] | 38/100 |
| sup reads | promoter | 30 | -0.008 | -0.005 | -0.003 | n/a (< 5 tasks) | 14/30 |
| sup reads | splice | 30 | -0.041 | -0.035 | -0.006 | n/a (< 5 tasks) | 9/30 |
| sup reads | enhancer | 20 | -0.003 | -0.003 | -0.000 | n/a (< 5 tasks) | 9/20 |
| hac reads | all 18 | 180 | -0.017 | -0.015 | -0.002 | [-0.006, +0.001] | 75/180 |
| hac reads | histone | 100 | -0.007 | -0.007 | -0.001 | [-0.003, +0.002] | 46/100 |
| hac reads | promoter | 30 | -0.015 | -0.011 | -0.004 | n/a (< 5 tasks) | 12/30 |
| hac reads | splice | 30 | -0.060 | -0.053 | -0.007 | n/a (< 5 tasks) | 9/30 |
| hac reads | enhancer | 20 | -0.006 | -0.005 | -0.001 | n/a (< 5 tasks) | 8/20 |

Single-nucleotide models: Caduceus (caduceus-ps-1.9m), fine-tuned like the other baselines; their tokens are nucleotides, so a drop is the loss from the read errors without any merging into tokens. Drops: mean over the task-folds scored for every model (each noisy set or basecall: mean over its 10 draws); n/a: the model has not yet been scored on every task-fold of the row. LCP - x > 0: LCP loses less. Intervals: bootstrap over tasks (4,000 resamples), for rows with at least 5 tasks only.

## Robustness with intervals: NT18, simulated errors (drop = set - clean test set)

| tasks | task-folds | set | LCP drop | BPE drop | fixed drop | Caduceus drop | LCP - BPE | LCP - fixed | LCP - Caduceus |
|---|---|---|---|---|---|---|---|---|---|
| splice | 30 | errors 1% | -0.025 [-0.027, -0.024] | -0.056 [-0.058, -0.053] | -0.219 [-0.225, -0.212] | -0.019 [-0.020, -0.017] | +0.030 [+0.028, +0.033] | +0.193 [+0.187, +0.199] | -0.007 [-0.009, -0.005] |
| splice | 30 | errors 3% | -0.076 [-0.079, -0.072] | -0.156 [-0.161, -0.151] | -0.366 [-0.374, -0.358] | -0.058 [-0.061, -0.055] | +0.080 [+0.076, +0.084] | +0.290 [+0.284, +0.297] | -0.018 [-0.021, -0.015] |
| splice | 30 | PBSIM3 reads | -0.102 [-0.105, -0.098] | -0.171 [-0.175, -0.166] | -0.302 [-0.308, -0.296] | -0.076 [-0.080, -0.073] | +0.069 [+0.066, +0.072] | +0.200 [+0.195, +0.205] | -0.025 [-0.029, -0.022] |
| splice | 30 | 3%, substitutions only | -0.057 [-0.060, -0.055] | -0.053 [-0.056, -0.051] | -0.055 [-0.057, -0.052] | -0.057 [-0.060, -0.055] | -0.004 [-0.006, -0.002] | -0.003 [-0.005, -0.000] | -0.000 [-0.002, +0.003] |
| splice | 30 | 3%, no edit within 100 bp of the junction | -0.019 [-0.021, -0.017] | -0.079 [-0.083, -0.075] | -0.282 [-0.290, -0.275] | -0.008 [-0.009, -0.007] | +0.060 [+0.056, +0.063] | +0.263 [+0.256, +0.270] | -0.011 [-0.013, -0.009] |
| promoter | 30 | errors 1% | -0.007 [-0.011, -0.003] | -0.006 [-0.009, -0.003] | -0.014 [-0.021, -0.007] | -0.008 [-0.013, -0.003] | -0.001 [-0.004, +0.003] | +0.007 [+0.000, +0.013] | +0.001 [-0.004, +0.006] |
| promoter | 30 | errors 3% | -0.010 [-0.016, -0.003] | -0.011 [-0.018, -0.005] | -0.026 [-0.035, -0.016] | -0.020 [-0.028, -0.013] | +0.001 [-0.005, +0.008] | +0.016 [+0.006, +0.026] | +0.010 [+0.002, +0.018] |
| promoter | 30 | PBSIM3 reads | -0.017 [-0.023, -0.011] | -0.018 [-0.023, -0.012] | -0.026 [-0.032, -0.019] | -0.027 [-0.035, -0.019] | +0.001 [-0.004, +0.006] | +0.009 [+0.002, +0.016] | +0.010 [+0.002, +0.017] |
| promoter | 30 | 3%, substitutions only | -0.021 [-0.028, -0.013] | -0.020 [-0.027, -0.012] | -0.025 [-0.032, -0.019] | -0.017 [-0.024, -0.011] | -0.001 [-0.008, +0.006] | +0.005 [-0.003, +0.012] | -0.003 [-0.011, +0.005] |

Drop: mean over the task-folds of [mean over the 10 error draws (or reads) of the MCC on the set - the MCC on the reference set]; LCP - x > 0: LCP loses less. Intervals: 2.5th-97.5th percentile over 2,000 resamples of the test windows (the same resample for every model, fold and set; benchmarks/noisy_reads/window_intervals.py). Only task-folds scored for every model are used.

## Robustness with intervals: NT18 test windows as real HG002 nanopore reads (drop = reads - reference sequences of the same windows)

| tasks | task-folds | set | LCP drop | BPE drop | fixed drop | Caduceus drop | LCP - BPE | LCP - fixed | LCP - Caduceus |
|---|---|---|---|---|---|---|---|---|---|
| splice | 30 | sup reads | -0.041 [-0.044, -0.038] | -0.081 [-0.084, -0.077] | -0.217 [-0.223, -0.211] | -0.035 [-0.037, -0.032] | +0.040 [+0.038, +0.043] | +0.176 [+0.171, +0.181] | -0.006 [-0.009, -0.003] |
| splice | 30 | hac reads | -0.060 [-0.063, -0.056] | -0.123 [-0.127, -0.118] | -0.295 [-0.302, -0.288] | -0.053 [-0.056, -0.050] | +0.063 [+0.059, +0.066] | +0.236 [+0.230, +0.241] | -0.007 [-0.011, -0.004] |
| promoter | 30 | sup reads | -0.008 [-0.014, -0.002] | -0.007 [-0.013, -0.001] | -0.021 [-0.029, -0.012] | -0.005 [-0.012, +0.001] | -0.001 [-0.006, +0.004] | +0.012 [+0.005, +0.020] | -0.003 [-0.010, +0.004] |
| promoter | 30 | hac reads | -0.015 [-0.022, -0.008] | -0.014 [-0.021, -0.007] | -0.031 [-0.040, -0.020] | -0.011 [-0.018, -0.003] | -0.001 [-0.007, +0.004] | +0.016 [+0.007, +0.024] | -0.004 [-0.012, +0.003] |

Drop: mean over the task-folds of [mean over the 10 error draws (or reads) of the MCC on the set - the MCC on the reference set]; LCP - x > 0: LCP loses less. Intervals: 2.5th-97.5th percentile over 2,000 resamples of the test windows (the same resample for every model, fold and set; benchmarks/noisy_reads/window_intervals.py). Only task-folds scored for every model are used.

## Robustness with intervals: GUE splice sites, simulated errors (drop = set - clean test set)

| tasks | task-folds | set | LCP drop | BPE drop | fixed drop | LCP - BPE | LCP - fixed |
|---|---|---|---|---|---|---|---|
| gue_splice | 10 | errors 1% | -0.025 [-0.028, -0.023] | -0.037 [-0.041, -0.034] | -0.179 [-0.188, -0.170] | +0.012 [+0.009, +0.015] | +0.154 [+0.145, +0.162] |
| gue_splice | 10 | errors 3% | -0.084 [-0.089, -0.079] | -0.131 [-0.137, -0.124] | -0.354 [-0.365, -0.342] | +0.047 [+0.042, +0.052] | +0.270 [+0.260, +0.279] |
| gue_splice | 10 | PBSIM3 reads | -0.103 [-0.107, -0.098] | -0.145 [-0.151, -0.139] | -0.253 [-0.260, -0.245] | +0.042 [+0.039, +0.046] | +0.150 [+0.144, +0.156] |
| gue_splice | 10 | 3%, substitutions only | -0.079 [-0.084, -0.075] | -0.075 [-0.080, -0.071] | -0.074 [-0.078, -0.069] | -0.004 [-0.008, -0.001] | -0.006 [-0.009, -0.002] |
| gue_splice | 10 | 3%, no edit within 100 bp of the junction | -0.011 [-0.013, -0.008] | -0.033 [-0.037, -0.030] | -0.217 [-0.227, -0.207] | +0.023 [+0.020, +0.026] | +0.207 [+0.197, +0.216] |

Drop: mean over the task-folds of [mean over the 10 error draws (or reads) of the MCC on the set - the MCC on the reference set]; LCP - x > 0: LCP loses less. Intervals: 2.5th-97.5th percentile over 2,000 resamples of the test windows (the same resample for every model, fold and set; benchmarks/noisy_reads/window_intervals.py). Only task-folds scored for every model are used.

## Parsing: segments changed by one edit (4 kb chromosome-1 windows, edit at least 1 kb from both ends)

| segmenter | edit | changed segments (mean) | reach, median (bp) | reach, 95th percentile (bp) | reach, max (bp) |
|---|---|---|---|---|---|
| LCP | substitution | 2.1 | 9 | 13 | 49 |
| LCP | insertion 1 | 1.9 | 8 | 13 | 38 |
| LCP | deletion 1 | 2.0 | 9 | 13 | 35 |
| LCP | insertion 3 | 1.9 | 8 | 13 | 61 |
| LCP | deletion 3 | 2.4 | 10 | 14 | 47 |
| fixed | substitution | 2.0 | 8 | 9 | 9 |
| fixed | insertion 1 | 412.9 | 2057 | 2972 | 3071 |
| fixed | deletion 1 | 410.8 | 2060 | 2958 | 3069 |
| fixed | insertion 3 | 408.5 | 2038 | 2954 | 3070 |
| fixed | deletion 3 | 410.0 | 2044 | 2966 | 3071 |
| BPE | substitution | 1.6 | 5 | 12 | 52 |
| BPE | insertion 1 | 1.5 | 5 | 13 | 34 |
| BPE | deletion 1 | 1.7 | 5 | 13 | 36 |
| BPE | insertion 3 | 1.5 | 5 | 12 | 68 |
| BPE | deletion 3 | 2.1 | 7 | 14 | 26 |

2,000 edits of each kind; reach = largest distance from the edit to a base of a changed segment. Segments: LCP 10.5 bp on average, 191 per kb; fixed 10.0 bp on average, 200 per kb; BPE 5.3 bp on average, 189 per kb.

## Parsing: the same 300 bp stretch in two different contexts (share of its segments parsed identically)

| segmenter | all segments | 0-10 bp from the stretch end | 10-20 bp from the stretch end | 20-50 bp from the stretch end | 50-150 bp from the stretch end |
|---|---|---|---|---|---|
| LCP | 0.999 | 0.983 | 1.000 | 1.000 | 1.000 |
| fixed | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 |
| BPE | 0.988 | 0.832 | 0.996 | 1.000 | 1.000 |

Flanks of 496-500 bp from other chromosome-1 windows; a segment counts when it lies inside the stretch.

## LCP segmentation statistics (2,000 windows of 4,096 bp from the pretraining chromosomes)

| cores | per kb | mean start spacing (bp) | bases per core | mean length (bp) | max length (bp) | core types |
|---|---|---|---|---|---|---|
| level 1 | 442 | 2.26 | 2.26 | 3.68 | 56 | between cores 0.5%, local maximum 10.1%, local minimum 45.0%, run 44.4% |
| level 2 (segments) | 190 | 5.24 | 5.26 | 10.50 | 85 | between cores 8.1%, local maximum 9.3%, local minimum 82.1%, run 0.5% |

Level-1 cores other than runs with 3 bases: 100.0%. Bases covered by 0 / 1 / 2 / 3 / 4 or more segments: 0.1% / 14.1% / 71.8% / 14.0% / 0.0% (85.8% in two or more). Bases at the window start / end covered by no segment: 3.0 / 3.1 on average (by no level-1 core: 0.9 / 0.9).

## LCP worst-case edit locality: one edit at the centre of 10,000 sequences of 2,048 bp per set

| sequences | cores | edits | changed cores (mean) | changed cores (max) | reach, median (bp) | reach, 99th percentile (bp) | reach, max (bp) | changed cores (max), no homopolymer of 6+ bp within 40 bp | reach, max (bp), no homopolymer of 6+ bp within 40 bp |
|---|---|---|---|---|---|---|---|---|---|
| chromosome 1 | level 1 | 70,000 | 1.68 | 4 | 3 | 8 | 30 (27 bp homopolymer nearby) | 4 | 8 |
| chromosome 1 | level 2 | 70,000 | 2.08 | 5 | 9 | 17 | 67 (23 bp homopolymer nearby) | 5 | 40 |
| random | level 1 | 70,000 | 1.66 | 4 | 3 | 6 | 10 (11 bp homopolymer nearby) | 4 | 8 |
| random | level 2 | 70,000 | 2.05 | 5 | 8 | 14 | 20 (5 bp homopolymer nearby) | 5 | 20 |

Edits: a substitution, or an insertion or deletion of 1, 2 or 3 bases, each on every sequence. A core is changed when the edited parse has no core with the same bases at the same, shifted position; reach = largest distance from the edit to a base of a changed core.

## Segment boundaries and masked bases: accuracy of the pretrained models on the bases they have to predict (1,000 sequences per set)

| model | sequences | hidden by MASK | overlap symbol | random bases | left unchanged | all targets | bases hidden by MASK |
|---|---|---|---|---|---|---|---|
| locus-0.5m | random | 0.444 | 0.435 | 0.399 | 0.940 | 0.485 | 450,774 |
| locus-0.5m | chromosome 1 | 0.586 | 0.570 | 0.526 | 0.968 | 0.615 | 468,962 |
| locus-2m | random | 0.449 | 0.441 | 0.432 | 0.989 | 0.497 | 450,774 |
| locus-2m | chromosome 1 | 0.610 | 0.598 | 0.582 | 0.993 | 0.642 | 468,962 |
| twin-fixed-2m | random | 0.249 | 0.250 | 0.250 | 0.987 | 0.317 | 459,410 |
| twin-fixed-2m | chromosome 1 | 0.439 | 0.415 | 0.383 | 0.980 | 0.483 | 473,995 |
| twin-bpe-2m | random | 0.405 | - | 0.393 | 0.991 | 0.463 | 252,957 |
| twin-bpe-2m | chromosome 1 | 0.539 | - | 0.513 | 0.993 | 0.581 | 260,260 |

Crops, segmentation and masking as in pretraining (benchmarks/parsing/boundary_information.py): the clean crop is segmented and 15% of its segments are hidden (MASK 80%, random bases 10%, unchanged 10%; the overlap symbol where two hidden segments disagree). random: uniform random sequences, whose context carries no information, so accuracy above chance (25%) on bases hidden by MASK comes from the segment boundaries; chromosome 1: windows held out of pretraining. twin-fixed-2m (boundaries independent of the bases) is the control.

## Embedding stability: one edit at the centre of 10,000 chromosome-1 sequences of 2,048 bp (frozen 2M twins)

| edit | changed segments per strand, LCP | changed segments per strand, BPE | changed segments per strand, fixed | per-base change within 10 bp, LCP | per-base change within 10 bp, BPE | per-base change within 10 bp, fixed | per-base change <= 0.01 from (bp), LCP | per-base change <= 0.01 from (bp), BPE | per-base change <= 0.01 from (bp), fixed | per-base change 500-900 bp away, LCP | per-base change 500-900 bp away, BPE | per-base change 500-900 bp away, fixed | read change x 1e-4, LCP | read change x 1e-4, BPE | read change x 1e-4, fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| substitution | 2.09 | 1.67 | 2.00 | 0.201 | 0.209 | 0.066 | 13 | 15 | 11 | 0.000 | 0.000 | 0.000 | 2.44 | 1.64 | 0.62 |
| insertion 1 | 1.88 | 1.50 | 206.00 | 0.181 | 0.217 | 0.240 | 13 | 15 | never | 0.000 | 0.000 | 0.183 | 2.07 | 1.57 | 79.46 |
| insertion 2 | 1.88 | 1.54 | 206.00 | 0.207 | 0.235 | 0.276 | 13 | 15 | never | 0.000 | 0.000 | 0.193 | 2.60 | 1.84 | 83.03 |
| insertion 3 | 1.88 | 1.52 | 206.00 | 0.222 | 0.233 | 0.297 | 13 | 16 | never | 0.000 | 0.000 | 0.193 | 3.03 | 2.05 | 83.45 |
| deletion 1 | 2.07 | 1.71 | 206.00 | 0.171 | 0.216 | 0.231 | 12 | 15 | never | 0.000 | 0.000 | 0.183 | 1.94 | 1.56 | 79.25 |
| deletion 2 | 2.26 | 1.93 | 206.00 | 0.194 | 0.230 | 0.252 | 13 | 15 | never | 0.000 | 0.000 | 0.193 | 2.43 | 1.79 | 82.94 |
| deletion 3 | 2.46 | 2.10 | 206.00 | 0.212 | 0.229 | 0.266 | 13 | 16 | never | 0.000 | 0.000 | 0.193 | 2.90 | 2.00 | 83.19 |

Changed segments: no model. Per-base change: (1 - cos(u, u')) / (1 - c0) per base, c0 = mean cosine of unrelated bases of the model (LCP 0.572, BPE 0.348, fixed 0.504). Read change: the same for the read embedding (mean over bases), c0 between reads of different sequences (LCP 0.950, BPE 0.891, fixed 0.852).

## Embedding stability: paired ratios over sequences (95% bootstrap interval)

| edit | BPE/LCP changed segments | fixed/LCP changed segments | BPE/LCP per-base change within 10 bp | fixed/LCP per-base change within 10 bp | BPE/LCP read change | fixed/LCP read change |
|---|---|---|---|---|---|---|
| substitution | 0.799 [0.792, 0.805] | 0.959 [0.955, 0.963] | 1.037 [1.022, 1.053] | 0.328 [0.320, 0.336] | 0.672 [0.656, 0.688] | 0.256 [0.248, 0.265] |
| insertion 1 | 0.797 [0.791, 0.804] | 109.849 [109.397, 110.335] | 1.200 [1.185, 1.215] | 1.324 [1.310, 1.338] | 0.757 [0.739, 0.774] | 38.364 [37.610, 39.089] |
| insertion 2 | 0.816 [0.810, 0.823] | 109.374 [108.903, 109.817] | 1.134 [1.123, 1.146] | 1.333 [1.322, 1.345] | 0.707 [0.692, 0.723] | 31.918 [31.318, 32.485] |
| insertion 3 | 0.811 [0.804, 0.818] | 109.563 [109.113, 110.043] | 1.052 [1.042, 1.063] | 1.340 [1.328, 1.352] | 0.676 [0.663, 0.689] | 27.556 [27.082, 27.997] |
| deletion 1 | 0.826 [0.820, 0.833] | 99.746 [99.346, 100.148] | 1.266 [1.249, 1.283] | 1.351 [1.337, 1.365] | 0.804 [0.786, 0.824] | 40.900 [40.116, 41.669] |
| deletion 2 | 0.853 [0.847, 0.859] | 91.022 [90.673, 91.385] | 1.187 [1.174, 1.200] | 1.299 [1.288, 1.311] | 0.740 [0.725, 0.755] | 34.197 [33.664, 34.698] |
| deletion 3 | 0.854 [0.848, 0.859] | 83.864 [83.570, 84.164] | 1.081 [1.070, 1.092] | 1.252 [1.242, 1.263] | 0.689 [0.675, 0.703] | 28.664 [28.156, 29.159] |

Ratio of the means over sequences; > 1: the other segmenter changes more than LCP. Paired bootstrap over sequences (2,000 resamples).

## BEND zero-shot noncoding variant effects (AUROC over all variants, no training)

| model | expression | disease | mean of the 2 | expression, held-out chromosomes | disease, held-out chromosomes |
|---|---|---|---|---|---|
| LoCUS-0.5M | 0.516 | 0.530 | 0.523 | 0.516 | 0.532 |
| LoCUS-2M | 0.507 | 0.531 | 0.519 | 0.509 | 0.532 |

Held-out chromosomes: 1, 2, X and Y, not used for pretraining. Every model in float32.

## Pretraining-split ablation: NT18 test MCC without the test chromosomes in pretraining

| group | task | LoCUS-0.5M, released (chromosomes 3-22) | LoCUS-0.5M, chromosomes 1-19 and 22 | LoCUS-2M, released (chromosomes 3-22) | LoCUS-2M, chromosomes 1-19 and 22 | fixed-segment twin, released (chromosomes 3-22) | fixed-segment twin, chromosomes 1-19 and 22 | BPE twin, released (chromosomes 3-22) | BPE twin, chromosomes 1-19 and 22 |
|---|---|---|---|---|---|---|---|---|---|
| histone | H3K4me3 | 0.609 ± 0.012 | 0.611 ± 0.010 | 0.615 ± 0.014 | 0.602 ± 0.008 | 0.607 ± 0.009 | 0.608 ± 0.013 | 0.622 ± 0.011 | 0.613 ± 0.017 |
| histone | H3K27ac | 0.463 ± 0.009 | 0.463 ± 0.009 | 0.469 ± 0.009 | 0.462 ± 0.019 | 0.454 ± 0.011 | 0.453 ± 0.016 | 0.469 ± 0.008 | 0.461 ± 0.008 |
| histone | H3K36me3 | 0.525 ± 0.006 | 0.522 ± 0.009 | 0.527 ± 0.012 | 0.529 ± 0.007 | 0.536 ± 0.008 | 0.541 ± 0.006 | 0.535 ± 0.006 | 0.538 ± 0.013 |
| histone | H3K9me3 | 0.393 ± 0.023 | 0.380 ± 0.014 | 0.385 ± 0.015 | 0.380 ± 0.013 | 0.390 ± 0.018 | 0.387 ± 0.020 | 0.381 ± 0.025 | 0.393 ± 0.017 |
| histone | H2AFZ | 0.455 ± 0.009 | 0.448 ± 0.006 | 0.460 ± 0.013 | 0.454 ± 0.006 | 0.437 ± 0.006 | 0.437 ± 0.009 | 0.451 ± 0.006 | 0.449 ± 0.006 |
| histone | H3K27me3 | 0.542 ± 0.009 | 0.540 ± 0.007 | 0.543 ± 0.007 | 0.544 ± 0.007 | 0.545 ± 0.008 | 0.547 ± 0.010 | 0.552 ± 0.004 | 0.549 ± 0.008 |
| histone | H3K4me1 | 0.432 ± 0.009 | 0.430 ± 0.006 | 0.445 ± 0.010 | 0.443 ± 0.007 | 0.447 ± 0.009 | 0.447 ± 0.013 | 0.448 ± 0.011 | 0.450 ± 0.011 |
| histone | H3K4me2 | 0.501 ± 0.007 | 0.496 ± 0.006 | 0.505 ± 0.011 | 0.505 ± 0.009 | 0.504 ± 0.010 | 0.503 ± 0.011 | 0.508 ± 0.011 | 0.504 ± 0.014 |
| histone | H3K9ac | 0.508 ± 0.017 | 0.511 ± 0.009 | 0.512 ± 0.010 | 0.525 ± 0.018 | 0.512 ± 0.022 | 0.521 ± 0.016 | 0.516 ± 0.015 | 0.531 ± 0.013 |
| histone | H4K20me1 | 0.591 ± 0.008 | 0.589 ± 0.009 | 0.590 ± 0.012 | 0.585 ± 0.007 | 0.584 ± 0.007 | 0.585 ± 0.006 | 0.586 ± 0.008 | 0.582 ± 0.008 |
| promoter | promoter_all | 0.721 ± 0.004 | 0.721 ± 0.006 | 0.716 ± 0.008 | 0.723 ± 0.010 | 0.739 ± 0.014 | 0.732 ± 0.012 | 0.719 ± 0.010 | 0.721 ± 0.010 |
| promoter | promoter_no_tata | 0.746 ± 0.009 | 0.739 ± 0.011 | 0.734 ± 0.011 | 0.743 ± 0.005 | 0.746 ± 0.007 | 0.744 ± 0.009 | 0.734 ± 0.007 | 0.734 ± 0.009 |
| promoter | promoter_tata | 0.821 ± 0.049 | 0.766 ± 0.048 | 0.843 ± 0.022 | 0.863 ± 0.022 | 0.924 ± 0.020 | 0.915 ± 0.025 | 0.837 ± 0.019 | 0.851 ± 0.022 |
| splice | splice_sites_all | 0.925 ± 0.007 | 0.925 ± 0.004 | 0.936 ± 0.006 | 0.940 ± 0.004 | 0.963 ± 0.003 | 0.961 ± 0.003 | 0.949 ± 0.005 | 0.951 ± 0.004 |
| splice | splice_sites_acceptors | 0.917 ± 0.007 | 0.919 ± 0.006 | 0.923 ± 0.006 | 0.929 ± 0.006 | 0.957 ± 0.003 | 0.952 ± 0.004 | 0.939 ± 0.006 | 0.938 ± 0.007 |
| splice | splice_sites_donors | 0.930 ± 0.007 | 0.930 ± 0.010 | 0.942 ± 0.004 | 0.944 ± 0.006 | 0.965 ± 0.003 | 0.966 ± 0.003 | 0.958 ± 0.006 | 0.957 ± 0.004 |
| enhancer | enhancers | 0.505 ± 0.007 | 0.506 ± 0.009 | 0.514 ± 0.005 | 0.512 ± 0.007 | 0.497 ± 0.004 | 0.501 ± 0.014 | 0.500 ± 0.006 | 0.510 ± 0.006 |
| enhancer | enhancers_types | 0.477 ± 0.007 | 0.469 ± 0.005 | 0.466 ± 0.009 | 0.472 ± 0.005 | 0.455 ± 0.006 | 0.470 ± 0.010 | 0.468 ± 0.011 | 0.464 ± 0.008 |
| **mean** | histone | 0.502 | 0.499 | 0.505 | 0.503 | 0.502 | 0.503 | 0.507 | 0.507 |
| **mean** | promoter | 0.763 | 0.742 | 0.764 | 0.776 | 0.803 | 0.797 | 0.764 | 0.769 |
| **mean** | splice | 0.924 | 0.925 | 0.934 | 0.938 | 0.962 | 0.960 | 0.949 | 0.948 |
| **mean** | enhancer | 0.491 | 0.488 | 0.490 | 0.492 | 0.476 | 0.485 | 0.484 | 0.487 |
| **mean** | all 18 | 0.614 | 0.609 | 0.618 | 0.620 | 0.626 | 0.626 | 0.621 | 0.622 |

NT18 tests on chromosomes 20 and 21, which the released models saw in pretraining (chromosomes 3-22) and the models of the split ablation did not (chromosomes 1-19 and 22; scripts/reproduce.py split-pretrain). Same recipe, micro-batches, folds and seeds as the released model's runs (plans nt18 and tokenizer-study); mean ± SD over the folds.

## Pretraining-split ablation: test MCC, without chromosomes 20 and 21 minus released (same task and fold)

| model | tasks | mean difference | 95% CI (tasks resampled) | task-folds higher |
|---|---|---|---|---|
| LoCUS-0.5M | histone | -0.003 | [-0.006, -0.000] | 42/100 |
| LoCUS-0.5M | promoter | -0.020 | n/a (< 5 tasks) | 9/30 |
| LoCUS-0.5M | splice | +0.001 | n/a (< 5 tasks) | 15/30 |
| LoCUS-0.5M | enhancer | -0.003 | n/a (< 5 tasks) | 6/20 |
| LoCUS-0.5M | all 18 | -0.005 | [-0.012, -0.001] | 72/180 |
| LoCUS-2M | histone | -0.002 | [-0.006, +0.002] | 45/100 |
| LoCUS-2M | promoter | +0.012 | n/a (< 5 tasks) | 22/30 |
| LoCUS-2M | splice | +0.004 | n/a (< 5 tasks) | 20/30 |
| LoCUS-2M | enhancer | +0.003 | n/a (< 5 tasks) | 11/20 |
| LoCUS-2M | all 18 | +0.002 | [-0.002, +0.005] | 98/180 |
| fixed-segment twin | histone | +0.001 | [-0.001, +0.004] | 51/100 |
| fixed-segment twin | promoter | -0.006 | n/a (< 5 tasks) | 14/30 |
| fixed-segment twin | splice | -0.002 | n/a (< 5 tasks) | 12/30 |
| fixed-segment twin | enhancer | +0.009 | n/a (< 5 tasks) | 14/20 |
| fixed-segment twin | all 18 | +0.000 | [-0.002, +0.003] | 91/180 |
| BPE twin | histone | -0.000 | [-0.005, +0.005] | 51/100 |
| BPE twin | promoter | +0.005 | n/a (< 5 tasks) | 18/30 |
| BPE twin | splice | -0.000 | n/a (< 5 tasks) | 12/30 |
| BPE twin | enhancer | +0.003 | n/a (< 5 tasks) | 12/20 |
| BPE twin | all 18 | +0.001 | [-0.002, +0.004] | 93/180 |

> 0: the model that never saw the NT18 test chromosomes scores higher. Interval: bootstrap over tasks (4,000 resamples; the folds of a task stay together), for groups of at least 5 tasks.

## Pretraining-split ablation: lowest validation loss, without chromosomes 20 and 21 minus released

| model | tasks | mean difference | 95% CI (tasks resampled) | task-folds higher |
|---|---|---|---|---|
| LoCUS-0.5M | histone | +0.0019 | [+0.0004, +0.0033] | 64/100 |
| LoCUS-0.5M | promoter | +0.0175 | n/a (< 5 tasks) | 19/30 |
| LoCUS-0.5M | splice | +0.0030 | n/a (< 5 tasks) | 18/30 |
| LoCUS-0.5M | enhancer | -0.0005 | n/a (< 5 tasks) | 10/20 |
| LoCUS-0.5M | all 18 | +0.0044 | [+0.0007, +0.0103] | 111/180 |
| LoCUS-2M | histone | -0.0013 | [-0.0028, +0.0000] | 46/100 |
| LoCUS-2M | promoter | +0.0018 | n/a (< 5 tasks) | 11/30 |
| LoCUS-2M | splice | -0.0028 | n/a (< 5 tasks) | 9/30 |
| LoCUS-2M | enhancer | -0.0006 | n/a (< 5 tasks) | 9/20 |
| LoCUS-2M | all 18 | -0.0010 | [-0.0022, +0.0005] | 75/180 |
| fixed-segment twin | histone | -0.0021 | [-0.0037, -0.0006] | 37/100 |
| fixed-segment twin | promoter | -0.0056 | n/a (< 5 tasks) | 10/30 |
| fixed-segment twin | splice | +0.0013 | n/a (< 5 tasks) | 22/30 |
| fixed-segment twin | enhancer | -0.0025 | n/a (< 5 tasks) | 7/20 |
| fixed-segment twin | all 18 | -0.0022 | [-0.0038, -0.0008] | 76/180 |
| BPE twin | histone | +0.0011 | [-0.0001, +0.0024] | 62/100 |
| BPE twin | promoter | -0.0029 | n/a (< 5 tasks) | 17/30 |
| BPE twin | splice | +0.0008 | n/a (< 5 tasks) | 17/30 |
| BPE twin | enhancer | +0.0023 | n/a (< 5 tasks) | 14/20 |
| BPE twin | all 18 | +0.0005 | [-0.0010, +0.0018] | 110/180 |

> 0: the model without chromosomes 20 and 21 has the higher (worse) validation loss at its selected epoch. The validation folds come from the NT18 training sets (chromosomes 1-19, X and Y), never from chromosomes 20 and 21; about 18% of their records lie on chromosomes 1 and 2, which only the models without chromosomes 20 and 21 saw in pretraining. A test difference matched by a validation difference in the same direction is a difference between the pretraining runs; a test difference without one would point to the test chromosomes.

## Pretraining-split ablation: drops on the modified test sets and real reads, without chromosomes 20 and 21 minus released (same task and fold)

| model, test set | tasks | mean difference of drops | 95% CI (tasks resampled) | task-folds higher |
|---|---|---|---|---|
| LoCUS-2M, 1% | histone | +0.001 | [-0.001, +0.003] | 55/100 |
| LoCUS-2M, 1% | promoter | -0.001 | n/a (< 5 tasks) | 10/30 |
| LoCUS-2M, 1% | splice | -0.000 | n/a (< 5 tasks) | 15/30 |
| LoCUS-2M, 1% | enhancer | +0.000 | n/a (< 5 tasks) | 10/20 |
| LoCUS-2M, 1% | all 18 | +0.000 | [-0.001, +0.001] | 90/180 |
| LoCUS-2M, 3% | histone | +0.002 | [-0.002, +0.006] | 62/100 |
| LoCUS-2M, 3% | promoter | -0.006 | n/a (< 5 tasks) | 11/30 |
| LoCUS-2M, 3% | splice | +0.001 | n/a (< 5 tasks) | 16/30 |
| LoCUS-2M, 3% | enhancer | -0.001 | n/a (< 5 tasks) | 8/20 |
| LoCUS-2M, 3% | all 18 | +0.000 | [-0.002, +0.003] | 97/180 |
| LoCUS-2M, PBSIM3 | splice | -0.002 | n/a (< 5 tasks) | 14/30 |
| LoCUS-2M, PBSIM3 | all 18 | +0.001 | [-0.001, +0.003] | 91/180 |
| LoCUS-2M, 3% subst. only | splice | -0.001 | n/a (< 5 tasks) | 16/30 |
| LoCUS-2M, 3% subst. only | all 18 | +0.001 | [-0.002, +0.004] | 94/180 |
| LoCUS-2M, 1% subst. only | splice | -0.000 | n/a (< 5 tasks) | 17/30 |
| LoCUS-2M, 1% subst. only | all 18 | -0.000 | [-0.002, +0.002] | 91/180 |
| LoCUS-2M, sup reads | splice | -0.001 | n/a (< 5 tasks) | 14/30 |
| LoCUS-2M, sup reads | all 18 | -0.000 | [-0.001, +0.001] | 86/180 |
| LoCUS-2M, hac reads | splice | -0.000 | n/a (< 5 tasks) | 14/30 |
| LoCUS-2M, hac reads | all 18 | -0.000 | [-0.002, +0.001] | 85/180 |
| fixed-segment twin, 1% | histone | +0.000 | [-0.001, +0.001] | 55/100 |
| fixed-segment twin, 1% | promoter | +0.003 | n/a (< 5 tasks) | 16/30 |
| fixed-segment twin, 1% | splice | +0.002 | n/a (< 5 tasks) | 14/30 |
| fixed-segment twin, 1% | enhancer | -0.001 | n/a (< 5 tasks) | 10/20 |
| fixed-segment twin, 1% | all 18 | +0.001 | [-0.001, +0.003] | 95/180 |
| fixed-segment twin, 3% | histone | -0.000 | [-0.002, +0.001] | 49/100 |
| fixed-segment twin, 3% | promoter | +0.004 | n/a (< 5 tasks) | 18/30 |
| fixed-segment twin, 3% | splice | -0.004 | n/a (< 5 tasks) | 15/30 |
| fixed-segment twin, 3% | enhancer | -0.002 | n/a (< 5 tasks) | 10/20 |
| fixed-segment twin, 3% | all 18 | -0.000 | [-0.003, +0.002] | 92/180 |
| fixed-segment twin, PBSIM3 | splice | +0.005 | n/a (< 5 tasks) | 17/30 |
| fixed-segment twin, PBSIM3 | all 18 | +0.001 | [-0.002, +0.003] | 94/180 |
| fixed-segment twin, 3% subst. only | splice | +0.002 | n/a (< 5 tasks) | 17/30 |
| fixed-segment twin, 3% subst. only | all 18 | -0.000 | [-0.002, +0.002] | 95/180 |
| fixed-segment twin, 1% subst. only | splice | +0.001 | n/a (< 5 tasks) | 19/30 |
| fixed-segment twin, 1% subst. only | all 18 | -0.000 | [-0.001, +0.001] | 89/180 |
| fixed-segment twin, sup reads | splice | +0.006 | n/a (< 5 tasks) | 16/30 |
| fixed-segment twin, sup reads | all 18 | +0.002 | [-0.001, +0.005] | 97/180 |
| fixed-segment twin, hac reads | splice | +0.003 | n/a (< 5 tasks) | 18/30 |
| fixed-segment twin, hac reads | all 18 | +0.001 | [-0.001, +0.004] | 97/180 |
| BPE twin, 1% | histone | -0.002 | [-0.003, -0.001] | 38/100 |
| BPE twin, 1% | promoter | -0.001 | n/a (< 5 tasks) | 11/30 |
| BPE twin, 1% | splice | +0.005 | n/a (< 5 tasks) | 15/30 |
| BPE twin, 1% | enhancer | -0.001 | n/a (< 5 tasks) | 9/20 |
| BPE twin, 1% | all 18 | -0.001 | [-0.003, +0.002] | 73/180 |
| BPE twin, 3% | histone | -0.002 | [-0.007, +0.001] | 34/100 |
| BPE twin, 3% | promoter | -0.003 | n/a (< 5 tasks) | 10/30 |
| BPE twin, 3% | splice | -0.013 | n/a (< 5 tasks) | 11/30 |
| BPE twin, 3% | enhancer | -0.002 | n/a (< 5 tasks) | 7/20 |
| BPE twin, 3% | all 18 | -0.004 | [-0.007, -0.001] | 62/180 |
| BPE twin, PBSIM3 | splice | -0.013 | n/a (< 5 tasks) | 12/30 |
| BPE twin, PBSIM3 | all 18 | -0.003 | [-0.006, -0.001] | 67/180 |
| BPE twin, 3% subst. only | splice | +0.002 | n/a (< 5 tasks) | 19/30 |
| BPE twin, 3% subst. only | all 18 | -0.002 | [-0.004, +0.000] | 72/180 |
| BPE twin, 1% subst. only | splice | +0.000 | n/a (< 5 tasks) | 18/30 |
| BPE twin, 1% subst. only | all 18 | -0.001 | [-0.003, -0.000] | 77/180 |
| BPE twin, sup reads | splice | -0.000 | n/a (< 5 tasks) | 14/30 |
| BPE twin, sup reads | all 18 | -0.001 | [-0.002, +0.001] | 84/180 |
| BPE twin, hac reads | splice | -0.010 | n/a (< 5 tasks) | 11/30 |
| BPE twin, hac reads | all 18 | -0.003 | [-0.005, -0.001] | 77/180 |

Drop = score on the set - score on the clean test set (real reads: on the reference sequences of the same windows), each set the mean over its 10 draws, as in the tokenizer study; > 0: the model without chromosomes 20 and 21 loses less. Every NT18 group for the 1% and 3% sets, all 18 tasks and the splice sites for the others.
