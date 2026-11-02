"""Collect the results into the paper tables (Markdown, plus every number as JSON).

    python scripts/collect_results.py --results results --output results/tables

Expected layout (written by scripts/reproduce.py, or by the scripts it runs):
    results/<model>/nt18/<task>/fold-XX.json                 NT18 revised, test MCC (LoCUS models, locus-2m-random-init, baselines)
    results/caduceus-ps-<size>-authors[-bf16]/nt18/<task>/fold-XX.json   Caduceus-PS with its own settings (bf16: overflowed folds)
    results/<model>-no-chr20-21/nt18/<task>/fold-XX.json (+ fold-XX.scored.json)   pretraining-split ablation
    results/<model>/genomic_benchmarks/<task>/fold-XX.json   Genomic Benchmarks, test accuracy
    results/tokenizer_study/<lcp|fixed|bpe>/nt18/<task>/fold-XX.json (+ fold-XX.scored.json)
    results/tokenizer_study/<lcp|fixed|bpe>/gue/gue_splice/fold-XX.json (+ fold-XX.scored.json)   GUE splice
    results/gue_independent/<lcp|fixed|bpe>/gue/gue_splice/fold-XX.scored.json   GUE splice, test rows independent of NT18
    results/real_reads/<lcp|fixed|bpe>/nt18/<task>/fold-XX.scored.json            real nanopore reads of the test windows
    results/single_nucleotide/<model>/nt18/<task>/fold-XX.scored.json   single-nucleotide baselines on the modified test sets
    results/real_reads/<model>/nt18/<task>/fold-XX.scored.json          single-nucleotide baselines and split models on the real reads
    results/intervals/<nt18|real_reads|gue>.json             intervals from resampling test windows
    results/<locus-2m|locus-2m-random-init>/nt18_heldout/<task>/fold-XX.json   NT18 tested on chromosomes outside pretraining
    results/long_range/frozen/fit_<model>_lrb_<task>_4k.json             frozen-embedding long-range tasks
    results/parsing/locality.json                            parse-level edit locality and context independence
    results/parsing/lcp_statistics.json                      LCP segmentation statistics and worst-case edit locality
    results/parsing/boundary_information.json                masked bases recovered from the segment boundaries (pretrained models)
    results/embedding/stability.json                         embedding stability after one edit (frozen 2M twins)
    results/bend/results.json                                BEND zero-shot variant effects
    results/efficiency/finetune_speed.json                   fine-tuning throughput
    results/efficiency/segmentation_time.json                segmentation time of the NT18 datasets (dataset creation)
    results/timing/<model>/nt18/<task>/fold-00.json          full fine-tuning wall clock
NT18 results of the 2M twins (LoCUS-2M, twin-fixed-2m, twin-bpe-2m) are the tokenizer study's runs (clean test set) when
results/<model>/nt18 does not exist. Every fine-tuning
score is the test metric at the epoch with the lowest validation loss, recorded during fine-tuning; cells are mean +-
standard deviation over the folds (--nt18-folds, default 10, for NT18, the tokenizer study, GUE splice and the real reads;
--other-folds, default 10, for the Genomic Benchmarks and the held-out check); group means are means of the task means and appear only when every task of the group has all folds. A model and
task with fewer folds than expected are listed as warnings at the end of the run. In the tokenizer study, the score of a fold on a
noisy test set is the mean over the set's error draws in fold-XX.scored.json (<set>, <set>-d1, ...;
benchmarks/noisy_reads/noisy_sets.py); the tables state how many."""
import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Optional

MODELS = ["locus-0.5m", "locus-2m"]
BASELINES = ["caduceus-ps-1.9m"]
# Caduceus-PS at the size of LoCUS-0.5M with the shared protocol, on NT18 only (not scored on the modified test sets)
NT18_ONLY_BASELINES = ["caduceus-ps-0.47m"]
TWINS = {"locus-2m": "lcp", "twin-fixed-2m": "fixed", "twin-bpe-2m": "bpe"}
# Caduceus-PS 0.47M and 1.9M fine-tuned with Caduceus's own settings (scripts/reproduce.py plan baselines-authors; supplementary
# table), next to LoCUS and Caduceus-PS 1.9M with the shared protocol
AUTHORS_BASELINES = ["caduceus-ps-0.47m-authors", "caduceus-ps-1.9m-authors"]
# folds whose fp16 training overflowed are run again in bfloat16 (scripts/reproduce.py plan baselines-authors-bf16)
AUTHORS_BF16 = {run: f"{run}-bf16" for run in AUTHORS_BASELINES}
# timing only: the Caduceus-PS recipes in bfloat16, the precision of LoCUS (scripts/reproduce.py plan timing)
TIMING_BF16 = ["caduceus-ps-0.47m-bf16", "caduceus-ps-1.9m-bf16"]
FROZEN_LABELS = {"locus-0.5m": "locus-0.5m", "locus-2m": "locus-2m", "tss": "no sequence: tissue + TSS distance"}
BEND_ORDER = ["locus-0.5m", "locus-2m"]
# the work each model's fine-tuning data needs before training (benchmarks/efficiency/segmentation_time.py)
SEGMENTER_OF = {"locus-0.5m": "lcp", "locus-2m": "lcp", "twin-fixed-2m": "fixed", "twin-bpe-2m": "bpe", "LoCUS-0.5M": "lcp", "LoCUS-2M": "lcp"}
NT18_GROUPS = [("histone", ["H3K4me3", "H3K27ac", "H3K36me3", "H3K9me3", "H2AFZ", "H3K27me3", "H3K4me1", "H3K4me2", "H3K9ac", "H4K20me1"]),
               ("promoter", ["promoter_all", "promoter_no_tata", "promoter_tata"]),
               ("splice", ["splice_sites_all", "splice_sites_acceptors", "splice_sites_donors"]),
               ("enhancer", ["enhancers", "enhancers_types"])]
GB_TASKS = ["demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "dummy_mouse_enhancers_ensembl", "human_enhancers_cohn",
            "human_enhancers_ensembl", "human_ensembl_regulatory", "human_nontata_promoters", "human_ocr_ensembl", "drosophila_enhancers_stark"]
TOKENIZERS = [("LCP", "lcp"), ("BPE", "bpe"), ("fixed", "fixed")]
# modified test sets of the paired drop tests: (label, set name); substitutions only = an independent draw of the error
# schedule (benchmarks/noisy_reads/make_controls.py), every event one substitution
STUDY_SETS = [("1%", "ont1"), ("3%", "ont3"), ("PBSIM3", "pbsim97"), ("3% subst. only", "sub3"), ("1% subst. only", "sub1")]
FOLDS = 10  # default fold count of a cell
OTHER_FOLDS = 10  # Genomic Benchmarks and the held-out-chromosome check (--other-folds)
NT18_FOLDS = 10  # NT18, the tokenizer study, GUE splice and the real reads (--nt18-folds)
SPLICE = ["splice_sites_acceptors", "splice_sites_all", "splice_sites_donors"]
# the tasks with real reads (benchmarks/noisy_reads/real_reads.py), by NT18 group
REAL_GROUPS = NT18_GROUPS
REAL_DRAWS = 10  # reads per window (benchmarks/noisy_reads/real_reads.py); a task-fold counts only with all of them scored
NOISY_DRAWS = 10  # error draws per noisy set (benchmarks/noisy_reads/noisy_sets.py), required in the single-nucleotide comparison
# single-nucleotide models (scripts/reproduce.py plan single-nucleotide): no nucleotides merged into tokens
SINGLE_NUCLEOTIDE = [("Caduceus", "caduceus-ps-1.9m")]  # (label, run); the label also seeds the model's bootstrap intervals
# pretraining-split ablation (scripts/reproduce.py split-nt18, split-score): the model pretrained without chromosomes 20 and 21,
# its label, and the paper runs of its released counterpart (relative to the results directory); the 2M ones are also scored
SPLIT = [("locus-0.5m-no-chr20-21", "LoCUS-0.5M", "locus-0.5m/nt18"), ("locus-2m-no-chr20-21", "LoCUS-2M", "tokenizer_study/lcp/nt18"),
         ("twin-fixed-2m-no-chr20-21", "fixed-segment twin", "tokenizer_study/fixed/nt18"), ("twin-bpe-2m-no-chr20-21", "BPE twin", "tokenizer_study/bpe/nt18")]
SPLIT_REAL_READS = {"locus-2m-no-chr20-21": "lcp", "twin-fixed-2m-no-chr20-21": "fixed", "twin-bpe-2m-no-chr20-21": "bpe"}  # released twin's real-read folder
WARNINGS = []  # model / task cells with fewer folds than expected, printed at the end


def selected_entry(report: dict) -> dict:
    """History entry of the selected epoch (lowest finite validation loss, as training selects it; ties: higher validation
    MCC). Epochs whose validation loss is not finite (fp16 overflow) are never selected, unless no epoch has a finite one."""
    history = report["history"]
    finite = [i for i in range(len(history)) if math.isfinite(history[i]["validation_loss"])] or list(range(len(history)))
    return history[min(finite, key=lambda i: (history[i]["validation_loss"], -history[i]["validation_metrics"]["mcc"]))]


def fold_scores(directory: Path, metric: str, folds: int = FOLDS) -> list[float]:
    """Test metric of each finished fold: the selected epoch's recorded test score, or the final test score of the selected
    checkpoint when per-epoch test scoring was switched off."""
    scores = []
    for fold in range(folds):
        path = directory / f"fold-{fold:02d}.json"
        if path.is_file():
            report = json.loads(path.read_text())
            scores.append(selected_entry(report).get("test_metrics", report["test_metrics"])[metric])
    return scores


def last_epoch_selected(directory: Path, folds: int = FOLDS) -> tuple[int, int]:
    """(finished folds whose selected epoch is their last epoch, finished folds)."""
    reports = [json.loads(path.read_text()) for path in (directory / f"fold-{fold:02d}.json" for fold in range(folds)) if path.is_file()]
    return sum(selected_entry(report) is report["history"][-1] for report in reports), len(reports)


def study_folds(directory: Path, folds: int = FOLDS) -> dict[int, dict]:
    """{fold: {set name: test MCC}} for the tokenizer study: "clean", the sets scored during fine-tuning (extra test sets
    of the selected epoch) and the sets scored afterwards from the saved weights. A noisy set's value is the mean over its
    error draws (<set>, <set>-d1, ...), and "<set> draws" is their number."""
    out = {}
    for fold in range(folds):
        path = directory / f"fold-{fold:02d}.json"
        if not path.is_file():
            continue
        report = json.loads(path.read_text())
        entry = selected_entry(report)
        row = {"clean": entry["test_metrics"]["mcc"]}
        row.update({name: metrics["mcc"] for name, metrics in entry.get("extra_test_metrics", {}).items()})
        scored = directory / f"fold-{fold:02d}.scored.json"
        if scored.is_file():
            later = json.loads(scored.read_text())
            if later["best_step"] != report["best_step"]:
                raise ValueError(f"{scored} was scored from a different checkpoint")
            row.update({name: metrics["mcc"] for name, metrics in later["sets"].items() if name not in row})
        draws = {}
        for name, value in row.items():
            base, separator, draw = name.rpartition("-d")
            draws.setdefault(base if separator and draw.isdigit() else name, []).append(value)
        row = {name: statistics.mean(values) for name, values in draws.items()}
        row.update({f"{name} draws": len(values) for name, values in draws.items() if name != "clean"})
        out[fold] = row
    return out


def cell(values: list, folds: int = FOLDS) -> dict:
    complete = len(values) == folds
    return {"values": values, "mean": statistics.mean(values) if values else None,
            "sd": statistics.stdev(values) if complete else None, "complete": complete, "folds": folds}


def text(c) -> str:
    if c is None or not c["values"]:
        return "-"
    if c["complete"]:
        return f"{c['mean']:.3f} ± {c['sd']:.3f}"
    return f"{c['mean']:.3f} ({len(c['values'])}/{c.get('folds', FOLDS)})"


def group_mean(cells: list):
    return statistics.mean(c["mean"] for c in cells) if cells and all(c and c["complete"] for c in cells) else None


def markdown(title: str, header: list, rows: list, notes: str = "") -> str:
    lines = [f"## {title}", "", "| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(lines) + ("\n\n" + notes if notes else "") + "\n"


def number(value, signed=False) -> str:
    return "-" if value is None else (f"{value:+.3f}" if signed else f"{value:.3f}")


def nt18_directory(results: Path, model: str) -> Path:
    own = results / model / "nt18"
    return results / "tokenizer_study" / TWINS[model] / "nt18" if model in TWINS and not own.is_dir() else own


def nt18_table(results: Path, numbers: dict) -> str:
    models = [m for m in MODELS + ["twin-fixed-2m", "twin-bpe-2m", "locus-2m-random-init"] + NT18_ONLY_BASELINES + BASELINES if nt18_directory(results, m).is_dir()]
    rows, cells = [], {}
    for group, tasks in NT18_GROUPS:
        for task in tasks:
            cells[task] = {m: cell(fold_scores(nt18_directory(results, m) / f"{task}_revised", "mcc", NT18_FOLDS), NT18_FOLDS) for m in models}
            WARNINGS.extend(f"NT18 {m} {task}: {len(c['values'])} of {NT18_FOLDS} folds" for m, c in cells[task].items() if not c["complete"])
            rows.append([group, task] + [text(cells[task][m]) for m in models])
    for group, tasks in NT18_GROUPS + [("all 18", [t for _, ts in NT18_GROUPS for t in ts])]:
        rows.append(["**mean**", group] + [number(group_mean([cells[t][m] for t in tasks])) for m in models])
    numbers["nt18"] = cells
    notes = ("twin-fixed-2m and twin-bpe-2m: LoCUS-2M with fixed 10 bp segments every 5 bp or BPE tokens (tokenizer study), same recipe and seeds. "
             if "twin-fixed-2m" in models or "twin-bpe-2m" in models else "")
    notes += ("locus-2m-random-init: the LoCUS-2M architecture fine-tuned from random initialization (no pretraining), same recipe and seeds. "
              if "locus-2m-random-init" in models else "")
    if "locus-2m-random-init" in models:  # runs still improving at their last epoch may not have converged in the 20 epochs
        last = {task: last_epoch_selected(nt18_directory(results, "locus-2m-random-init") / f"{task}_revised", NT18_FOLDS)
                for _, tasks in NT18_GROUPS for task in tasks}
        splice = dict(NT18_GROUPS)["splice"]
        count = lambda tasks: (sum(last[t][0] for t in tasks), sum(last[t][1] for t in tasks))
        numbers["nt18_random_init_last_epoch"] = {"all 18": count(list(last)), "splice": count(splice), "per task": last}
        notes += "Its selected epoch is the last one in {} of {} runs ({} of {} on the splice tasks). ".format(*count(list(last)), *count(splice))
    notes += ("Baselines (caduceus-ps-0.47m, caduceus-ps-1.9m): fine-tuned with the same protocol (scripts/finetune_baseline.py)."
              if set(models) & set(BASELINES + NT18_ONLY_BASELINES) else "")
    return markdown("Nucleotide Transformer benchmark, revised (test MCC)", ["group", "task"] + models, rows, notes)


def authors_scores(results: Path, model: str, task: str) -> tuple[list[float], int]:
    """(test MCC of each finished fold, number of bf16 reruns used) of an own-settings run: a fold that was run again in
    bfloat16 after its fp16 training overflowed counts with the bfloat16 run."""
    own = nt18_directory(results, model) / f"{task}_revised"
    rerun = results / AUTHORS_BF16[model] / "nt18" / f"{task}_revised" if model in AUTHORS_BF16 else None
    scores, reruns = [], 0
    for fold in range(NT18_FOLDS):
        path = own / f"fold-{fold:02d}.json"
        if rerun is not None and (rerun / path.name).is_file():
            path, reruns = rerun / path.name, reruns + 1
        if path.is_file():
            report = json.loads(path.read_text())
            scores.append(selected_entry(report).get("test_metrics", report["test_metrics"])["mcc"])
    return scores, reruns


def authors_table(results: Path, numbers: dict) -> str:
    """NT18: Caduceus-PS fine-tuned with Caduceus's own settings next to LoCUS and the shared-protocol Caduceus-PS runs."""
    if not any((results / m / "nt18").is_dir() for m in AUTHORS_BASELINES):
        return ""
    models = [m for m in MODELS + NT18_ONLY_BASELINES + BASELINES + AUTHORS_BASELINES if nt18_directory(results, m).is_dir()]
    rows, cells, reruns = [], {}, 0
    for group, tasks in NT18_GROUPS:
        for task in tasks:
            cells[task] = {}
            for m in models:
                if m in AUTHORS_BF16:
                    scores, used = authors_scores(results, m, task)
                    reruns += used
                else:
                    scores = fold_scores(nt18_directory(results, m) / f"{task}_revised", "mcc", NT18_FOLDS)
                cells[task][m] = cell(scores, NT18_FOLDS)
            WARNINGS.extend(f"NT18 {m} {task}: {len(c['values'])} of {NT18_FOLDS} folds"
                            for m, c in cells[task].items() if m in AUTHORS_BASELINES and not c["complete"])
            rows.append([group, task] + [text(cells[task][m]) for m in models])
    for group, tasks in NT18_GROUPS + [("all 18", [t for _, ts in NT18_GROUPS for t in ts])]:
        rows.append(["**mean**", group] + [number(group_mean([cells[t][m] for t in tasks])) for m in models])
    numbers["nt18_authors"] = cells
    numbers["nt18_authors_bf16_reruns"] = reruns
    notes = ("caduceus-ps-<size>-authors: Caduceus-PS fine-tuned with Caduceus's own settings for the NT tasks "
             "(configs/finetune/caduceus-ps-<size>-authors.yaml: per task the learning rate and batch size of its paper's Table 6, "
             "its weight-decay groups, no dropout, the forward strand in training and evaluation); caduceus-ps-1.9m: the protocol shared with "
             "LoCUS. The folds, seeds and epoch selection (lowest validation loss) are the same for every column. "
             f"{reruns} folds whose fp16 training overflowed (non-finite steps or validation losses) are the bfloat16 reruns "
             "(caduceus-ps-<size>-authors-bf16.yaml).")
    return markdown("Nucleotide Transformer benchmark, Caduceus-PS with its own fine-tuning settings (test MCC)",
                    ["group", "task"] + models, rows, notes)


def heldout_table(results: Path, numbers: dict) -> str:
    """LoCUS-2M (and its random-initialization control) fine-tuned without the training examples on chromosomes outside the
    pretraining chromosomes and tested on them (benchmarks/nt18/heldout_split.py), next to the official test set scored at the
    same selected epoch."""
    models = [(label, results / name / "nt18_heldout") for label, name in (("LoCUS-2M", "locus-2m"), ("random init", "locus-2m-random-init"))]
    models = [(label, root) for label, root in models if root.is_dir()]
    if not models:
        return ""
    rows, cells = [], {}
    for group, tasks in NT18_GROUPS:
        for task in tasks:
            cells[task] = {}
            for label, root in models:
                heldout, official = [], []
                for fold in range(OTHER_FOLDS):
                    path = root / f"{task}_revised" / f"fold-{fold:02d}.json"
                    if path.is_file():
                        report = json.loads(path.read_text())
                        entry = selected_entry(report)
                        heldout.append(entry.get("test_metrics", report["test_metrics"])["mcc"])
                        official.append(entry["extra_test_metrics"]["official"]["mcc"])
                cells[task].update({f"{label}: held-out chromosomes": cell(heldout, OTHER_FOLDS), f"{label}: official test": cell(official, OTHER_FOLDS),
                                    f"{label}: held-out - official": cell([h - o for h, o in zip(heldout, official)], OTHER_FOLDS)})
            rows.append([group, task] + [text(c) for c in cells[task].values()])
    for group, tasks in NT18_GROUPS + [("all 18", [t for _, ts in NT18_GROUPS for t in ts])]:
        rows.append(["**mean**", group] + [number(group_mean([cells[t][k] for t in tasks]), k.endswith("official")) for k in cells[tasks[0]]])
    numbers["nt18_heldout"] = cells
    notes = ("Fine-tuned on the official training set without its examples on chromosomes outside the pretraining chromosomes (3-22), "
             "tested on those examples (seen neither in pretraining nor in fine-tuning); the official test set (chromosomes 20 and 21, "
             "part of the pretraining chromosomes) is scored at the same selected epoch. Random init: the same architecture without "
             "pretrained weights, so neither test set was seen before fine-tuning; its held-out - official gap measures how much harder "
             f"the held-out chromosomes are. {OTHER_FOLDS} folds.")
    table = markdown("NT18 on chromosomes outside pretraining (test MCC)", ["group", "task"] + list(cells[NT18_GROUPS[0][1][0]]), rows, notes)
    if len(models) < 2:
        return table
    # pretraining gain (LoCUS-2M - random init) on each test set; if seeing the official test chromosomes in pretraining helped,
    # the gain would be larger on the official test set than on the held-out chromosomes
    keys = {"held-out": ("LoCUS-2M: held-out chromosomes", "random init: held-out chromosomes"),
            "official": ("LoCUS-2M: official test", "random init: official test")}
    rows, gains = [], {}
    for scope, tasks in [("all 18", [t for _, ts in NT18_GROUPS for t in ts])] + NT18_GROUPS:
        if not all(cells[t][k]["complete"] for t in tasks for pair in keys.values() for k in pair):
            continue
        per_task = {which: [cells[t][a]["mean"] - cells[t][b]["mean"] for t in tasks] for which, (a, b) in keys.items()}
        difference = [o - h for h, o in zip(per_task["held-out"], per_task["official"])]
        rng = random.Random(0 if scope == "all 18" else f"heldout {scope}")
        boot = sorted(statistics.mean([difference[rng.randrange(len(difference))] for _ in difference]) for _ in range(4000))
        gains[scope] = {"gain held-out": statistics.mean(per_task["held-out"]), "gain official": statistics.mean(per_task["official"]),
                        "official - held-out": statistics.mean(difference), "official - held-out interval": [boot[100], boot[3899]],
                        "tasks with official - held-out > 0": f"{sum(d > 0 for d in difference)}/{len(tasks)}"}
        g = gains[scope]
        rows.append([scope, len(tasks), number(g["gain held-out"], True), number(g["gain official"], True), number(g["official - held-out"], True),
                     f"[{boot[100]:+.3f}, {boot[3899]:+.3f}]", g["tasks with official - held-out > 0"]])
    numbers["nt18_heldout_gain"] = gains
    return table + "\n" + markdown(
        "NT18 on chromosomes outside pretraining: pretraining gain (LoCUS-2M - random init, test MCC)",
        ["tasks", "number", "gain, held-out chromosomes", "gain, official test", "official - held-out", "95% CI", "tasks with official - held-out > 0"], rows,
        "Gain per task: mean over the folds of LoCUS-2M minus that of random init, on the same test set; > 0 in the official - held-out "
        "column would mean that seeing the official test chromosomes in pretraining helps. Interval: bootstrap over tasks (4,000 resamples).")


def gb_table(results: Path, numbers: dict) -> str:
    models = [m for m in MODELS if (results / m / "genomic_benchmarks").is_dir()]
    cells = {task: {m: cell(fold_scores(results / m / "genomic_benchmarks" / f"gb_{task}", "accuracy", OTHER_FOLDS), OTHER_FOLDS) for m in models}
             for task in GB_TASKS}
    for task in GB_TASKS:
        WARNINGS.extend(f"Genomic Benchmarks {m} {task}: {len(c['values'])} of {OTHER_FOLDS} folds" for m, c in cells[task].items() if not c["complete"])
    rows = [[task] + [text(cells[task][m]) for m in models] for task in GB_TASKS]
    rows.append(["**mean, 8 tasks (no drosophila)**"] + [number(group_mean([cells[t][m] for t in GB_TASKS[:-1]])) for m in models])
    rows.append(["**mean, all 9**"] + [number(group_mean([cells[t][m] for t in GB_TASKS])) for m in models])
    numbers["genomic_benchmarks"] = cells
    return markdown("Genomic Benchmarks (test accuracy)", ["task"] + models, rows, f"Mean ± SD over {OTHER_FOLDS} folds.")


def frozen_fit(results: Path, model: str, task: str, length: str) -> dict:
    path = results / "long_range/frozen" / f"fit_{model}_lrb_{task}_{length}.json"
    return json.loads(path.read_text())["mean"] if path.is_file() else {}


def frozen_table(results: Path, numbers: dict) -> str:
    if not (results / "long_range/frozen").is_dir():
        return ""
    rows, values = [], {}
    for model, label in FROZEN_LABELS.items():
        eqtl, rna = frozen_fit(results, model, "eqtl", "4k"), frozen_fit(results, model, "bulk_rna", "4k")
        if eqtl or rna:
            values[model] = {"eqtl": eqtl, "bulk_rna": rna}
            rows.append([label, number(eqtl.get("auroc")), number(eqtl.get("accuracy")), number(rna.get("spearman_genes")), number(rna.get("r2"))])
    numbers["long_range_frozen"] = values
    return markdown("Long-range benchmark, frozen embeddings, 4 kb input: causal eQTL and bulk RNA expression",
                    ["model", "eQTL AUROC", "eQTL accuracy", "bulk RNA Spearman (genes)", "bulk RNA R2"], rows,
                    "Means over the 5 chromosome folds (benchmarks/long_range/frozen.py).")


def parsing_tables(results: Path, numbers: dict) -> str:
    path = results / "parsing/locality.json"
    if not path.is_file():
        return ""
    parsed = json.loads(path.read_text())
    numbers["parsing"] = parsed
    rows = []
    for name, kinds in parsed["edit_locality"].items():
        for kind, r in kinds.items():
            rows.append([name, kind, f"{r['changed_segments_mean']:.1f}", f"{r['reach_median_bp']:.0f}", f"{r['reach_95th_bp']:.0f}", f"{r['reach_max_bp']:.0f}"])
    edits = markdown("Parsing: segments changed by one edit (4 kb chromosome-1 windows, edit at least 1 kb from both ends)",
                     ["segmenter", "edit", "changed segments (mean)", "reach, median (bp)", "reach, 95th percentile (bp)", "reach, max (bp)"], rows,
                     f"{next(iter(next(iter(parsed['edit_locality'].values())).values()))['edits']:,} edits of each kind; reach = largest distance "
                     "from the edit to a base of a changed segment. Segments: " + "; ".join(
                         f"{n} {r['mean_segment_length_bp']:.1f} bp on average, {r['segments_per_kb']:.0f} per kb" for n, r in parsed["segments"].items()) + ".")
    bins = list(next(iter(parsed["context_independence"].values()))["share_consistent_by_distance_bp"])
    rows = [[name, f"{r['share_consistent']:.3f}"] + [number(r["share_consistent_by_distance_bp"][b]) for b in bins]
            for name, r in parsed["context_independence"].items()]
    context = markdown("Parsing: the same 300 bp stretch in two different contexts (share of its segments parsed identically)",
                       ["segmenter", "all segments"] + [f"{b} bp from the stretch end" for b in bins], rows,
                       "Flanks of 496-500 bp from other chromosome-1 windows; a segment counts when it lies inside the stretch.")
    return edits + "\n" + context


def lcp_statistics_tables(results: Path, numbers: dict) -> str:
    path = results / "parsing/lcp_statistics.json"
    if not path.is_file():
        return ""
    stats = json.loads(path.read_text())
    numbers["lcp_statistics"] = stats
    seg = stats["segmentation"]
    rows = []
    for level in ("level 1", "level 2"):
        r = seg[level]
        rows.append([level + (" (segments)" if level == "level 2" else ""), f"{r['cores_per_kb']:.0f}", f"{r['mean_start_spacing_bp']:.2f}",
                     f"{r['bases_per_core']:.2f}", f"{r['mean_length_bp']:.2f}", f"{r['max_length_bp']}",
                     ", ".join(f"{kind} {v['share']:.1%}" for kind, v in r["by_type"].items())])
    level1, level2 = seg["level 1"], seg["level 2"]
    covered = level2["share_of_bases_covered_by"]
    table = markdown(f"LCP segmentation statistics ({seg['windows']:,} windows of 4,096 bp from the pretraining chromosomes)",
                     ["cores", "per kb", "mean start spacing (bp)", "bases per core", "mean length (bp)", "max length (bp)", "core types"], rows,
                     f"Level-1 cores other than runs with 3 bases: {level1['share_of_non_run_cores_with_3_bases']:.1%}. Bases covered by 0 / 1 / 2 / 3 / "
                     f"4 or more segments: " + " / ".join(f"{v:.1%}" for v in covered.values()) +
                     f" ({level2['share_of_bases_in_2_or_more']:.1%} in two or more). Bases at the window start / end covered by no segment: "
                     f"{level2['uncovered_bases_at_start_mean']:.1f} / {level2['uncovered_bases_at_end_mean']:.1f} on average" +
                     (f" (by no level-1 core: {level1['uncovered_bases_at_start_mean']:.1f} / {level1['uncovered_bases_at_end_mean']:.1f})"
                      if "uncovered_bases_at_start_mean" in level1 else "") + ".")
    worst = stats["worst_case"]
    rows = []
    for name, levels in worst["sets"].items():
        for level, r in levels.items():
            w = r["without_homopolymer_6_bp_nearby"]
            rows.append([name, level, f"{r['edits']:,}", f"{r['changed_mean']:.2f}", r["changed_max"], f"{r['reach_median_bp']:.0f}",
                         f"{r['reach_99th_bp']:.0f}", f"{r['reach_max_bp']} ({r['longest_homopolymer_near_the_largest_reach_bp']} bp homopolymer nearby)",
                         w["changed_max"], w["reach_max_bp"]])
    return table + "\n" + markdown(
        f"LCP worst-case edit locality: one edit at the centre of {worst['sequences_per_set']:,} sequences of {worst['length']:,} bp per set",
        ["sequences", "cores", "edits", "changed cores (mean)", "changed cores (max)", "reach, median (bp)", "reach, 99th percentile (bp)", "reach, max (bp)",
         "changed cores (max), no homopolymer of 6+ bp within 40 bp", "reach, max (bp), no homopolymer of 6+ bp within 40 bp"], rows,
        "Edits: a substitution, or an insertion or deletion of 1, 2 or 3 bases, each on every sequence. A core is changed when the edited parse "
        "has no core with the same bases at the same, shifted position; reach = largest distance from the edit to a base of a changed core.")


def boundary_table(results: Path, numbers: dict) -> str:
    path = results / "parsing/boundary_information.json"
    if not path.is_file():
        return ""
    stats = json.loads(path.read_text())
    numbers["boundary_information"] = stats
    columns = [("mask", "hidden by MASK"), ("overlap_conflict", "overlap symbol"), ("random", "random bases"), ("unchanged", "left unchanged"),
               ("all", "all targets")]
    rows = [[name, set_name] + [f"{r[key]['accuracy']:.3f}" if r[key]["accuracy"] is not None else "-" for key, _ in columns]
            + [f"{r['mask']['targets']:,}"]
            for name, sets in stats["models"].items() for set_name, r in sets.items()]
    return markdown(f"Segment boundaries and masked bases: accuracy of the pretrained models on the bases they have to predict ({stats['sequences_per_set']:,} "
                    "sequences per set)", ["model", "sequences", *[label for _, label in columns], "bases hidden by MASK"], rows,
                    "Crops, segmentation and masking as in pretraining (benchmarks/parsing/boundary_information.py): the clean crop is segmented and "
                    "15% of its segments are hidden (MASK 80%, random bases 10%, unchanged 10%; the overlap symbol where two hidden segments disagree). "
                    "random: uniform random sequences, whose context carries no information, so accuracy above chance (25%) on bases hidden by MASK "
                    "comes from the segment boundaries; chromosome 1: windows held out of pretraining. twin-fixed-2m (boundaries independent of the bases) "
                    "is the control.")


def stability_table(results: Path, numbers: dict) -> str:
    path = results / "embedding/stability.json"
    if not path.is_file():
        return ""
    stab = json.loads(path.read_text())
    numbers["embedding_stability"] = {key: stab[key] for key in ("sequences", "segments", "cos0", "cos0_read", "paired")}
    numbers["embedding_stability"]["curves"] = {m: {k: {x: v for x, v in c.items() if x != "normalized_change_by_distance"} for k, c in kinds.items()}
                                                for m, kinds in stab["curves"].items()}
    order = ["LCP", "BPE", "fixed"]
    rows = []
    for kind in stab["curves"]["LCP"]:
        seg = [stab["segments"][m][kind]["changed_segments_per_strand_mean"] for m in order]
        curve = [stab["curves"][m][kind] for m in order]
        rows.append([kind] + [f"{x:.2f}" for x in seg] + [f"{c['within_10bp']:.3f}" for c in curve] +
                    ["never" if c["stays_at_or_below_0.01_from_bp"] is None else str(c["stays_at_or_below_0.01_from_bp"]) for c in curve] +
                    [f"{c['far_field']:.3f}" for c in curve] + [f"{1e4 * c['read_change_mean']:.2f}" for c in curve])
    header = (["edit"] + [f"changed segments per strand, {m}" for m in order] + [f"per-base change within 10 bp, {m}" for m in order] +
              [f"per-base change <= 0.01 from (bp), {m}" for m in order] + [f"per-base change 500-900 bp away, {m}" for m in order] +
              [f"read change x 1e-4, {m}" for m in order])
    paired = [[kind] + [f"{v['ratio']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]" for v in stab["paired"][kind].values()] for kind in stab["paired"]]
    return (markdown(f"Embedding stability: one edit at the centre of {stab['sequences']:,} chromosome-1 sequences of {stab['length']:,} bp (frozen 2M twins)",
                     header, rows,
                     "Changed segments: no model. Per-base change: (1 - cos(u, u')) / (1 - c0) per base, c0 = mean cosine of unrelated bases of the "
                     f"model (LCP {stab['cos0']['LCP']:.3f}, BPE {stab['cos0']['BPE']:.3f}, fixed {stab['cos0']['fixed']:.3f}). Read change: the same for "
                     f"the read embedding (mean over bases), c0 between reads of different sequences (LCP {stab['cos0_read']['LCP']:.3f}, BPE "
                     f"{stab['cos0_read']['BPE']:.3f}, fixed {stab['cos0_read']['fixed']:.3f}).") + "\n" +
            markdown("Embedding stability: paired ratios over sequences (95% bootstrap interval)", ["edit"] + list(next(iter(stab["paired"].values()))), paired,
                     "Ratio of the means over sequences; > 1: the other segmenter changes more than LCP. Paired bootstrap over sequences (2,000 resamples)."))


def bend_table(results: Path, numbers: dict) -> str:
    path = results / "bend/results.json"
    if not path.is_file():
        return ""
    models = json.loads(path.read_text())["models"]
    rows, values = [], {}
    for name in sorted(models, key=lambda n: (BEND_ORDER.index(n) if n in BEND_ORDER else len(BEND_ORDER), n)):
        entry = models[name]
        values[name] = {task: {"auroc": entry[task]["auroc"], "held_out_chromosomes": entry[task]["held_out_chromosomes"]} for task in ("expression", "disease")}
        rows.append([entry["label"], number(entry["expression"]["auroc"]), number(entry["disease"]["auroc"]),
                     number((entry["expression"]["auroc"] + entry["disease"]["auroc"]) / 2),
                     number(entry["expression"]["held_out_chromosomes"]), number(entry["disease"]["held_out_chromosomes"])])
    numbers["bend"] = values
    return markdown("BEND zero-shot noncoding variant effects (AUROC over all variants, no training)",
                    ["model", "expression", "disease", "mean of the 2", "expression, held-out chromosomes", "disease, held-out chromosomes"], rows,
                    "Held-out chromosomes: 1, 2, X and Y, not used for pretraining. Every model in float32.")


def efficiency_table(results: Path, numbers: dict) -> str:
    path = results / "efficiency/finetune_speed.json"
    if not path.is_file():
        return ""
    measured = json.loads(path.read_text())
    setup, rows = measured.pop("_setup", {}), []
    per_sequence = segmentation_per_sequence(results, setup.get("task", "splice_sites_all_revised").removesuffix("_revised"))
    for name, r in measured.items():
        if "error" not in r:
            extra = per_sequence.get(SEGMENTER_OF.get(name, "single-base"))
            with_segmentation = "-" if extra is None else f"{1 / (1 / r['sequences_per_second'] + extra):.1f}"
            rows.append([name, f"{r['parameters']:,}", f"{r['sequences_per_second']:.1f}", with_segmentation, f"{1000 * r['seconds_per_step']:.1f}",
                         f"{r['epoch_minutes']:.2f}", "-" if r["peak_memory_gb"] is None else f"{r['peak_memory_gb']:.2f}"])
    failed = [f"{name} ({r['error'].splitlines()[0][:80]})" for name, r in measured.items() if "error" in r]
    numbers["efficiency"] = {"setup": setup, **measured}
    return markdown("Fine-tuning throughput (one GPU, same task, batch and precision for every model)",
                    ["model", "parameters (fine-tuned model)", "sequences per second", "sequences per second, segmenting each sequence first",
                     "ms per step", "minutes per epoch", "peak GPU memory (GB)"], rows,
                    f"{setup.get('device', '')}, task {setup.get('task', '')}, batch {setup.get('batch', '')}, {setup.get('precision', '')}, "
                    f"{setup.get('warmup', '')} warm-up and {setup.get('steps', '')} timed steps. Inputs are segmented before timing; the "
                    "second column adds the segmentation time of one sequence (one strand, one CPU core, segmentation table) to each "
                    "sequence, the worst case of segmenting on the fly without overlap; fine-tuning segments each record once and reuses "
                    "it in every epoch." + (f" Failed: {'; '.join(failed)}." if failed else ""))


def timing_table(results: Path, numbers: dict) -> str:
    """Wall clock of the full fine-tunes of the timing plan (fold 0 of every NT18 task, one run per GPU); resumed runs are left out."""
    root = results / "timing"
    if not root.is_dir():
        return ""
    runs = {}
    for model in MODELS + NT18_ONLY_BASELINES + BASELINES + TIMING_BF16:
        for path in sorted((root / model / "nt18").glob("*/fold-00.json")):
            report = json.loads(path.read_text())
            if not report["resources"]["resumed"]:
                runs.setdefault(model, {})[path.parent.name] = report["resources"]
    if not runs:
        return ""
    common = set.intersection(*(set(tasks) for tasks in runs.values()))
    segmentation = segmentation_seconds(results)
    rows, split = [], {}
    for model, tasks in runs.items():
        # segmentation of a segment model: measured inside the run (resources.segmentation_seconds, part of its wall clock) or, for
        # runs marked segmented_before_timing (the paper's timed runs read stored segment boundaries), the separate measurement of
        # the same data (results/efficiency/segmentation_time.json); otherwise it is inside the wall clock but not known on its own.
        # Single-base baselines segment nothing (bases are encoded per batch, inside the clock).
        segmenter = SEGMENTER_OF.get(model, "single-base")
        train, segment, total = [], [], []
        for t in sorted(common):
            r = tasks[t]
            if segmenter == "single-base":
                parts = (r["wall_seconds"], 0.0)
            elif "segmentation_seconds" in r:
                parts = (r["wall_seconds"] - r["segmentation_seconds"], r["segmentation_seconds"])
            elif r.get("segmented_before_timing"):
                parts = (r["wall_seconds"], segmentation.get(segmenter, {}).get(t.removesuffix("_revised")))
            else:
                parts = (None, None)
            train.append(parts[0])
            segment.append(parts[1])
            total.append(r["wall_seconds"] + (parts[1] or 0) if r.get("segmented_before_timing") else r["wall_seconds"])
        known = None not in train and None not in segment
        split[model] = {"tasks": sorted(common), "fine-tuning seconds": train if known else None,
                        "segmentation seconds": segment if known else None, "total seconds": total}
        minutes = lambda values: f"{sum(values) / len(values) / 60:.2f}" if values else "-"
        unmeasured = any(tasks[t].get("segmented_before_timing") for t in common)
        rows.append([model, len(tasks), len(common), minutes(train) if known else "-",
                     minutes(segment) if known else "not measured" if unmeasured else "included",
                     minutes(total), f"{sum(total) / 3600:.2f}",
                     f"{max(r.get('peak_gpu_memory_allocated_gb', 0) for r in tasks.values()):.2f}", ", ".join(sorted({r.get("gpu", "cpu") for r in tasks.values()}))])
    numbers["timing"] = {"runs": runs, "split": split}
    return markdown("Full fine-tuning wall clock: fold 0 of every NT18 task, one run per GPU",
                    ["model", "tasks timed (of 18)", "tasks timed for every model", "fine-tuning, mean per task (minutes)",
                     "segmentation (dataset creation), mean per task (minutes)", "total, mean per task (minutes)", "total on those tasks (hours)",
                     "peak GPU memory allocated (GB)", "GPU"], rows,
                    "Fine-tuning: training and every per-epoch evaluation (resources.wall_seconds of the fold result, without its segmentation "
                    "time). Segmentation: the time to segment the task's training and test records on both strands, once per run "
                    "(resources.segmentation_seconds, or, for runs that segmented before the clock started, the measurement of the "
                    "segmentation table on the same data; \"included\": inside the fine-tuning time, not measured on its own); the "
                    "single-base baselines segment nothing (their bases are encoded per batch, inside the fine-tuning time). LoCUS runs in "
                    "bf16 and Caduceus-PS in float32, as in the accuracy runs; caduceus-ps-<size>-bf16: the same Caduceus-PS recipe in bf16, "
                    "timed only, so the models can also be compared at the same precision.")


# fewest tasks for an interval from resampling tasks: with 2 or 3 tasks, drawing one task for every slot is more likely than
# 2.5%, so the "95% interval" would be the range of the task means
MIN_BOOTSTRAP_TASKS = 5


def bootstrap_interval(pairs: list, differences: list, rng: random.Random) -> Optional[tuple[float, float]]:
    """95% interval of the mean difference over task-folds, resampling whole tasks (the folds of a task share its test set);
    None for fewer than MIN_BOOTSTRAP_TASKS tasks."""
    by_task = {}
    for (task, _), value in zip(pairs, differences):
        by_task.setdefault(task, []).append(value)
    groups = list(by_task.values())
    if len(groups) < MIN_BOOTSTRAP_TASKS:
        return None
    means = sorted(statistics.mean([x for g in (rng.choice(groups) for _ in groups) for x in g]) for _ in range(4000))
    return means[100], means[3899]


def interval_text(interval) -> str:
    return "n/a (< 5 tasks)" if interval is None else f"[{interval[0]:+.3f}, {interval[1]:+.3f}]"


def study_tables(results: Path, numbers: dict) -> str:
    root = results / "tokenizer_study"
    if not root.is_dir():
        return ""
    all_tasks = [t for _, ts in NT18_GROUPS for t in ts]
    folds = {(name, task): study_folds(root / directory / "nt18" / f"{task}_revised", NT18_FOLDS) for name, directory in TOKENIZERS for task in all_tasks}
    stat = lambda name, task, key: cell([row[key] for _, row in sorted(folds[name, task].items()) if key in row], NT18_FOLDS)
    out, numbers["tokenizer_study"] = [study_legend()], {}
    counts = sorted({n for rows in folds.values() for row in rows.values() for name, n in row.items() if name.endswith(" draws")})
    numbers["tokenizer_study"]["error draws per fold"] = counts
    span = "" if not counts else str(counts[0]) if counts[0] == counts[-1] else f"{counts[0]} to {counts[-1]}"
    draws_note = ("" if not counts else "Noisy sets: one error draw per set and fold." if counts == [1] else
                  f"Noisy sets: each score is the mean over {span} independent error draws of the set per fold.")
    for title, keys in (("clean, 1% / 3% errors and PBSIM3 reads", ("clean", "ont1", "ont3", "pbsim97")),
                        ("controls at 3%: errors anywhere / substitutions only", ("clean", "ont3", "sub3")),
                        ("controls at 1%: errors anywhere / substitutions only", ("clean", "ont1", "sub1"))):
        rows, cells = [], {}
        for group, tasks in NT18_GROUPS:
            for task in tasks:
                cells[task] = {f"{name} {key}": stat(name, task, key) for name, _ in TOKENIZERS for key in keys}
                rows.append([group, task] + [text(c) for c in cells[task].values()])
        for label, tasks in NT18_GROUPS + [("all 18", all_tasks)]:
            rows.append(["**mean**", label] + [number(group_mean([cells[t][k] for t in tasks])) for k in cells[all_tasks[0]]])
        numbers["tokenizer_study"][title] = cells
        out.append(markdown(f"Tokenizer study: test MCC, {title}", ["group", "task"] + list(cells[all_tasks[0]]), rows, draws_note))
    # errors outside the splice junction region: the splice tasks only (the annotated junction sits at the read centre, and the
    # protected window [200, 400) of the 600 bp reads keeps every edit at least 100 bp from it); the other tasks have no label
    # region that could be protected reliably
    title, keys = "errors outside the splice junction region (no edit within 100 bp of the junction)", ("clean", "ontw1", "ontw3")
    cells = {task: {f"{name} {key}": stat(name, task, key) for name, _ in TOKENIZERS for key in keys} for task in SPLICE}
    rows = [["splice", task] + [text(c) for c in cells[task].values()] for task in SPLICE]
    rows.append(["**mean**", "splice sites"] + [number(group_mean([cells[t][k] for t in SPLICE])) for k in cells[SPLICE[0]]])
    numbers["tokenizer_study"][title] = cells
    out.append(markdown(f"Tokenizer study: test MCC, {title}", ["group", "task"] + list(cells[SPLICE[0]]), rows, draws_note))

    def fold_spread(values: list, pairs: list, tasks: list):
        """SD over folds of the per-fold mean over the tasks; only when every task has every fold."""
        by_fold = {}
        for (task, fold), value in zip(pairs, values):
            by_fold.setdefault(fold, []).append(value)
        complete = len(by_fold) == NT18_FOLDS and all(len(v) == len(tasks) for v in by_fold.values())
        return statistics.stdev([statistics.mean(v) for v in by_fold.values()]) if complete and NT18_FOLDS > 1 else None

    def with_sd(mean: float, sd, signed: bool = True) -> str:
        return number(mean, signed) + ("" if sd is None else f" ± {sd:.3f}")

    rows, tests = [], []
    for label, key in STUDY_SETS:
        for scope, tasks in [("all 18", all_tasks)] + NT18_GROUPS:
            pairs = [(t, f) for t in tasks for f in range(NT18_FOLDS) if all(key in folds[name, t].get(f, {}) for name, _ in TOKENIZERS)]
            if not pairs:
                continue
            drop = {name: [folds[name, t][f][key] - folds[name, t][f]["clean"] for t, f in pairs] for name, _ in TOKENIZERS}
            versus = {other: [a - b for a, b in zip(drop["LCP"], drop[other])] for other in ("BPE", "fixed")}
            # one seed per row and comparison: an interval does not depend on the other rows
            interval = {other: bootstrap_interval(pairs, versus[other], random.Random(f"{key} {scope} {other}")) for other in versus}
            test = {"set": label, "scope": scope, "task_folds": len(pairs)}
            for name, _ in TOKENIZERS:
                test[f"{name} drop"], test[f"{name} drop fold sd"] = statistics.mean(drop[name]), fold_spread(drop[name], pairs, tasks)
            for other in versus:
                test[f"LCP - {other}"], test[f"LCP - {other} fold sd"] = statistics.mean(versus[other]), fold_spread(versus[other], pairs, tasks)
                test[f"LCP - {other} interval"] = None if interval[other] is None else list(interval[other])
                test[f"LCP less than {other}"] = f"{sum(x > 0 for x in versus[other])}/{len(pairs)}"
            tests.append(test)
            rows.append([label, scope, len(pairs)] + [with_sd(test[f"{name} drop"], test[f"{name} drop fold sd"]) for name, _ in TOKENIZERS] +
                        [x for other in versus for x in (with_sd(test[f"LCP - {other}"], test[f"LCP - {other} fold sd"]),
                                                         interval_text(interval[other]), test[f"LCP less than {other}"])])
    numbers["tokenizer_study"]["drop tests"] = tests
    out.append(markdown("Tokenizer study: paired drop tests by NT18 group (drop = set - clean on the same fold; > 0: LCP loses less)",
                        ["test set", "tasks", "task-folds", "LCP drop", "BPE drop", "fixed drop", "LCP - BPE", "LCP - BPE 95% CI", "LCP drops less than BPE",
                         "LCP - fixed", "LCP - fixed 95% CI", "LCP drops less than fixed"], rows,
                        "Drops and differences: mean over task-folds ± SD over folds of the per-fold mean over the row's tasks. Intervals: bootstrap "
                        "over tasks (4,000 resamples, one seed per row), for rows with at least 5 tasks only (with 2 or 3 tasks it would "
                        "be the range of the task means); the folds of a task share its test set and are kept together. Substitutions "
                        "only: the control sets' own error schedule, every event one substitution (no shift of the segment grid)."
                        + (" " + draws_note if draws_note else "")))
    return "\n".join(out)


def segmentation_seconds(results: Path) -> dict:
    """{segmenter: {task: seconds to segment the task's training and test records on both strands}}; empty when not measured."""
    path = results / "efficiency/segmentation_time.json"
    if not path.is_file():
        return {}
    tasks = json.loads(path.read_text())["tasks"]
    names = [k for k in next(iter(tasks.values())) if k not in ("records", "bases")]
    return {name: {task: sum(entry[name].values()) for task, entry in tasks.items()} for name in names}


def segmentation_per_sequence(results: Path, task: str) -> dict:
    """{segmenter: seconds per sequence and strand} on one task's records; empty when not measured."""
    path = results / "efficiency/segmentation_time.json"
    if not path.is_file():
        return {}
    entry = json.loads(path.read_text())["tasks"][task]
    count = 2 * sum(entry["records"].values())
    return {name: sum(entry[name].values()) / count for name in entry if name not in ("records", "bases")}


def segmentation_table(results: Path, numbers: dict) -> str:
    path = results / "efficiency/segmentation_time.json"
    if not path.is_file():
        return ""
    measured = json.loads(path.read_text())
    numbers["segmentation_time"] = measured
    per_task = segmentation_seconds(results)
    splice = segmentation_per_sequence(results, "splice_sites_all")
    rows = [[name, f"{total['seconds']:.1f}", f"{statistics.mean(per_task[name].values()):.1f}", f"{max(per_task[name].values()):.1f}",
             f"{total['bases_per_second'] / 1e6:.2f}", f"{1000 * splice[name]:.3f}"] for name, total in measured["totals"].items()]
    return markdown("Dataset creation: segmentation time of the NT18 fine-tuning data (one CPU core)",
                    ["segmenter", "18 tasks (seconds)", "mean per task (seconds)", "largest task (seconds)", "million bases per second",
                     "ms per 600 bp sequence (one strand)"], rows,
                    "Every training and test record of each task, both strands (benchmarks/efficiency/segmentation_time.py): lcp = LCP "
                    "level 2 (LoCUS), bpe = BPE with 4,096 tokens (twin-bpe-2m), fixed = 10 bp every 5 bp (twin-fixed-2m), single-base = "
                    f"base encoding (Caduceus). {measured.get('processor', '')}.")


def study_legend() -> str:
    rows = [["ont1, ont3", "nanopore-like errors (benchmarks/noisy_reads/noisy_sets.py): every base starts an error event with probability 1% or "
             "3%; an event is a substitution (40%), a deletion (35%) or an insertion before the base (25%); deletions and insertions are "
             "1, 2 or 3 bases long (probability 0.75, 0.1875, 0.0625), and an inserted stretch repeats the current base with probability "
             "0.5, otherwise random bases"],
            ["sub1, sub3", "control sets (benchmarks/noisy_reads/make_controls.py): an independent draw of the nanopore-like error schedule "
             "at 1% or 3%, every event a single-base substitution (no shift of the segment grid)"],
            ["ontw1, ontw3", "the full nanopore-like model on the same schedule as sub1 / sub3, without any edit inside a protected window; "
             "reported for the splice tasks, whose window keeps every edit at least 100 bp from the junction ([200, 400) of the 600 bp "
             "NT18 records, [100, 300) of the 400 bp GUE records)"],
            ["pbsim97", "PBSIM3 reads (ERRHMM-ONT-HQ error model, requested mean read accuracy 97%, realized 95.9% by PBSIM3's own report, each test sequence read once)"],
            ["<set>-d1 .. -d9", "further independent error draws of a set; the score of a fold on a set is the mean over its 10 draws"],
            ["realref, realsup, realhac", "real HG002 nanopore reads of the test windows (benchmarks/noisy_reads/real_reads.py): the reference "
             "sequences of the windows with 10 reads in both basecalls, and those reads in the sup and hac basecalls (10 draws each)"]]
    return markdown("Tokenizer study: modified test sets", ["set", "definition"], rows,
                    "Only the test set is modified; every model is fine-tuned on the clean training data. Drop = score on the set - score on "
                    "the clean test set, on the same fold.")


def scored_folds(directory: Path, folds: int) -> dict[int, dict]:
    """{fold: {set: test MCC}} from the score files of a study scored into its own directory (score.py --output-dir), whose
    "clean" is that study's own test set; error draws averaged as in study_folds."""
    out = {}
    for fold in range(folds):
        path = directory / f"fold-{fold:02d}.scored.json"
        if not path.is_file():
            continue
        scored, result = json.loads(path.read_text()), directory / f"fold-{fold:02d}.json"
        if result.is_file() and json.loads(result.read_text())["best_step"] != scored["best_step"]:
            raise ValueError(f"{path} was scored from a different checkpoint")
        draws = {}
        for name, metrics in scored["sets"].items():
            base, separator, draw = name.rpartition("-d")
            draws.setdefault(base if separator and draw.isdigit() else name, []).append(metrics["mcc"])
        out[fold] = {name: statistics.mean(values) for name, values in draws.items()}
        out[fold].update({f"{name} draws": len(values) for name, values in draws.items() if name != "clean"})
    return out


def drop_rows(rows: dict, folds: list, sets: list, reference: str = "clean") -> tuple[list, dict]:
    """Table rows (each twin: reference MCC and the drop on each set, mean ± SD over folds; then LCP's drop minus each other
    twin's, mean ± SD over folds and the folds where LCP loses less) and the numbers behind them."""
    msd = lambda values, signed: "-" if not values else (f"{statistics.mean(values):+.3f}" if signed else f"{statistics.mean(values):.3f}") + (
        f" ± {statistics.stdev(values):.3f}" if len(values) > 1 else "")
    table, values = [], {}
    for name, _ in TOKENIZERS:
        drops = {key: [rows[name][f][key] - rows[name][f][reference] for f in folds if key in rows[name][f]] for _, key in sets}
        values[name] = {reference: [rows[name][f][reference] for f in folds], **{f"drop {key}": d for key, d in drops.items()}}
        table.append([name, len(folds), msd(values[name][reference], False)] + [msd(drops[key], True) for _, key in sets])
    for other in ("BPE", "fixed"):
        cells = []
        for _, key in sets:
            have = [f for f in folds if key in rows["LCP"][f] and key in rows[other][f]]
            differences = [(rows["LCP"][f][key] - rows["LCP"][f][reference]) - (rows[other][f][key] - rows[other][f][reference]) for f in have]
            values[f"LCP - {other}"] = {**values.get(f"LCP - {other}", {}), key: differences}
            cells.append("-" if not differences else f"{msd(differences, True)} ({sum(d > 0 for d in differences)}/{len(differences)})")
        table.append([f"LCP - {other} (drop)", len(folds), ""] + cells)
    return table, values


GUE_SETS = [("1%", "ont1"), ("3%", "ont3"), ("PBSIM3", "pbsim97"), ("3% subst. only", "sub3"), ("3%, no edit within 100 bp of the junction", "ontw3")]


def gue_tables(results: Path, numbers: dict) -> str:
    """The tokenizer study on GUE splice (benchmarks/gue/prepare.py): every test row, and the test rows that share no splice site
    with NT18 (benchmarks/gue/independent_sets.py)."""
    out, numbers["gue_splice"] = [], {}
    header = ["model", "folds", "clean test MCC"] + [f"drop, {label}" for label, _ in GUE_SETS]
    note = ("Drop = test MCC on the set - clean test MCC, same fold; mean ± SD over folds. LCP - x: LCP's drop minus x's (> 0: LCP "
            "loses less), mean ± SD over folds, (folds where LCP loses less / folds). Noisy sets: mean over 10 error draws per fold.")
    for title, root, reader, extra in (
            ("GUE splice sites (4,562 test sequences of 400 bp): test MCC and drop from clean, 2M twins", results / "tokenizer_study",
             lambda d: study_folds(d, NT18_FOLDS), ""),
            ("GUE splice sites, test sequences that share no splice site with NT18 (3,998): test MCC and drop from clean, 2M twins",
             results / "gue_independent", lambda d: scored_folds(d, NT18_FOLDS),
             " The 564 test sequences whose 32 bases around the junction occur in an NT18 splice record are removed (same weights).")):
        rows = {name: reader(root / directory / "gue" / "gue_splice") for name, directory in TOKENIZERS}
        folds = [f for f in range(NT18_FOLDS) if all(f in rows[name] and "clean" in rows[name][f] for name, _ in TOKENIZERS)]
        if not folds:
            if any(rows.values()):
                WARNINGS.append(f"{title.split(':')[0]}: no fold finished by all three twins; table left out")
            continue
        if len(folds) < NT18_FOLDS:
            WARNINGS.append(f"{title.split(':')[0]}: {len(folds)} of {NT18_FOLDS} folds")
        table, values = drop_rows(rows, folds, GUE_SETS)
        numbers["gue_splice"][title] = values
        out.append(markdown(title, header, table, note + extra))
    return "\n".join(out)


def real_reads_table(results: Path, numbers: dict) -> str:
    """The tokenizer study's NT18 folds scored on real nanopore reads of the test windows (benchmarks/noisy_reads/real_reads.py)."""
    root = results / "real_reads"
    if not root.is_dir():
        return ""
    tasks = [t for _, ts in REAL_GROUPS for t in ts]
    rows = {(name, t): scored_folds(root / directory / "nt18" / f"{t}_revised", NT18_FOLDS) for name, directory in TOKENIZERS for t in tasks}
    have = [(t, f) for t in tasks for f in range(NT18_FOLDS) if all("realref" in rows[name, t].get(f, {}) and all(
        rows[name, t][f].get(f"{k} draws") == REAL_DRAWS for k in ("realsup", "realhac")) for name, _ in TOKENIZERS)]
    if not have:
        if any(rows.values()):
            WARNINGS.append("real reads: no task-fold scored for all three twins; table left out")
        return ""
    missing = len(tasks) * NT18_FOLDS - len(have)
    if missing:
        WARNINGS.append(f"real reads: {missing} of {len(tasks) * NT18_FOLDS} task-folds missing")
    table, values = [], {}
    for scope, scope_tasks in REAL_GROUPS + [("all 18", tasks)]:
        pairs = [(t, f) for t, f in have if t in scope_tasks]
        if not pairs:
            continue
        drop = {(name, b): [rows[name, t][f][f"real{b}"] - rows[name, t][f]["realref"] for t, f in pairs] for name, _ in TOKENIZERS for b in ("sup", "hac")}
        for name, _ in TOKENIZERS:
            cells = [scope if name == "LCP" else "", len(pairs) if name == "LCP" else "", name,
                     number(statistics.mean(rows[name, t][f]["realref"] for t, f in pairs)),
                     number(statistics.mean(drop[name, "sup"]), True), number(statistics.mean(drop[name, "hac"]), True)]
            for b in ("sup", "hac"):
                if name == "LCP":
                    cells.append("")
                    continue
                differences = [a - c for a, c in zip(drop["LCP", b], drop[name, b])]
                interval = bootstrap_interval(pairs, differences, random.Random(f"real {b} {scope} {name}"))
                cells.append(f"{statistics.mean(differences):+.3f} {interval_text(interval)} ({sum(d > 0 for d in differences)}/{len(differences)})")
                values.setdefault(scope, {})[f"LCP - {name}, {b}"] = {"mean": statistics.mean(differences),
                                                                      "interval": None if interval is None else list(interval),
                                                                      "LCP loses less": sum(d > 0 for d in differences), "task-folds": len(differences)}
            table.append(cells)
            values.setdefault(scope, {})[name] = {"realref": statistics.mean(rows[name, t][f]["realref"] for t, f in pairs),
                                                  "drop sup": statistics.mean(drop[name, "sup"]), "drop hac": statistics.mean(drop[name, "hac"])}
    numbers["real_reads"] = values
    return markdown("Real nanopore reads (HG002, R10.4.1) of the NT18 test windows: test MCC and drop from the reference sequences, 2M twins",
                    ["tasks", "task-folds", "model", "reference sequences (realref)", "drop, sup reads", "drop, hac reads",
                     "LCP - model, sup (95% CI over tasks) (LCP loses less)", "LCP - model, hac (95% CI over tasks) (LCP loses less)"], table,
                    "Windows with 10 reads in both basecalls (sup and hac, the same molecules); the windows left out are mostly repetitive "
                    "negatives, where reads do not map uniquely (counts in summary.json); "
                    "each fold's score on reads is the mean over the 10 reads per window, its drop is relative to the reference sequences "
                    "of the same windows. Divergence of the reads from the reference: data/benchmarks/real_reads/summary.json. "
                    "LCP - model: LCP's drop minus the model's (> 0: LCP loses less); interval: bootstrap over tasks (4,000 resamples), for "
                    "rows with at least 5 tasks only (with 2 or 3 tasks it would be the range of the task means).")


def single_nucleotide_table(results: Path, numbers: dict) -> str:
    """LCP's drops next to those of the single-nucleotide models (plan single-nucleotide), on the tokenizer study's simulated
    sets and on the real reads; the single-nucleotide drops show the loss from read errors without any merging into tokens."""
    out, numbers["single_nucleotide"] = [], {}
    all_tasks = [t for _, ts in NT18_GROUPS for t in ts]
    for title, root, sets, reference, lcp_directory, lcp_reader in (
            ("NT18, simulated errors (drop = set - clean test set)", results / "single_nucleotide", STUDY_SETS, "clean",
             results / "tokenizer_study" / "lcp" / "nt18", study_folds),
            ("NT18 test windows as real HG002 nanopore reads (drop = reads - reference sequences of the same windows)", results / "real_reads",
             [("sup reads", "realsup"), ("hac reads", "realhac")], "realref", results / "real_reads" / "lcp" / "nt18", scored_folds)):
        models = [(name, directory) for name, directory in SINGLE_NUCLEOTIDE if (root / directory / "nt18").is_dir()]
        if not models:
            continue
        names = ["LCP"] + [name for name, _ in models]
        rows = {("LCP", t): lcp_reader(lcp_directory / f"{t}_revised", NT18_FOLDS) for t in all_tasks}
        rows.update({(name, t): scored_folds(root / directory / "nt18" / f"{t}_revised", NT18_FOLDS) for name, directory in models for t in all_tasks})
        table, values = [], {}
        for label, key in sets:
            for scope, tasks in [("all 18", all_tasks)] + NT18_GROUPS:
                full = [(t, f) for t in tasks for f in range(NT18_FOLDS)]
                scored = {name: [(t, f) for t, f in full if reference in rows[name, t].get(f, {})
                                 and rows[name, t][f].get(f"{key} draws") == NOISY_DRAWS] for name in names}
                # a model enters a row once it has every task-fold of the row, so the rows of the other models do not change while
                # its runs come in; without LCP and one complete model, the row uses the task-folds scored for every model
                present = [name for name in names if len(scored[name]) == len(full)]
                if "LCP" not in present or len(present) < 2:
                    present = names
                pairs = [p for p in full if all(p in scored[name] for name in present)]
                missing = [name for name in names if name not in present]
                if scope == "all 18" and (len(pairs) < len(full) or missing):
                    WARNINGS.append(f"single-nucleotide models, {title.split(' (')[0]}, {label}: {len(pairs)} of {len(full)} task-folds"
                                    + (f"; not complete: {', '.join(f'{name} ({len(scored[name])})' for name in missing)}" if missing else ""))
                if not pairs:
                    continue
                drop = {name: [rows[name, t][f][key] - rows[name, t][f][reference] for t, f in pairs] for name in present}
                entry = {"task-folds": len(pairs), **{f"{name} drop": statistics.mean(drop[name]) if name in present else None for name in names}}
                cells = []
                for name in names[1:]:
                    if name not in present:
                        entry[f"LCP - {name}"] = None
                        cells += [f"n/a ({len(scored[name])} of {len(full)} task-folds)", "", ""]
                        continue
                    differences = [a - b for a, b in zip(drop["LCP"], drop[name])]
                    interval = bootstrap_interval(pairs, differences, random.Random(f"single-nucleotide {key} {scope} {name}"))
                    entry[f"LCP - {name}"] = {"mean": statistics.mean(differences), "interval": None if interval is None else list(interval),
                                              "LCP loses less": sum(d > 0 for d in differences)}
                    cells += [number(statistics.mean(differences), True), interval_text(interval), f"{sum(d > 0 for d in differences)}/{len(differences)}"]
                values.setdefault(label, {})[scope] = entry
                table.append([label, scope, len(pairs)] + [number(entry[f"{name} drop"], True) if name in present else "n/a" for name in names] + cells)
        numbers["single_nucleotide"][title] = values
        header = ["test set", "tasks", "task-folds"] + [f"{name} drop" for name in names] + [
            x for name in names[1:] for x in (f"LCP - {name}", f"LCP - {name} 95% CI", f"LCP loses less than {name}")]
        out.append(markdown(f"LCP and single-nucleotide models: {title}", header, table,
                            "Single-nucleotide models: " + ", ".join(f"{name} ({directory})" for name, directory in models) + ", fine-tuned "
                            "like the other baselines; their tokens are nucleotides, so a drop is the loss from the read errors without any "
                            "merging into tokens. Drops: mean over the task-folds scored for every model (each noisy set or basecall: mean "
                            "over its 10 draws); n/a: the model has not yet been scored on every task-fold of the row. LCP - x > 0: LCP loses less. Intervals: bootstrap over tasks (4,000 resamples), for rows "
                            "with at least 5 tasks only."))
    return "\n".join(out)


def split_interval(per_task: dict, seed: int = 0) -> list:
    """95% bootstrap interval of the mean over task-folds, resampling tasks (the folds of a task stay together): 4,000
    resamples with random.Random(seed).choices, as the split-ablation runs were first reported."""
    tasks, generator, means = list(per_task), random.Random(seed), []
    for _ in range(4000):
        sample = [value for task in generator.choices(tasks, k=len(tasks)) for value in per_task[task]]
        means.append(statistics.mean(sample))
    means.sort()
    return [means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1]]


def split_paired_rows(label: str, values: dict, digits: int, numbers: dict) -> list:
    """Rows of mean paired differences per NT18 group and over all 18 tasks; values: {(task, fold): difference}."""
    rows, all_tasks = [], [t for _, ts in NT18_GROUPS for t in ts]
    for group, tasks in NT18_GROUPS + [("all 18", all_tasks)]:
        per_task = {t: [values[t, f] for f in range(NT18_FOLDS)] for t in tasks if all((t, f) in values for f in range(NT18_FOLDS))}
        if len(per_task) < len(tasks):
            rows.append([label, group, f"{len(per_task)} of {len(tasks)} tasks complete", "", ""])
            continue
        differences = [v for vs in per_task.values() for v in vs]
        interval = split_interval(per_task) if len(tasks) >= MIN_BOOTSTRAP_TASKS else None
        numbers[f"{label}/{group}"] = {"mean": statistics.mean(differences), "ci95": interval, "higher": sum(d > 0 for d in differences),
                                       "n": len(differences)}
        rows.append([label, group, f"{statistics.mean(differences):+.{digits}f}",
                     f"[{interval[0]:+.{digits}f}, {interval[1]:+.{digits}f}]" if interval else "n/a (< 5 tasks)",
                     f"{sum(d > 0 for d in differences)}/{len(differences)}"])
    return rows


def split_tables(results: Path, numbers: dict) -> str:
    """The pretraining-split ablation: NT18 for the models pretrained without chromosomes 20 and 21 next to the released models'
    paper runs (same recipe, task and fold), the paired differences of test MCC and of the lowest validation loss, and, for the 2M
    models, the paired differences of the drops on the modified test sets and the real reads."""
    pairs = [(name, label, results / released) for name, label, released in SPLIT if (results / name / "nt18").is_dir()]
    if not pairs:
        return ""
    numbers["split_ablation"] = out = {"test_mcc": {}, "validation_loss": {}, "drops": {}}
    all_tasks = [t for _, ts in NT18_GROUPS for t in ts]
    reports = lambda directory: {(t, f): json.loads((directory / f"{t}_revised" / f"fold-{f:02d}.json").read_text())
                                 for t in all_tasks for f in range(NT18_FOLDS) if (directory / f"{t}_revised" / f"fold-{f:02d}.json").is_file()}
    loaded = {name: (reports(results / name / "nt18"), reports(released)) for name, _, released in pairs}
    rows, cells = [], {}
    for group, tasks in NT18_GROUPS:
        for task in tasks:
            for name, _, released in pairs:
                cells[task, name] = cell(fold_scores(results / name / "nt18" / f"{task}_revised", "mcc", NT18_FOLDS), NT18_FOLDS)
                cells[task, f"{name} released"] = cell(fold_scores(released / f"{task}_revised", "mcc", NT18_FOLDS), NT18_FOLDS)
                WARNINGS.extend(f"split ablation {name} {task}: {len(cells[task, name]['values'])} of {NT18_FOLDS} folds"
                                for _ in [0] if not cells[task, name]["complete"])
            rows.append([group, task] + [text(cells[task, key]) for name, _, _ in pairs for key in (f"{name} released", name)])
    for group, tasks in NT18_GROUPS + [("all 18", all_tasks)]:
        rows.append(["**mean**", group] + [number(group_mean([cells[t, key] for t in tasks])) for name, _, _ in pairs
                                          for key in (f"{name} released", name)])
    header = ["group", "task"] + [x for _, label, _ in pairs for x in (f"{label}, released (chromosomes 3-22)", f"{label}, chromosomes 1-19 and 22")]
    sections = [markdown("Pretraining-split ablation: NT18 test MCC without the test chromosomes in pretraining", header, rows,
                         "NT18 tests on chromosomes 20 and 21, which the released models saw in pretraining (chromosomes 3-22) and the "
                         "models of the split ablation did not (chromosomes 1-19 and 22; scripts/reproduce.py split-pretrain). Same "
                         "recipe, micro-batches, folds and seeds as the released model's runs (plans nt18 and tokenizer-study); mean ± SD "
                         "over the folds.")]
    rows = []
    for name, label, _ in pairs:
        new, old = loaded[name]
        mcc_of = lambda report: selected_entry(report).get("test_metrics", report["test_metrics"])["mcc"]
        rows += split_paired_rows(label, {k: mcc_of(new[k]) - mcc_of(old[k]) for k in new if k in old}, 3, out["test_mcc"])
    sections.append(markdown("Pretraining-split ablation: test MCC, without chromosomes 20 and 21 minus released (same task and fold)",
                             ["model", "tasks", "mean difference", "95% CI (tasks resampled)", "task-folds higher"], rows,
                             "> 0: the model that never saw the NT18 test chromosomes scores higher. Interval: bootstrap over tasks (4,000 "
                             "resamples; the folds of a task stay together), for groups of at least 5 tasks."))
    rows = []
    for name, label, _ in pairs:
        new, old = loaded[name]
        rows += split_paired_rows(label, {k: new[k]["validation_loss"] - old[k]["validation_loss"] for k in new if k in old}, 4,
                                  out["validation_loss"])
    sections.append(markdown("Pretraining-split ablation: lowest validation loss, without chromosomes 20 and 21 minus released",
                             ["model", "tasks", "mean difference", "95% CI (tasks resampled)", "task-folds higher"], rows,
                             "> 0: the model without chromosomes 20 and 21 has the higher (worse) validation loss at its selected epoch. The "
                             "validation folds come from the NT18 training sets (chromosomes 1-19, X and Y), never from chromosomes 20 and 21; "
                             "about 18% of their records lie on chromosomes 1 and 2, which only the models without chromosomes 20 and 21 saw "
                             "in pretraining. A test difference matched by a validation difference in the same direction is a difference "
                             "between the pretraining runs; a test difference without one would point to the test chromosomes."))
    scored = [(name, label, released) for name, label, released in pairs if name in SPLIT_REAL_READS]
    rows = []
    for name, label, released in scored:
        new_sim, old_sim = ({t: study_folds(d / f"{t}_revised", NT18_FOLDS) for t in all_tasks} for d in (results / name / "nt18", released))
        new_real, old_real = ({t: scored_folds(results / "real_reads" / d / "nt18" / f"{t}_revised", NT18_FOLDS) for t in all_tasks}
                              for d in (name, SPLIT_REAL_READS[name]))
        for set_label, key, reference, new_rows, old_rows, draws in (
                [(lab, k, "clean", new_sim, old_sim, NOISY_DRAWS) for lab, k in STUDY_SETS]
                + [("sup reads", "realsup", "realref", new_real, old_real, REAL_DRAWS), ("hac reads", "realhac", "realref", new_real, old_real, REAL_DRAWS)]):
            values = {}
            for t in all_tasks:
                for f in range(NT18_FOLDS):
                    a, b = new_rows[t].get(f, {}), old_rows[t].get(f, {})
                    if a.get(f"{key} draws") == draws and b.get(f"{key} draws") == draws and reference in a and reference in b:
                        values[t, f] = (a[key] - a[reference]) - (b[key] - b[reference])
            if not values:
                continue
            out["drops"].setdefault(label, {})
            group_rows = split_paired_rows(f"{label}, {set_label}", values, 3, out["drops"][label])
            rows += [row for row in group_rows if row[1] in ("all 18", "splice")] if set_label not in ("1%", "3%") else group_rows
    if rows:
        sections.append(markdown("Pretraining-split ablation: drops on the modified test sets and real reads, without chromosomes 20 and 21 "
                                 "minus released (same task and fold)",
                                 ["model, test set", "tasks", "mean difference of drops", "95% CI (tasks resampled)", "task-folds higher"], rows,
                                 "Drop = score on the set - score on the clean test set (real reads: on the reference sequences of the same "
                                 "windows), each set the mean over its 10 draws, as in the tokenizer study; > 0: the model without chromosomes "
                                 "20 and 21 loses less. Every NT18 group for the 1% and 3% sets, all 18 tasks and the splice sites for the others."))
    return "\n".join(sections)


INTERVAL_STUDIES = [("nt18", "NT18, simulated errors (drop = set - clean test set)"),
                    ("real_reads", "NT18 test windows as real HG002 nanopore reads (drop = reads - reference sequences of the same windows)"),
                    ("gue", "GUE splice sites, simulated errors (drop = set - clean test set)")]
INTERVAL_FAMILIES = {"ont1": "errors 1%", "ont3": "errors 3%", "pbsim97": "PBSIM3 reads", "sub3": "3%, substitutions only",
                     "ontw3": "3%, no edit within 100 bp of the junction", "realsup": "sup reads", "realhac": "hac reads"}
# the protected window of ontw1 / ontw3 is placed around the splice junction, so these sets are reported for splice scopes only (as in
# study_tables); window_intervals.py computes every family for every scope
SPLICE_ONLY_FAMILIES = {"ontw1", "ontw3"}


def interval_tables(results: Path, numbers: dict) -> str:
    """Drops with intervals from resampling test windows (benchmarks/noisy_reads/window_intervals.py, plan intervals), for the
    twins and, where scored, the single-nucleotide models."""
    out, numbers["intervals"] = [], {}
    for study, title in INTERVAL_STUDIES:
        path = results / "intervals" / f"{study}.json"
        if not path.is_file():
            WARNINGS.append(f"intervals: {path} missing (plan intervals); table left out")
            continue
        data = json.loads(path.read_text())
        for scope, entry in data["scopes"].items():
            if "splice" not in scope:
                entry["families"] = {f: row for f, row in entry["families"].items() if f not in SPLICE_ONLY_FAMILIES}
        numbers["intervals"][study] = data
        models = list(data["models"])
        cell = lambda v: f"{v['value']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
        rows = []
        for scope, entry in data["scopes"].items():
            for family, row in entry["families"].items():
                rows.append([scope, entry["task-folds"], INTERVAL_FAMILIES.get(family, family)] + [cell(row[f"{m} drop"]) for m in models]
                            + [cell(row[f"{models[0]} - {m}"]) for m in models[1:]])
        header = ["tasks", "task-folds", "set"] + [f"{m} drop" for m in models] + [f"{models[0]} - {m}" for m in models[1:]]
        out.append(markdown(f"Robustness with intervals: {title}", header, rows,
                            "Drop: mean over the task-folds of [mean over the 10 error draws (or reads) of the MCC on the set - the MCC on the "
                            f"reference set]; {models[0]} - x > 0: {models[0]} loses less. Intervals: 2.5th-97.5th percentile over "
                            f"{data['resamples']:,} resamples of the test windows (the same resample for every model, fold and set; "
                            "benchmarks/noisy_reads/window_intervals.py). Only task-folds scored for every model are used."))
    return "\n".join(out)


def main() -> None:
    global NT18_FOLDS, OTHER_FOLDS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/tables"))
    parser.add_argument("--nt18-folds", type=int, default=NT18_FOLDS,
                        help=f"folds of the NT18, tokenizer-study, GUE splice and real-read runs (default {NT18_FOLDS}, the paper's)")
    parser.add_argument("--other-folds", type=int, default=OTHER_FOLDS,
                        help=f"folds of the Genomic Benchmarks and held-out-chromosome runs (default {OTHER_FOLDS}, the paper's)")
    arguments = parser.parse_args()
    NT18_FOLDS, OTHER_FOLDS = arguments.nt18_folds, arguments.other_folds
    numbers = {}
    sections = [nt18_table(arguments.results, numbers), authors_table(arguments.results, numbers), heldout_table(arguments.results, numbers), gb_table(arguments.results, numbers),
                frozen_table(arguments.results, numbers), segmentation_table(arguments.results, numbers),
                efficiency_table(arguments.results, numbers), timing_table(arguments.results, numbers),
                study_tables(arguments.results, numbers), gue_tables(arguments.results, numbers), real_reads_table(arguments.results, numbers), single_nucleotide_table(arguments.results, numbers), interval_tables(arguments.results, numbers), parsing_tables(arguments.results, numbers), lcp_statistics_tables(arguments.results, numbers),
                boundary_table(arguments.results, numbers),
                stability_table(arguments.results, numbers), bend_table(arguments.results, numbers), split_tables(arguments.results, numbers)]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{arguments.output}.md").write_text("\n".join(s for s in sections if s))
    Path(f"{arguments.output}.json").write_text(json.dumps(numbers, indent=1) + "\n")
    print(f"wrote {arguments.output}.md and {arguments.output}.json")
    if WARNINGS:
        print(f"warning: {len(WARNINGS)} cell(s) with fewer folds than expected (--nt18-folds {NT18_FOLDS}):\n  " + "\n  ".join(WARNINGS))


if __name__ == "__main__":
    main()
