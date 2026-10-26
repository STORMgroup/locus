"""Collect the results into the paper tables (Markdown, plus every number as JSON).

    python scripts/collect_results.py --results results --output results/tables

Expected layout (written by scripts/reproduce.py, or by the scripts it runs):
    results/<model>/nt18/<task>/fold-XX.json                 NT18 revised, test MCC (LoCUS models, locus-2m-random-init, baselines)
    results/<model>/genomic_benchmarks/<task>/fold-XX.json   Genomic Benchmarks, test accuracy
    results/<model>/long_range_ep20/<task>/fold-XX.json       context-length ablation, enhancer test AUROC
    results/locus-2m-16k/long_range_ep20_constant/<task>/fold-XX.json   the same with a constant learning rate
    results/tokenizer_study/<lcp|fixed|bpe>/nt18/<task>/fold-XX.json (+ fold-XX.scored.json)
    results/locus-2m/nt18_heldout/<task>/fold-XX.json        NT18 tested on chromosomes outside pretraining
    results/long_range/frozen/fit_<model>_lrb_<task>_<4k|16k>.json       frozen-embedding long-range tasks
    results/parsing/locality.json                            parse-level edit locality and context independence
    results/bend/results.json                                BEND zero-shot variant effects
    results/efficiency/finetune_speed.json                   fine-tuning throughput
    results/timing/<model>/nt18/<task>/fold-00.json          full fine-tuning wall clock
NT18 results of the 2M twins (LoCUS-2M, twin-fixed-2m, twin-bpe-2m) are the tokenizer study's runs (clean test set) when
results/<model>/nt18 does not exist. Every fine-tuning
score is the test metric at the epoch with the lowest validation loss, recorded during fine-tuning; cells are mean +-
standard deviation over the 3 folds; group means are means of the task means and appear only when every task of the
group has all 3 folds. In the tokenizer study, the score of a fold on a noisy test set is the mean over the set's error
draws in fold-XX.scored.json (<set>, <set>-d1, ...; benchmarks/noisy_reads/noisy_sets.py); the tables state how many."""
import argparse
import json
import random
import statistics
from pathlib import Path

MODELS = ["locus-0.5m", "locus-2m", "locus-7m"]
BASELINES = ["caduceus-ps-1.9m", "hyenadna-tiny-0.44m", "hyenadna-small-3.3m"]
TWINS = {"locus-2m": "lcp", "twin-fixed-2m": "fixed", "twin-bpe-2m": "bpe"}
FROZEN_LABELS = {"locus-0.5m": "locus-0.5m", "locus-2m": "locus-2m", "locus-7m": "locus-7m", "tss": "no sequence: tissue + TSS distance"}
BEND_ORDER = ["locus-0.5m", "locus-2m", "locus-7m", "twin-fixed-2m", "twin-bpe-2m"]
NT18_GROUPS = [("histone", ["H3K4me3", "H3K27ac", "H3K36me3", "H3K9me3", "H2AFZ", "H3K27me3", "H3K4me1", "H3K4me2", "H3K9ac", "H4K20me1"]),
               ("promoter", ["promoter_all", "promoter_no_tata", "promoter_tata"]),
               ("splice", ["splice_sites_all", "splice_sites_acceptors", "splice_sites_donors"]),
               ("enhancer", ["enhancers", "enhancers_types"])]
GB_TASKS = ["demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "dummy_mouse_enhancers_ensembl", "human_enhancers_cohn",
            "human_enhancers_ensembl", "human_ensembl_regulatory", "human_nontata_promoters", "human_ocr_ensembl", "drosophila_enhancers_stark"]
# task types of the tokenizer study: C composition, P/W position with a label window, C/P borderline, M mixed
STUDY_GROUPS = [("C", "composition", ["H2AFZ", "H3K27ac", "H3K27me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac", "H3K9me3", "H4K20me1"]),
                ("P/W", "position (label window)", ["promoter_tata", "splice_sites_acceptors", "splice_sites_all", "splice_sites_donors"]),
                ("C/P", "position (borderline)", ["H3K36me3"]),
                ("M", "mixed", ["enhancers", "enhancers_types", "promoter_all", "promoter_no_tata"])]
TOKENIZERS = [("LCP", "lcp"), ("BPE", "bpe"), ("fixed", "fixed")]
FOLDS = 3  # Genomic Benchmarks and the long-range tasks
NT18_FOLDS = 3  # NT18 and the tokenizer study (--nt18-folds, up to 10)
SPLICE = ["splice_sites_acceptors", "splice_sites_all", "splice_sites_donors"]


def selected_entry(report: dict) -> dict:
    """History entry of the selected epoch (lowest validation loss; ties: higher validation MCC)."""
    history = report["history"]
    return history[min(range(len(history)), key=lambda i: (history[i]["validation_loss"], -history[i]["validation_metrics"]["mcc"]))]


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
    models = [m for m in MODELS + ["twin-fixed-2m", "twin-bpe-2m", "locus-2m-random-init"] + BASELINES if nt18_directory(results, m).is_dir()]
    rows, cells = [], {}
    for group, tasks in NT18_GROUPS:
        for task in tasks:
            cells[task] = {m: cell(fold_scores(nt18_directory(results, m) / f"{task}_revised", "mcc", NT18_FOLDS), NT18_FOLDS) for m in models}
            rows.append([group, task] + [text(cells[task][m]) for m in models])
    for group, tasks in NT18_GROUPS + [("all 18", [t for _, ts in NT18_GROUPS for t in ts])]:
        rows.append(["**mean**", group] + [number(group_mean([cells[t][m] for t in tasks])) for m in models])
    numbers["nt18"] = cells
    notes = ("twin-fixed-2m and twin-bpe-2m: LoCUS-2M with fixed 10 bp segments every 5 bp or BPE tokens (tokenizer study), same recipe and seeds. "
             if "twin-fixed-2m" in models or "twin-bpe-2m" in models else "")
    notes += ("locus-2m-random-init: the LoCUS-2M architecture fine-tuned from random initialization (no pretraining), same recipe and seeds. "
              if "locus-2m-random-init" in models else "")
    notes += "Baselines (caduceus-ps-1.9m, hyenadna-*): fine-tuned with the same protocol (scripts/finetune_baseline.py)." if set(models) & set(BASELINES) else ""
    return markdown("Nucleotide Transformer benchmark, revised (test MCC)", ["group", "task"] + models, rows, notes)


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
                for fold in range(FOLDS):
                    path = root / f"{task}_revised" / f"fold-{fold:02d}.json"
                    if path.is_file():
                        report = json.loads(path.read_text())
                        entry = selected_entry(report)
                        heldout.append(entry.get("test_metrics", report["test_metrics"])["mcc"])
                        official.append(entry["extra_test_metrics"]["official"]["mcc"])
                cells[task].update({f"{label}: held-out chromosomes": cell(heldout), f"{label}: official test": cell(official),
                                    f"{label}: held-out - official": cell([h - o for h, o in zip(heldout, official)])})
            rows.append([group, task] + [text(c) for c in cells[task].values()])
    for group, tasks in NT18_GROUPS + [("all 18", [t for _, ts in NT18_GROUPS for t in ts])]:
        rows.append(["**mean**", group] + [number(group_mean([cells[t][k] for t in tasks]), k.endswith("official")) for k in cells[tasks[0]]])
    numbers["nt18_heldout"] = cells
    notes = ("Fine-tuned on the official training set without its examples on chromosomes outside the pretraining chromosomes (3-22), "
             "tested on those examples (seen neither in pretraining nor in fine-tuning); the official test set (chromosomes 20 and 21, "
             "part of the pretraining chromosomes) is scored at the same selected epoch. Random init: the same architecture without "
             "pretrained weights, so neither test set was seen before fine-tuning; its held-out - official gap measures how much harder "
             "the held-out chromosomes are. Folds 0-2.")
    return markdown("NT18 on chromosomes outside pretraining (test MCC)", ["group", "task"] + list(cells[NT18_GROUPS[0][1][0]]), rows, notes)


def gb_table(results: Path, numbers: dict) -> str:
    models = [m for m in MODELS if (results / m / "genomic_benchmarks").is_dir()]
    cells = {task: {m: cell(fold_scores(results / m / "genomic_benchmarks" / f"gb_{task}", "accuracy")) for m in models} for task in GB_TASKS}
    rows = [[task] + [text(cells[task][m]) for m in models] for task in GB_TASKS]
    rows.append(["**mean, 8 tasks (no drosophila)**"] + [number(group_mean([cells[t][m] for t in GB_TASKS[:-1]])) for m in models])
    rows.append(["**mean, all 9**"] + [number(group_mean([cells[t][m] for t in GB_TASKS])) for m in models])
    numbers["genomic_benchmarks"] = cells
    return markdown("Genomic Benchmarks (test accuracy)", ["task"] + models, rows)


def frozen_fit(results: Path, model: str, task: str, length: str) -> dict:
    path = results / "long_range/frozen" / f"fit_{model}_lrb_{task}_{length}.json"
    return json.loads(path.read_text())["mean"] if path.is_file() else {}


def context_table(results: Path, numbers: dict) -> str:
    """LoCUS-2M and LoCUS-2M-16k fine-tuned for 20 epochs at 4 kb and 16 kb input, and LoCUS-2M-16k with a constant learning
    rate; the frozen eQTL AUROC of each pretrained model at both input lengths."""
    rows, cells = [], {}
    for label, model, directory in (("locus-2m", "locus-2m", results / "locus-2m" / "long_range_ep20"),
                                    ("locus-2m-16k", "locus-2m-16k", results / "locus-2m-16k" / "long_range_ep20"),
                                    ("locus-2m-16k, constant learning rate", None, results / "locus-2m-16k" / "long_range_ep20_constant")):
        if not directory.is_dir():
            continue
        cells[label] = {length: cell(fold_scores(directory / f"lrb_enhancer_{length}", "roc_auc")) for length in ("4k", "16k")}
        difference = (cells[label]["16k"]["mean"] - cells[label]["4k"]["mean"]
                      if cells[label]["16k"]["complete"] and cells[label]["4k"]["complete"] else None)
        frozen = [frozen_fit(results, model, "eqtl", length).get("auroc") if model else None for length in ("4k", "16k")]
        rows.append([label, text(cells[label]["4k"]), text(cells[label]["16k"]), number(difference, True)] + [number(v) for v in frozen])
    numbers["context_ablation"] = cells
    return markdown("Context-length ablation: long-range enhancer task, 20 epochs (test AUROC)",
                    ["model", "4 kb input", "16 kb input", "16 kb - 4 kb", "frozen eQTL AUROC, 4 kb", "frozen eQTL AUROC, 16 kb"], rows,
                    "Frozen eQTL columns: the pretrained model (long-range benchmark table), not the fine-tuned one.")


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
                    "Held-out chromosomes: 1, 2, X and Y, not used for pretraining.")


def efficiency_table(results: Path, numbers: dict) -> str:
    path = results / "efficiency/finetune_speed.json"
    if not path.is_file():
        return ""
    measured = json.loads(path.read_text())
    setup, rows = measured.pop("_setup", {}), []
    for name, r in measured.items():
        if "error" not in r:
            rows.append([name, f"{r['parameters']:,}", f"{r['sequences_per_second']:.1f}", f"{1000 * r['seconds_per_step']:.1f}", f"{r['epoch_minutes']:.2f}",
                         "-" if r["peak_memory_gb"] is None else f"{r['peak_memory_gb']:.2f}"])
    failed = [f"{name} ({r['error'].splitlines()[0][:80]})" for name, r in measured.items() if "error" in r]
    numbers["efficiency"] = {"setup": setup, **measured}
    return markdown("Fine-tuning throughput (one GPU, same task, batch and precision for every model)",
                    ["model", "parameters (fine-tuned model)", "sequences per second", "ms per step", "minutes per epoch", "peak GPU memory (GB)"], rows,
                    f"{setup.get('device', '')}, task {setup.get('task', '')}, batch {setup.get('batch', '')}, {setup.get('precision', '')}, "
                    f"{setup.get('warmup', '')} warm-up and {setup.get('steps', '')} timed steps." + (f" Failed: {'; '.join(failed)}." if failed else ""))


def timing_table(results: Path, numbers: dict) -> str:
    """Wall clock of the full fine-tunes of the timing plan (fold 0 of every NT18 task, one run per GPU); resumed runs are left out."""
    root = results / "timing"
    if not root.is_dir():
        return ""
    runs = {}
    for model in MODELS + BASELINES:
        for path in sorted((root / model / "nt18").glob("*/fold-00.json")):
            report = json.loads(path.read_text())
            if not report["resources"]["resumed"]:
                runs.setdefault(model, {})[path.parent.name] = report["resources"]
    if not runs:
        return ""
    common = set.intersection(*(set(tasks) for tasks in runs.values()))
    rows = []
    for model, tasks in runs.items():
        hours = sum(tasks[t]["wall_seconds"] for t in common) / 3600
        rows.append([model, len(tasks), len(common), f"{hours:.2f}", f"{60 * hours / len(common):.1f}" if common else "-",
                     f"{max(r.get('peak_gpu_memory_allocated_gb', 0) for r in tasks.values()):.2f}", ", ".join(sorted({r.get("gpu", "cpu") for r in tasks.values()}))])
    numbers["timing"] = runs
    return markdown("Full fine-tuning wall clock: fold 0 of every NT18 task, one run per GPU",
                    ["model", "tasks timed (of 18)", "tasks timed for every model", "wall clock on those tasks (hours)", "mean per task (minutes)",
                     "peak GPU memory allocated (GB)", "GPU"], rows,
                    "Wall clock of fine-tuning as recorded in each fold result (resources.wall_seconds): training and every per-epoch "
                    "evaluation, from after the datasets are loaded to the result.")


def bootstrap_interval(pairs: list, differences: list, rng: random.Random) -> tuple[float, float]:
    """95% interval of the mean difference over task-folds, resampling whole tasks (the folds of a task share its test set)."""
    by_task = {}
    for (task, _), value in zip(pairs, differences):
        by_task.setdefault(task, []).append(value)
    groups = list(by_task.values())
    means = sorted(statistics.mean([x for g in (rng.choice(groups) for _ in groups) for x in g]) for _ in range(4000))
    return means[100], means[3899]


def study_tables(results: Path, numbers: dict) -> str:
    root = results / "tokenizer_study"
    if not root.is_dir():
        return ""
    folds = {(name, task): study_folds(root / directory / "nt18" / f"{task}_revised", NT18_FOLDS)
             for name, directory in TOKENIZERS for _, _, tasks in STUDY_GROUPS for task in tasks}
    stat = lambda name, task, key: cell([row[key] for _, row in sorted(folds[name, task].items()) if key in row], NT18_FOLDS)
    out, numbers["tokenizer_study"] = [], {}
    counts = sorted({n for rows in folds.values() for row in rows.values() for name, n in row.items() if name.endswith(" draws")})
    numbers["tokenizer_study"]["error draws per fold"] = counts
    span = "" if not counts else str(counts[0]) if counts[0] == counts[-1] else f"{counts[0]} to {counts[-1]}"
    draws_note = ("" if not counts else "Noisy sets: one error draw per set and fold." if counts == [1] else
                  f"Noisy sets: each score is the mean over {span} independent error draws of the set per fold.")
    for title, keys in (("clean, 1% / 3% errors and PBSIM3 reads", ("clean", "ont1", "ont3", "pbsim97")),
                        ("controls at 3%: errors anywhere / substitutions only", ("clean", "ont3", "sub3")),
                        ("controls at 1%: errors anywhere / substitutions only", ("clean", "ont1", "sub1"))):
        rows, cells = [], {}
        for code, _, tasks in STUDY_GROUPS:
            for task in tasks:
                cells[task] = {f"{name} {key}": stat(name, task, key) for name, _ in TOKENIZERS for key in keys}
                rows.append([code, task] + [text(c) for c in cells[task].values()])
        all_tasks = [t for _, _, ts in STUDY_GROUPS for t in ts]
        for label, tasks in [(name, ts) for _, name, ts in STUDY_GROUPS if len(ts) > 1] + [("all 18", all_tasks)]:
            rows.append(["**mean**", label] + [number(group_mean([cells[t][k] for t in tasks])) for k in cells[all_tasks[0]]])
        numbers["tokenizer_study"][title] = cells
        out.append(markdown(f"Tokenizer study: test MCC, {title}", ["type", "task"] + list(cells[all_tasks[0]]), rows, draws_note))
    # errors outside the splice junction region: the splice tasks only (the annotated junction sits at the read centre, and the
    # protected window [200, 400) of the 600 bp reads keeps every edit at least 100 bp from it); the other tasks have no label
    # region that could be protected reliably
    title, keys = "errors outside the splice junction region (no edit within 100 bp of the junction)", ("clean", "ontw1", "ontw3")
    cells = {task: {f"{name} {key}": stat(name, task, key) for name, _ in TOKENIZERS for key in keys} for task in SPLICE}
    rows = [["P/W", task] + [text(c) for c in cells[task].values()] for task in SPLICE]
    rows.append(["**mean**", "splice sites"] + [number(group_mean([cells[t][k] for t in SPLICE])) for k in cells[SPLICE[0]]])
    numbers["tokenizer_study"][title] = cells
    out.append(markdown(f"Tokenizer study: test MCC, {title}", ["type", "task"] + list(cells[SPLICE[0]]), rows, draws_note))

    all_tasks = [t for _, _, ts in STUDY_GROUPS for t in ts]
    scopes = [("all 18 tasks", all_tasks), ("position, label window", STUDY_GROUPS[1][2]), ("position (5 tasks)", STUDY_GROUPS[1][2] + STUDY_GROUPS[2][2]),
              ("composition + mixed (13 tasks)", STUDY_GROUPS[0][2] + STUDY_GROUPS[3][2])]
    rows, tests = [], []
    for key, label in (("ont3", "errors 3%, anywhere"), ("ont1", "errors 1%, anywhere"), ("pbsim97", "PBSIM3")):
        for scope, tasks in scopes:
            pairs = [(t, f) for t in tasks for f in range(NT18_FOLDS) if all(key in folds[name, t].get(f, {}) for name, _ in TOKENIZERS)]
            if not pairs:
                continue
            drop = {name: [folds[name, t][f][key] - folds[name, t][f]["clean"] for t, f in pairs] for name, _ in TOKENIZERS}
            versus_bpe = [a - b for a, b in zip(drop["LCP"], drop["BPE"])]
            versus_fixed = [a - b for a, b in zip(drop["LCP"], drop["fixed"])]
            rng = random.Random(f"{key} {scope}")  # one seed per row: an interval does not depend on the other rows
            low_bpe, high_bpe = bootstrap_interval(pairs, versus_bpe, rng)
            low_fixed, high_fixed = bootstrap_interval(pairs, versus_fixed, rng)
            test = {"set": label, "scope": scope, "task_folds": len(pairs), **{f"{name} drop": statistics.mean(drop[name]) for name, _ in TOKENIZERS},
                    "LCP - BPE": statistics.mean(versus_bpe), "LCP - BPE interval": [low_bpe, high_bpe],
                    "LCP less than BPE": f"{sum(x > 0 for x in versus_bpe)}/{len(versus_bpe)}",
                    "LCP - fixed": statistics.mean(versus_fixed), "LCP - fixed interval": [low_fixed, high_fixed]}
            tests.append(test)
            rows.append([label, scope, len(pairs), number(test["LCP drop"], True), number(test["BPE drop"], True), number(test["fixed drop"], True),
                         f"{test['LCP - BPE']:+.3f} [{low_bpe:+.3f}, {high_bpe:+.3f}]", test["LCP less than BPE"],
                         f"{test['LCP - fixed']:+.3f} [{low_fixed:+.3f}, {high_fixed:+.3f}]"])
    numbers["tokenizer_study"]["drop tests"] = tests
    out.append(markdown("Tokenizer study: paired drop tests (drop = set - clean on the same fold; > 0: LCP loses less)",
                        ["test set", "tasks", "task-folds", "LCP drop", "BPE drop", "fixed drop", "LCP - BPE [95% CI]", "LCP drops less",
                         "LCP - fixed [95% CI]"], rows,
                        "Intervals: bootstrap over tasks (4,000 resamples, one seed per row); the folds of a task share its test set and are kept together."
                        + (" " + draws_note if draws_note else "")))
    return "\n".join(out)


def main() -> None:
    global NT18_FOLDS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/tables"))
    parser.add_argument("--nt18-folds", type=int, default=NT18_FOLDS, help="folds of the NT18 and tokenizer-study runs (up to 10)")
    arguments = parser.parse_args()
    NT18_FOLDS = arguments.nt18_folds
    numbers = {}
    sections = [nt18_table(arguments.results, numbers), heldout_table(arguments.results, numbers), gb_table(arguments.results, numbers),
                frozen_table(arguments.results, numbers),
                context_table(arguments.results, numbers), efficiency_table(arguments.results, numbers), timing_table(arguments.results, numbers),
                study_tables(arguments.results, numbers), parsing_tables(arguments.results, numbers), bend_table(arguments.results, numbers)]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{arguments.output}.md").write_text("\n".join(s for s in sections if s))
    Path(f"{arguments.output}.json").write_text(json.dumps(numbers, indent=1) + "\n")
    print(f"wrote {arguments.output}.md and {arguments.output}.json")


if __name__ == "__main__":
    main()
