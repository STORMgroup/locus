"""Intervals from resampling test windows for the drops of several models on modified test sets, from the per-record
predictions saved by benchmarks/noisy_reads/score.py --save-predictions.

    python benchmarks/noisy_reads/window_intervals.py --models LCP=results/tokenizer_study/lcp/nt18 BPE=results/tokenizer_study/bpe/nt18 \\
        Caduceus=results/single_nucleotide/caduceus-ps-1.9m/nt18 --records data/benchmarks/nt18 \\
        --scopes splice=splice_sites_all,splice_sites_acceptors,splice_sites_donors --families ont1 ont3 pbsim97 sub3 --output results/intervals/nt18.json

Each model directory holds <task folder>/fold-XX.json, fold-XX.scored.json and fold-XX.predictions/<set>.npz (score.py
--output-dir keeps all three together). For a scope (one task or several) and a set family F (F, F-d1 .. F-d9), a model's
drop is the mean over the scope's task-folds of [mean over the draws of MCC(F-dk) - MCC(reference set)], the reference set
being the clean test set (or realref for the real reads). The first model is the reference model: "first - x" is its drop
minus x's (> 0: the first model loses less). Only task-folds that every model has, with every set scored, are used.

One bootstrap replicate draws the scope's distinct test windows with replacement and weights every record of a drawn window
by the window's multiplicity, the same weights for all models, folds, sets and draws; a window shared by tasks of the scope
(splice_sites_all and _acceptors, for example) is drawn once for all of them. NT18 windows are identified by the coordinates
in the record name (--ids coordinates), other benchmarks by the sequence up to reverse complement (--ids sequence). MCC
is computed from the weighted confusion matrix as locus/finetune/metrics.py computes it. Interval: 2.5th-97.5th percentile
of the replicates. Unweighted, every saved prediction must reproduce the MCC in its score file (the largest deviation is
reported, and a deviation above 1e-9 stops the script).
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def mcc_batch(conf: np.ndarray) -> np.ndarray:
    """MCC of confusion matrices (..., C, C), as locus.finetune.metrics.matthews_correlation."""
    samples = conf.sum(axis=(-1, -2))
    correct = np.trace(conf, axis1=-2, axis2=-1)
    predicted, observed = conf.sum(axis=-2), conf.sum(axis=-1)
    numerator = correct * samples - (predicted * observed).sum(-1)
    denominator = np.sqrt(np.maximum(samples ** 2 - (predicted ** 2).sum(-1), 0) * np.maximum(samples ** 2 - (observed ** 2).sum(-1), 0))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def load_records(path: Path) -> list:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def window_ids(records: list, kind: str) -> list:
    if kind == "coordinates":
        return [r["name"].split("|")[0] for r in records]
    return [min(r["sequence"], r["sequence"].translate(COMPLEMENT)[::-1]) for r in records]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="NAME=DIR, the first is the reference model")
    p.add_argument("--records", type=Path, required=True, help="data root of the reference test records (window identities)")
    p.add_argument("--record-dir", default="{task}_revised", help="directory of a task's reference records under --records ({task} filled in)")
    p.add_argument("--task-dir", default="{task}_revised", help="directory of a task under each model directory ({task} filled in)")
    p.add_argument("--reference", default="clean", help="the reference set (clean, or realref for the real reads)")
    p.add_argument("--families", nargs="+", required=True)
    p.add_argument("--scopes", nargs="+", required=True, help="NAME=task,task,...")
    p.add_argument("--ids", default="coordinates", choices=["coordinates", "sequence"])
    p.add_argument("--exclude", nargs="*", default=[], help="task=metadata.json:key, test rows to leave out")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--draws", type=int, default=10)
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261027)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    models = [tuple(item.split("=", 1)) for item in a.models]
    exclude = {}
    for item in a.exclude:
        task, rest = item.split("=", 1)
        path, key = rest.split(":", 1)
        exclude[task] = set(json.loads(Path(path).read_text())[key])
    scopes = [(item.split("=", 1)[0], item.split("=", 1)[1].split(",")) for item in a.scopes]
    tasks = sorted({t for _, ts in scopes for t in ts})
    draw = lambda family, k: family + (f"-d{k}" if k else "")
    sets = [a.reference] + [draw(f, k) for f in a.families for k in range(a.draws)]
    data, ids, folds_of, check = {}, {}, {}, 0.0
    for task in tasks:
        records = load_records(a.records / a.record_dir.format(task=task) / "test.jsonl.gz")
        keep = np.array([i for i in range(len(records)) if i not in exclude.get(task, set())])
        all_ids = window_ids(records, a.ids)
        ids[task] = [all_ids[i] for i in keep]
        complete = []
        for fold in range(a.folds):
            per_model = {}
            for name, root in models:
                directory = Path(root) / a.task_dir.format(task=task)
                scored_path = directory / f"fold-{fold:02d}.scored.json"
                if not scored_path.is_file():
                    break
                scored = json.loads(scored_path.read_text())["sets"]
                result = json.loads((directory / f"fold-{fold:02d}.json").read_text())
                if not all((directory / f"fold-{fold:02d}.predictions" / f"{s}.npz").is_file() for s in sets):
                    break
                for s in sets:
                    z = np.load(directory / f"fold-{fold:02d}.predictions" / f"{s}.npz")
                    labels, predictions = z["labels"].astype(np.int64), z["logits"].argmax(axis=1)
                    classes = z["logits"].shape[1]
                    if len(labels) != len(records):
                        raise ValueError(f"{directory} fold {fold} {s}: {len(labels)} records, the reference set has {len(records)}")
                    conf = np.zeros((classes, classes))
                    np.add.at(conf, (labels, predictions), 1)
                    stored = scored[s]["mcc"] if s in scored else result["test_metrics"]["mcc"]
                    check = max(check, abs(float(mcc_batch(conf)) - stored))
                    per_model[(name, s)] = (labels[keep] * classes + predictions[keep], classes)
            else:
                complete.append(fold)
                data.update({(name, task, fold, s): v for (name, s), v in per_model.items()})
        folds_of[task] = complete
    print(f"largest |MCC from predictions - stored score|: {check:.2e}", flush=True)
    if check > 1e-9:
        raise SystemExit("saved predictions do not reproduce the stored scores")
    rng = np.random.default_rng(a.seed)
    out = {"models": dict(models), "reference set": a.reference, "resamples": a.resamples, "seed": a.seed,
           "check_max_abs_mcc_difference": check, "scopes": {}}
    reference_model = models[0][0]
    for scope, scope_tasks in scopes:
        pairs = [(t, f) for t in scope_tasks for f in folds_of[t]]
        if not pairs:
            print(f"{scope}: no task-fold has every model's predictions; left out", flush=True)
            continue
        universe = sorted({w for t in scope_tasks for w in ids[t]})
        index = {w: k for k, w in enumerate(universe)}
        counts = rng.multinomial(len(universe), np.full(len(universe), 1 / len(universe)), size=a.resamples).astype(np.float32)
        weights = {t: np.concatenate([np.ones((1, len(ids[t])), np.float32), counts[:, [index[w] for w in ids[t]]]]) for t in scope_tasks}
        mcc = {}
        for (name, task, fold, s), (cells, classes) in data.items():
            if task in scope_tasks:
                onehot = np.zeros((len(cells), classes * classes), np.float32)
                onehot[np.arange(len(cells)), cells] = 1
                mcc[(name, task, fold, s)] = mcc_batch((weights[task] @ onehot).astype(np.float64).reshape(-1, classes, classes))
        entry = {"tasks": scope_tasks, "task-folds": len(pairs), "windows": len(universe), "families": {}}
        for family in a.families:
            drops = {name: np.mean([np.mean([mcc[(name, t, f, draw(family, k))] for k in range(a.draws)], axis=0) - mcc[(name, t, f, a.reference)]
                                    for t, f in pairs], axis=0) for name, _ in models}
            row = {f"{name} drop": {"value": float(v[0]), "ci95": [float(np.percentile(v[1:], q)) for q in (2.5, 97.5)]} for name, v in drops.items()}
            for name, _ in models[1:]:
                diff = drops[reference_model] - drops[name]
                row[f"{reference_model} - {name}"] = {"value": float(diff[0]), "ci95": [float(np.percentile(diff[1:], q)) for q in (2.5, 97.5)],
                                                      "share > 0": float((diff[1:] > 0).mean())}
            entry["families"][family] = row
            print(f"{scope:12s} {family:8s} n={len(pairs):3d}  " + "  ".join(
                f"{k} {v['value']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" for k, v in row.items()), flush=True)
        out["scopes"][scope] = entry
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", a.output)


if __name__ == "__main__":
    main()
