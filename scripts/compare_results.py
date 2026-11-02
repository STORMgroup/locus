"""Compare collected results (scripts/collect_results.py) with the paper's (reference/paper_tables.json).

    python scripts/compare_results.py                                   # results/tables.json against the paper
    python scripts/compare_results.py --results other/tables.json --all # also list every cell, not only the summary

For every number present in both files (fold means of the fine-tuning tables, the frozen-embedding and BEND values) the difference rerun - paper is computed; the summary gives, per table, the number of cells compared, the mean
and largest absolute difference, and the cells whose difference exceeds --tolerance. Fine-tuning on GPU is not bit-for-bit
repeatable (floating-point atomics): in the paper, repeated fine-tunes with the same seed differed by about 0.013 MCC per
fold on average, so 18-task means of a rerun are expected to differ by about 0.01 and single small tasks such as
promoter_tata by more. The reference tables hold NT18 over all 10 folds (the default of reproduce.py); a rerun with
fewer folds is compared on its fold means. Throughput, wall clock and segmentation time depend on the hardware and are not
compared."""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIPPED = {"efficiency", "timing", "segmentation_time"}


def leaves(node, path=()):
    """(path, value) for every comparable number: the mean of a fold cell, or a plain float."""
    if isinstance(node, dict):
        if "mean" in node and "values" in node:
            if node["mean"] is not None:
                yield path, node["mean"], len(node["values"])
            return
        for key, value in node.items():
            yield from leaves(value, path + (str(key),))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from leaves(value, path + (str(index),))
    elif isinstance(node, float):
        yield path, node, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results/tables.json")
    parser.add_argument("--reference", type=Path, default=ROOT / "reference/paper_tables.json")
    parser.add_argument("--tolerance", type=float, default=0.03, help="flag differences larger than this (default 0.03)")
    parser.add_argument("--all", action="store_true", help="print every compared cell")
    arguments = parser.parse_args()
    reference = {path: (value, folds) for path, value, folds in leaves(json.loads(arguments.reference.read_text()))}
    results = {path: (value, folds) for path, value, folds in leaves(json.loads(arguments.results.read_text()))}
    by_table, flagged = {}, []
    for path in sorted(set(reference) & set(results)):
        if path[0] in SKIPPED or "interval" in " ".join(path):
            continue
        (paper, paper_folds), (rerun, rerun_folds) = reference[path], results[path]
        difference = rerun - paper
        by_table.setdefault(path[0], []).append(abs(difference))
        label = " / ".join(path)
        folds = "" if paper_folds is None else f"  (folds: paper {paper_folds}, rerun {rerun_folds})"
        if arguments.all:
            print(f"{label}: paper {paper:.4f}  rerun {rerun:.4f}  difference {difference:+.4f}{folds}")
        if abs(difference) > arguments.tolerance:
            flagged.append(f"{label}: paper {paper:.4f}  rerun {rerun:.4f}  difference {difference:+.4f}{folds}")
    if not by_table:
        sys.exit("no numbers in common; run scripts/collect_results.py first")
    print(f"{'table':20s} {'cells':>6s} {'mean |diff|':>12s} {'max |diff|':>11s}")
    for table, differences in by_table.items():
        print(f"{table:20s} {len(differences):6d} {statistics.mean(differences):12.4f} {max(differences):11.4f}")
    missing = len([path for path in reference if path[0] not in SKIPPED and path not in results])
    print(f"{missing} reference numbers have no counterpart in {arguments.results} (plans not run yet)")
    if flagged:
        print(f"\n{len(flagged)} cell(s) differ by more than {arguments.tolerance}:")
        print("\n".join(flagged))


if __name__ == "__main__":
    main()
