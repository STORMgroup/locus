"""Statistics of the LCP parse used by the LoCUS models (level 2, lcptools), without a model.

    segmentation   level-1 cores and level-2 segments (the LoCUS segments) of genome windows of the pretraining chromosomes
                   (every k-th window, --windows-used): level-1 core types and lengths, level-1 start spacing, and for the
                   segments their length, start spacing, how many segments cover each base, and how many bases at each end of
                   a window no segment covers
    worst case     one edit at the centre of --sequences chromosome-1 sequences of 2,048 bp (held out of pretraining) and of as
                   many uniform random sequences: a substitution, or an insertion or deletion of 1, 2 or 3 bases. Per level
                   (1 and 2) and edit, the changed cores (benchmarks/parsing/locality.py: a clean core is changed when the
                   edited parse has no core with the same bases at the same, shifted position) and their reach (largest
                   distance from the edit to a base of a changed core), also without the edits near a homopolymer of 6 or
                   more bases (within 40 bp), where a run core can be long

    python benchmarks/parsing/lcp_statistics.py --windows data/GRCh38.p14.acgt.w4096.s4096 --output results/parsing/lcp_statistics.json

The sequences, edits and random sequences are drawn from fixed seeds, so the result is deterministic."""
import argparse
import ctypes
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks/parsing"))

from locality import changed_segments  # noqa: E402
from locus.config import GRCH38_CHROMOSOMES, DataConfig  # noqa: E402
from locus.data.genome import GenomeDataset  # noqa: E402
from locus.segmentation import LCPSegmenter, _LPS  # noqa: E402

LENGTH, SITE, RUN_NEAR, RUN_LONG = 2048, 1024, 40, 6
EDITS = ["substitution", "insertion 1", "insertion 2", "insertion 3", "deletion 1", "deletion 2", "deletion 3"]
TYPES = {1: "local minimum", 2: "local maximum", 3: "run", 4: "between cores"}  # LCP_CORE_* of lcptools core.h


def cores(library, sequence: str, level: int) -> list:
    """(start, end, type) of the level-`level` cores of an A/C/G/T sequence, from lcptools."""
    value = _LPS()
    library.init_lps(ctypes.byref(value), sequence.encode("ascii"), len(sequence))
    try:
        if level > 1:
            library.lps_deepen(ctypes.byref(value), level)
        return [(int(value.cores[i].start), int(value.cores[i].end), int(value.cores[i].type)) for i in range(value.size)]
    finally:
        library.free_lps(ctypes.byref(value))


def segmentation(windows: list, library) -> dict:
    out = {"windows": len(windows), "bases": sum(map(len, windows))}
    for level in (1, 2):
        lengths, by_type, spacing, coverage, left, right, count = [], {}, [], Counter(), [], [], 0
        for sequence in windows:
            parsed = cores(library, sequence, level)
            count += len(parsed)
            lengths += [e - s for s, e, _ in parsed]
            for s, e, kind in parsed:
                by_type.setdefault(TYPES.get(kind, str(kind)), []).append(e - s)
            starts = sorted(s for s, _, _ in parsed)
            spacing += [b - a for a, b in zip(starts, starts[1:])]
            if level == 2:
                depth = [0] * (len(sequence) + 1)
                for s, e, _ in parsed:
                    depth[s] += 1
                    depth[e] -= 1
                running = 0
                for j in range(len(sequence)):
                    running += depth[j]
                    coverage[min(running, 4)] += 1
                left.append(min(starts))
                right.append(len(sequence) - max(e for _, e, _ in parsed))
        level_out = {"cores": count, "cores_per_kb": 1000 * count / out["bases"], "bases_per_core": out["bases"] / count,
                     "mean_length_bp": statistics.mean(lengths), "median_length_bp": statistics.median(lengths), "max_length_bp": max(lengths),
                     "mean_start_spacing_bp": statistics.mean(spacing),
                     "by_type": {kind: {"cores": len(v), "share": len(v) / count, "mean_length_bp": statistics.mean(v), "max_length_bp": max(v)}
                                 for kind, v in sorted(by_type.items())}}
        if level == 1:
            other = [x for kind, v in by_type.items() if kind != "run" for x in v]
            level_out["share_of_non_run_cores_with_3_bases"] = sum(x == 3 for x in other) / len(other)
        else:
            level_out["share_of_bases_covered_by"] = {("4 or more" if k == 4 else str(k)): coverage[k] / out["bases"] for k in range(5)}
            level_out["share_of_bases_in_2_or_more"] = sum(coverage[k] for k in (2, 3, 4)) / out["bases"]
            level_out["uncovered_bases_at_start_mean"] = statistics.mean(left)
            level_out["uncovered_bases_at_end_mean"] = statistics.mean(right)
        out[f"level {level}"] = level_out
    return out


def edit(sequence: str, kind: str, rng: random.Random) -> tuple:
    if kind == "substitution":
        return sequence[:SITE] + rng.choice([b for b in "ACGT" if b != sequence[SITE]]) + sequence[SITE + 1:], 0
    n = int(kind.split()[1])
    if kind.startswith("insertion"):
        return sequence[:SITE] + "".join(rng.choice("ACGT") for _ in range(n)) + sequence[SITE:], n
    return sequence[:SITE] + sequence[SITE + n:], -n


def longest_run(text: str) -> int:
    return max(len(m.group(0)) for m in re.finditer(r"A+|C+|G+|T+", text))


def worst_case(sets: dict) -> dict:
    parsers = {level: LCPSegmenter(level=level) for level in (1, 2)}
    out = {}
    for name, sequences in sets.items():
        rng = random.Random(42)
        values = {level: {"changed": [], "reach": [], "run": []} for level in (1, 2)}
        for sequence in sequences:
            for kind in EDITS:
                edited, shift = edit(sequence, kind, rng)
                run = longest_run(sequence[SITE - RUN_NEAR:SITE + RUN_NEAR])
                for level, parser in parsers.items():
                    hit = changed_segments(sequence, edited, parser.bounds(sequence), parser.bounds(edited), SITE, kind, shift)
                    values[level]["changed"].append(len(hit))
                    values[level]["reach"].append(max((max(SITE - s, e - 1 - SITE) for s, e in hit), default=0))
                    values[level]["run"].append(run)
        out[name] = {}
        for level, v in values.items():
            changed, reach = v["changed"], sorted(v["reach"])
            short = [(c, r) for c, r, run in zip(v["changed"], v["reach"], v["run"]) if run < RUN_LONG]
            longest = max(range(len(v["reach"])), key=lambda i: v["reach"][i])
            out[name][f"level {level}"] = {
                "edits": len(changed), "changed_mean": statistics.mean(changed), "changed_max": max(changed),
                "reach_median_bp": statistics.median(reach), "reach_99th_bp": reach[int(0.99 * (len(reach) - 1))], "reach_max_bp": reach[-1],
                "longest_homopolymer_near_the_largest_reach_bp": v["run"][longest],
                "without_homopolymer_6_bp_nearby": {"edits": len(short), "changed_max": max(c for c, _ in short), "reach_max_bp": max(r for _, r in short)}}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", type=Path, default=ROOT / "data/GRCh38.p14.acgt.w4096.s4096", help="genome windows (scripts/prepare_genome.sh)")
    parser.add_argument("--windows-used", type=int, default=2000, help="pretraining-chromosome windows for the segmentation statistics")
    parser.add_argument("--sequences", type=int, default=10000, help="chromosome-1 (and random) sequences for the worst case")
    parser.add_argument("--output", type=Path, default=ROOT / "results/parsing/lcp_statistics.json")
    arguments = parser.parse_args()
    library = LCPSegmenter(level=2).library

    pretraining = GenomeDataset(arguments.windows, DataConfig().train_sources, 4096)
    total = sum(1 for _ in pretraining)
    step = max(1, total // arguments.windows_used)
    windows = [read.sequence for i, read in enumerate(GenomeDataset(arguments.windows, DataConfig().train_sources, 4096)) if i % step == 0]
    windows = windows[:arguments.windows_used]
    chr1 = [read.sequence for read in GenomeDataset(arguments.windows, [GRCH38_CHROMOSOMES["1"]], 4096)]
    sets = {"chromosome 1": [read[1024:1024 + LENGTH] for read in chr1[::max(1, len(chr1) // arguments.sequences)][:arguments.sequences]]}
    rng = random.Random(7)
    sets["random"] = ["".join(rng.choice("ACGT") for _ in range(LENGTH)) for _ in range(arguments.sequences)]

    result = {"pretraining_windows": total, "windows_step": step, "segmentation": segmentation(windows, library),
              "worst_case": {"sequences_per_set": arguments.sequences, "length": LENGTH, "edit_position": SITE, "edits": EDITS,
                             "sets": worst_case(sets)}}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=1) + "\n")
    level1, level2 = result["segmentation"]["level 1"], result["segmentation"]["level 2"]
    print(f"level 1: a core every {level1['mean_start_spacing_bp']:.2f} bp; non-run cores of 3 bases {level1['share_of_non_run_cores_with_3_bases']:.4f}")
    print(f"level 2: {level2['bases_per_core']:.2f} bp per segment, mean length {level2['mean_length_bp']:.2f} bp, in >= 2 segments "
          f"{level2['share_of_bases_in_2_or_more']:.3f}, uncovered ends {level2['uncovered_bases_at_start_mean']:.1f} / {level2['uncovered_bases_at_end_mean']:.1f}")
    for name, levels in result["worst_case"]["sets"].items():
        for level, r in levels.items():
            print(f"{name:12s} {level}: changed mean {r['changed_mean']:.2f} max {r['changed_max']}, reach median {r['reach_median_bp']:.0f} "
                  f"99th {r['reach_99th_bp']:.0f} max {r['reach_max_bp']} (homopolymer {r['longest_homopolymer_near_the_largest_reach_bp']} bp)")
    print("wrote", arguments.output)


if __name__ == "__main__":
    main()
