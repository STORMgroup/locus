"""Parse-level tests of the three segmenters of the tokenizer study (LCP level 2, fixed 10 bp segments every 5 bp, BPE with
a vocabulary of 4,096), with the exact settings of the pretrained models (configs/pretrain). No model is involved.

    edit locality         One edit is applied to a 4,096 bp window of chromosome 1 (held out of pretraining), at a random
                          position at least 1,024 bp from both ends: a substitution, or an insertion or deletion of 1 or 3
                          bases. The clean and the edited window are parsed. A segment of the clean parse is unchanged when
                          the edited parse has a segment with the same bases at the same position (after the shift by the
                          insertion or deletion); every other clean segment is changed. Reported per segmenter and edit:
                          changed segments, and the reach of the change, the largest distance from the edit to a base of a
                          changed segment.
    context independence  A 300 bp stretch of a chromosome-1 window is placed between two different pairs of 500 bp flanks
                          taken from other windows. A segment of the stretch (inside it in both copies) is consistent when
                          the other copy has a segment with the same bases at the same position in the stretch. Reported
                          per segmenter: the share of consistent segments, by distance from the nearer end of the stretch;
                          and per base, by distance (0-39 bp) from the nearer end, the share of bases whose covering
                          segments (bases and position) are the same in both copies, the input of the per-base output.

    python benchmarks/parsing/locality.py --windows data/GRCh38.p14.acgt.w4096.s4096 --output results/parsing/locality.json

Windows are every n-th chromosome-1 window (n set by --windows-used), edits and flanks are drawn from a generator seeded
with --seed, so the result is deterministic."""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from locus.config import GRCH38_CHROMOSOMES, load_config, with_paths_relative_to  # noqa: E402
from locus.data.genome import GenomeDataset  # noqa: E402
from locus.segmentation import Segmenter  # noqa: E402

SEGMENTERS = {"LCP": "locus-2m", "fixed": "twin-fixed-2m", "BPE": "twin-bpe-2m"}
EDITS = ["substitution", "insertion 1", "deletion 1", "insertion 3", "deletion 3"]
MARGIN, STRETCH, FLANK = 1024, 300, 500
DISTANCE_BINS = [(0, 10), (10, 20), (20, 50), (50, 150)]
BASE_DISTANCES = 40


def segmenters() -> dict:
    return {name: Segmenter(with_paths_relative_to(load_config(ROOT / "configs/pretrain" / f"{config}.yaml"), ROOT).data)
            for name, config in SEGMENTERS.items()}


def edit(sequence: str, kind: str, position: int, rng: random.Random) -> tuple[str, int]:
    """(edited sequence, shift of the bases after the edit)."""
    if kind == "substitution":
        return sequence[:position] + rng.choice([b for b in "ACGT" if b != sequence[position]]) + sequence[position + 1:], 0
    length = int(kind.split()[1])
    if kind.startswith("insertion"):
        return sequence[:position] + "".join(rng.choice("ACGT") for _ in range(length)) + sequence[position:], length
    return sequence[:position] + sequence[position + length:], -length


def changed_segments(clean: str, edited: str, clean_bounds: list, edited_bounds: list, position: int, kind: str, shift: int) -> list:
    """Clean segments without a segment of the same bases at the same (shifted) position in the edited parse."""
    edited_set = {(start, end) for start, end in edited_bounds}
    edited_span = (position, position + (1 if kind == "substitution" else max(-shift, 0)))
    changed = []
    for start, end in clean_bounds:
        if start < edited_span[1] and end > edited_span[0] or (kind.startswith("insertion") and start < position < end):
            changed.append((start, end))  # contains the edited bases (or the insertion point)
            continue
        mapped = (start, end) if end <= position else (start + shift, end + shift)
        if mapped not in edited_set or edited[mapped[0]:mapped[1]] != clean[start:end]:
            changed.append((start, end))
    return changed


def segment_statistics(windows: list, parsers: dict) -> dict:
    """Mean segment length and segments per 1,000 bases of the clean windows (LCP segments overlap)."""
    out = {}
    for name, parser in parsers.items():
        lengths = [end - start for sequence in windows for start, end in parser.bounds(sequence)]
        out[name] = {"mean_segment_length_bp": statistics.mean(lengths), "segments_per_kb": 1000 * len(lengths) / sum(map(len, windows))}
    return out


def edit_locality(windows: list, parsers: dict, rng: random.Random) -> dict:
    out = {name: {kind: {"changed": [], "reach": []} for kind in EDITS} for name in parsers}
    for sequence in windows:
        for kind in EDITS:
            position = rng.randrange(MARGIN, len(sequence) - MARGIN)
            edited, shift = edit(sequence, kind, position, rng)
            for name, parser in parsers.items():
                changed = changed_segments(sequence, edited, parser.bounds(sequence), parser.bounds(edited), position, kind, shift)
                out[name][kind]["changed"].append(len(changed))
                out[name][kind]["reach"].append(max((max(position - start, end - 1 - position) for start, end in changed), default=0))
    summary = {}
    for name, kinds in out.items():
        summary[name] = {}
        for kind, values in kinds.items():
            reach = sorted(values["reach"])
            summary[name][kind] = {"edits": len(reach), "changed_segments_mean": statistics.mean(values["changed"]),
                                   "changed_segments_median": statistics.median(values["changed"]),
                                   "reach_median_bp": statistics.median(reach), "reach_95th_bp": reach[int(0.95 * (len(reach) - 1))],
                                   "reach_max_bp": reach[-1], "share_reach_over_100bp": sum(r > 100 for r in reach) / len(reach)}
    return summary


def context_independence(windows: list, parsers: dict, rng: random.Random) -> dict:
    counts = {name: {"consistent": 0, "segments": 0, "bins": {f"{a}-{b}": [0, 0] for a, b in DISTANCE_BINS},
                     "bases": [[0, 0] for _ in range(BASE_DISTANCES)]} for name in parsers}
    for index, sequence in enumerate(windows):
        start = rng.randrange(MARGIN, len(sequence) - MARGIN - STRETCH)
        stretch = sequence[start:start + STRETCH]
        copies = []
        for _ in range(2):
            left, right = (windows[rng.randrange(len(windows))] for _ in range(2))
            left_flank = left[rng.randrange(len(left) - FLANK):][:FLANK]
            right_flank = right[rng.randrange(len(right) - FLANK):][:FLANK]
            left_flank = left_flank[:rng.randrange(FLANK - 4, FLANK + 1)]  # flank lengths differ by up to 4 bp (grid phase)
            copies.append((left_flank + stretch + right_flank, len(left_flank)))
        for name, parser in parsers.items():
            inside, covering = [], []
            for text, offset in copies:
                bounds = parser.bounds(text)
                inside.append({(s - offset, e - offset) for s, e in bounds if s >= offset and e <= offset + STRETCH})
                by_base = {}
                for s, e in bounds:
                    for j in range(max(s, offset), min(e, offset + STRETCH)):
                        if j - offset < BASE_DISTANCES or j - offset >= STRETCH - BASE_DISTANCES:
                            by_base.setdefault(j - offset, set()).add((s - offset, text[s:e]))
                covering.append(by_base)
            for j in list(range(BASE_DISTANCES)) + list(range(STRETCH - BASE_DISTANCES, STRETCH)):
                distance = min(j, STRETCH - 1 - j)
                counts[name]["bases"][distance][0] += covering[0].get(j) == covering[1].get(j)
                counts[name]["bases"][distance][1] += 1
            for s, e in inside[0]:
                consistent = (s, e) in inside[1]
                distance = min(s, STRETCH - e)
                counts[name]["segments"] += 1
                counts[name]["consistent"] += consistent
                for a, b in DISTANCE_BINS:
                    if a <= distance < b:
                        counts[name]["bins"][f"{a}-{b}"][0] += consistent
                        counts[name]["bins"][f"{a}-{b}"][1] += 1
    return {name: {"stretches": len(windows), "segments": c["segments"], "share_consistent": c["consistent"] / c["segments"],
                   "share_consistent_by_distance_bp": {k: (v[0] / v[1] if v[1] else None) for k, v in c["bins"].items()},
                   "share_bases_consistent_by_distance_bp": [v[0] / v[1] for v in c["bases"]]}
            for name, c in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", type=Path, default=ROOT / "data/GRCh38.p14.acgt.w4096.s4096", help="genome windows (scripts/prepare_genome.sh)")
    parser.add_argument("--windows-used", type=int, default=2000, help="number of chromosome-1 windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "results/parsing/locality.json")
    arguments = parser.parse_args()
    reads = [read.sequence for read in GenomeDataset(arguments.windows, [GRCH38_CHROMOSOMES["1"]], 4096)]
    step = max(1, len(reads) // arguments.windows_used)
    windows = reads[::step][:arguments.windows_used]
    parsers = segmenters()
    result = {"windows": len(windows), "chromosome_1_windows": len(reads), "seed": arguments.seed,
              "segmenters": SEGMENTERS, "segments": segment_statistics(windows, parsers),
              "edit_locality": edit_locality(windows, parsers, random.Random(arguments.seed)),
              "context_independence": context_independence(windows, parsers, random.Random(arguments.seed + 1))}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=1) + "\n")
    for name, r in result["segments"].items():
        print(f"{name:6s} mean segment length {r['mean_segment_length_bp']:.1f} bp, {r['segments_per_kb']:.0f} segments per kb")
    for name, kinds in result["edit_locality"].items():
        for kind, r in kinds.items():
            print(f"{name:6s} {kind:13s} changed segments {r['changed_segments_mean']:7.1f}  reach median {r['reach_median_bp']:6.0f} bp, "
                  f"95th {r['reach_95th_bp']:6.0f}, max {r['reach_max_bp']:6.0f}, over 100 bp {r['share_reach_over_100bp']:.3f}")
    for name, r in result["context_independence"].items():
        print(f"{name:6s} consistent {r['share_consistent']:.3f}  by distance from the stretch end: "
              + "  ".join(f"{k} bp {v:.3f}" for k, v in r["share_consistent_by_distance_bp"].items() if v is not None))
        curve = r["share_bases_consistent_by_distance_bp"]
        recovered = next((d for d in range(len(curve)) if all(v >= 0.99 for v in curve[d:])), None)
        print(f"{name:6s} bases with identical covering segments, by distance: " + " ".join(f"{d}:{curve[d]:.2f}" for d in range(0, len(curve), 3))
              + f"  (>= 0.99 from {recovered} bp)")
    print("wrote", arguments.output)


if __name__ == "__main__":
    main()
