"""Embedding stability: how far one edit propagates through the frozen 2M twins (LoCUS-2M, twin-fixed-2m, twin-bpe-2m).

--sequences sequences of 2,048 bp from chromosome 1 (held out of pretraining; the centre of every k-th N-free 4 kb window) get
one edit at the centre: a substitution, or an insertion or deletion of 1, 2 or 3 bases (random bases from a fixed seed).
Per model and edit, three measures on the same sequences and edits:
  changed segments   no model: a segment of the clean parse is changed when the edited parse has no segment with the same
                     bases at the same position after the shift (benchmarks/parsing/locality.py), counted on each strand (the
                     reverse strand is segmented on its own, as the model does) and given per strand, with the reach
  per-base change    (1 - cos(u_i, u'_i)) / (1 - c0) for every base covered by a segment in both versions (deleted bases
                     skipped); u = per-base embedding with both strands averaged, c0 = mean cosine of two bases from different
                     sequences (20,000 pairs of bases at least 64 bp from the ends, first batch of 100 sequences), so the three
                     embedding spaces are comparable; by distance from the edit (the substituted base is at distance 0; after an
                     insertion or deletion, distance 1 is the base next to it on either side)
  read change        the read embedding of locus.load(...).embed (the mean of the per-base embeddings): (1 - cos(r, r')) /
                     (1 - c0_read), c0_read = mean cosine of the read embeddings of two different clean sequences (20,000 pairs)
Paired bootstrap over sequences (2,000 resamples) for BPE / LCP and fixed / LCP of the changed segments, the per-base change
within 10 bp and the read change.

    python benchmarks/embedding/stability.py --windows data/GRCh38.p14.acgt.w4096.s4096 --output results/embedding/stability.json

One A5000 GPU: about 40 minutes for 10,000 sequences. Repeated GPU runs differ in the last digits (floating-point atomics)."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks/parsing"))

import locus  # noqa: E402
from locality import changed_segments, segmenters  # noqa: E402
from locus.config import GRCH38_CHROMOSOMES  # noqa: E402
from locus.data.genome import GenomeDataset  # noqa: E402

MODELS = {"LCP": "locus-2m", "fixed": "twin-fixed-2m", "BPE": "twin-bpe-2m"}
LENGTH, SITE, EDGE, NEAR, FAR, BATCH, PAIRS = 2048, 1024, 64, 200, (500, 900), 100, 20000
EDITS = ["substitution", "insertion 1", "insertion 2", "insertion 3", "deletion 1", "deletion 2", "deletion 3"]
BINS = [(0, 1), (1, 4), (4, 7), (7, 11), (11, 16), (16, 21), (21, 31), (31, 51), (51, 101), (101, 201)]
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def edit(sequence: str, kind: str, rng: random.Random) -> str:
    if kind == "substitution":
        return sequence[:SITE] + rng.choice([b for b in "ACGT" if b != sequence[SITE]]) + sequence[SITE + 1:]
    n = int(kind.split()[1])
    if kind.startswith("insertion"):
        return sequence[:SITE] + "".join(rng.choice("ACGT") for _ in range(n)) + sequence[SITE:]
    return sequence[:SITE] + sequence[SITE + n:]


def follow(kind: str) -> list:
    """Per clean base x: (index in the edited sequence, distance from the edit), or None for a deleted base."""
    out = []
    n = 0 if kind == "substitution" else int(kind.split()[1])
    for x in range(LENGTH):
        if kind == "substitution":
            out.append((x, abs(x - SITE)))
        elif kind.startswith("insertion"):
            out.append((x, SITE - x) if x < SITE else (x + n, x - SITE + 1))
        else:
            out.append((x, SITE - x) if x < SITE else (None if x < SITE + n else (x - n, x - (SITE + n) + 1)))
    return out


def strand_edits(kind: str) -> list:
    """(position, shift) of the edit on the forward and on the reverse strand of the clean sequence."""
    n = 0 if kind == "substitution" else int(kind.split()[1])
    if kind == "substitution":
        return [(SITE, 0), (LENGTH - 1 - SITE, 0)]
    if kind.startswith("insertion"):
        return [(SITE, n), (LENGTH - SITE, n)]
    return [(SITE, -n), (LENGTH - SITE - n, -n)]


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)


def ratio_ci(a: np.ndarray, b: np.ndarray, boot: np.random.Generator) -> dict:
    """b / a with a paired bootstrap over sequences."""
    ratios = [b[s].mean() / a[s].mean() for s in (boot.integers(0, len(a), len(a)) for _ in range(2000))]
    return {"ratio": float(b.mean() / a.mean()), "ci95": [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", type=Path, default=ROOT / "data/GRCh38.p14.acgt.w4096.s4096", help="genome windows (scripts/prepare_genome.sh)")
    parser.add_argument("--sequences", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models-from", type=Path, help="use DIR/<name>/checkpoints/final.pt instead of the released models")
    parser.add_argument("--output", type=Path, default=ROOT / "results/embedding/stability.json")
    arguments = parser.parse_args()
    source = lambda name: str(arguments.models_from / name / "checkpoints/final.pt") if arguments.models_from else name

    reads = [read.sequence for read in GenomeDataset(arguments.windows, [GRCH38_CHROMOSOMES["1"]], 4096)]
    sequences = [read[1024:1024 + LENGTH] for read in reads[::max(1, len(reads) // arguments.sequences)][:arguments.sequences]]
    rng = random.Random(42)
    edited = {kind: [edit(s, kind, rng) for s in sequences] for kind in EDITS}
    maps = {}
    for kind in EDITS:  # clean index, edited index, distance of every base that survives the edit
        kept = [(x, *m) for x, m in enumerate(follow(kind)) if m is not None]
        maps[kind] = tuple(np.array(v) for v in zip(*kept))
    result = {"sequences": len(sequences), "length": LENGTH, "site": SITE, "models": MODELS, "segments": {}, "cos0": {}, "cos0_read": {},
              "curves": {}, "paired": {}}

    per_seq = {}  # (model, measure, edit) -> per-sequence values
    for name, parser_ in segmenters().items():  # changed segments, no model
        result["segments"][name] = {}
        for kind in EDITS:
            counts, reach = [], []
            for clean, changed in zip(sequences, edited[kind]):
                per_strand = []
                for (position, shift), a, b in zip(strand_edits(kind), (clean, reverse_complement(clean)), (changed, reverse_complement(changed))):
                    hit = changed_segments(a, b, parser_.bounds(a), parser_.bounds(b), position, kind, shift)
                    per_strand.append(len(hit))
                    reach.append(max((max(position - s, e - 1 - position) for s, e in hit), default=0))
                counts.append(np.mean(per_strand))
            counts, reach = np.array(counts), np.array(reach)
            per_seq[(name, "segments", kind)] = counts
            result["segments"][name][kind] = {"changed_segments_per_strand_mean": float(counts.mean()),
                                              "changed_segments_per_strand_median": float(np.median(counts)),
                                              "reach_median_bp": float(np.median(reach)), "reach_95th_bp": float(np.percentile(reach, 95)),
                                              "reach_max_bp": int(reach.max())}
            print(f"{name:6s} {kind:13s} changed segments per strand {counts.mean():7.2f}", flush=True)

    for name, model_name in MODELS.items():
        model = locus.load(source(model_name), device=arguments.device)
        sums = {k: np.zeros(NEAR + 1) for k in EDITS}
        counts = {k: np.zeros(NEAR + 1) for k in EDITS}
        far = {k: [] for k in EDITS}
        near10 = {k: [] for k in EDITS}
        read_cos = {k: [] for k in EDITS}
        clean_reads, cos0 = [], None
        for start in range(0, len(sequences), BATCH):
            raw = [np.asarray(v, dtype=np.float64) for v in model.embed(sequences[start:start + BATCH], per_base=True)]
            clean_reads += [v.mean(axis=0) for v in raw]  # = model.embed(...) without per_base
            clean = [unit(v) for v in raw]
            if cos0 is None:  # the first batch sets the per-base scale
                pool = np.concatenate([v[EDGE:-EDGE] for v in clean])
                pool = pool[np.linalg.norm(pool, axis=1) > 0]
                pick = np.random.default_rng(0)
                i, j = pick.integers(len(pool), size=PAIRS), pick.integers(len(pool), size=PAIRS)
                keep = (i // (LENGTH - 2 * EDGE)) != (j // (LENGTH - 2 * EDGE))
                cos0 = float(np.mean(np.sum(pool[i[keep]] * pool[j[keep]], axis=1)))
            for kind in EDITS:
                raw_edited = [np.asarray(v, dtype=np.float64) for v in model.embed(edited[kind][start:start + BATCH], per_base=True)]
                xs, ys, ds = maps[kind]
                for a_raw, b_raw, a in zip(raw, raw_edited, clean):
                    read_cos[kind].append(float(unit(a_raw.mean(axis=0)) @ unit(b_raw.mean(axis=0))))
                    va, vb = a[xs], unit(b_raw)[ys]
                    ok = va.any(axis=1) & vb.any(axis=1)  # bases covered by a segment in both
                    change = (1.0 - np.sum(va * vb, axis=1)) / (1.0 - cos0)
                    close = ok & (ds <= NEAR)
                    sums[kind] += np.bincount(ds[close], weights=change[close], minlength=NEAR + 1)[:NEAR + 1]
                    counts[kind] += np.bincount(ds[close], minlength=NEAR + 1)[:NEAR + 1]
                    near10[kind].append(float(change[ok & (ds <= 10)].mean()))
                    far[kind].append(float(change[ok & (ds >= FAR[0]) & (ds <= FAR[1])].mean()))
        reads_unit = unit(np.stack(clean_reads))
        pick = np.random.default_rng(1)
        i, j = pick.integers(len(reads_unit), size=PAIRS), pick.integers(len(reads_unit), size=PAIRS)
        keep = i != j
        cos0_read = float(np.mean(np.sum(reads_unit[i[keep]] * reads_unit[j[keep]], axis=1)))
        result["cos0"][name], result["cos0_read"][name] = cos0, cos0_read
        result["curves"][name] = {}
        for kind in EDITS:
            curve = sums[kind] / np.maximum(counts[kind], 1)
            first = 0 if kind == "substitution" else 1
            bins = {f"{lo}-{hi - 1}": float(sums[kind][max(lo, first):hi].sum() / max(counts[kind][max(lo, first):hi].sum(), 1))
                    for lo, hi in BINS if hi > first}
            below = next((d for d in range(first, NEAR + 1) if (curve[d:] <= 0.01).all()), None)
            read_change = (1.0 - np.array(read_cos[kind])) / (1.0 - cos0_read)
            per_seq[(name, "near10", kind)], per_seq[(name, "read", kind)] = np.array(near10[kind]), read_change
            result["curves"][name][kind] = {"normalized_change_by_distance": curve.tolist(), "bins": bins, "far_field": float(np.mean(far[kind])),
                                            "stays_at_or_below_0.01_from_bp": below, "within_10bp": float(np.mean(near10[kind])),
                                            "read_change_mean": float(read_change.mean()), "read_change_median": float(np.median(read_change)),
                                            "read_change_95th": float(np.percentile(read_change, 95)),
                                            "read_cosine_mean": float(np.mean(read_cos[kind]))}
            print(f"{name:6s} {kind:13s} within 10 bp {np.mean(near10[kind]):.4f} | far {np.mean(far[kind]):.4f} | <=0.01 from {below} bp | "
                  f"read change {read_change.mean():.6f}", flush=True)
        del model
    boot = np.random.default_rng(0)
    for kind in EDITS:
        result["paired"][kind] = {}
        for measure, label in (("segments", "changed segments"), ("near10", "per-base change within 10 bp"), ("read", "read change")):
            for other in ("BPE", "fixed"):
                result["paired"][kind][f"{other}/LCP {label}"] = ratio_ci(per_seq[("LCP", measure, kind)], per_seq[(other, measure, kind)], boot)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", arguments.output)


if __name__ == "__main__":
    main()
