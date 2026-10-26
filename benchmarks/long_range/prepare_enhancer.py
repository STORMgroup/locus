"""Enhancer task of the Genomics Long-Range Benchmark (Kao et al. 2024, InstaDeepAI/genomics-long-range-benchmark) as
fine-tuning records at several input lengths around the same elements, for the context-length ablation.

Each benchmark row is a 200 bp window labelled 1 when it overlaps an ENCODE SCREEN enhancer, with the benchmark's
train / test split (test = chromosomes 8 and 9). As the benchmark's own loader does, `length` bases of GRCh38 centred on
the window are cut from the genome (RefSeq GRCh38.p14 FASTA; primary chromosomes are named chrN from their FASTA
headers) and upper-cased. An element is dropped when its window runs off the chromosome or contains an N at any of the
requested lengths, so every length uses the same elements and results at different lengths are paired. The benchmark's
`_subset` table is sampled further to `--train` / `--test` rows per split (seeded, each label in proportion).

    python benchmarks/long_range/prepare_enhancer.py --fasta data/GRCh38.p14.fasta.gz --output data/benchmarks/long_range

writes <output>/lrb_enhancer_4k/{train,test}.jsonl.gz, <output>/lrb_enhancer_16k/{train,test}.jsonl.gz and
<output>/datasets.json (source revision, counts and dropped elements); fine-tune with scripts/finetune.py and
configs/finetune/long-range-enhancer-ep20.yaml (or long-range-enhancer-ep20-constant.yaml)."""
import argparse
import csv
import gzip
import io
import json
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from locus.finetune.data import SequenceRecord, write_records  # noqa: E402

REPOSITORY = "InstaDeepAI/genomics-long-range-benchmark"
REVISION = "b56f68ee30c01daf8f80a8d0b8a47f84199de08e"
TASK = "enhancer"
REFSEQ_HEADER = re.compile(r"^>(\S+) .*chromosome (\w+), GRCh38\.p14 Primary Assembly")


def benchmark_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{path}"


def read_table(path: str) -> list[dict]:
    """Rows of one CSV file of the benchmark repository at the pinned revision."""
    with urllib.request.urlopen(benchmark_url(path)) as response:
        return list(csv.DictReader(io.TextIOWrapper(response, encoding="utf-8")))


def sample_split(rows: list[dict], split: str, count: int, seed: int) -> list[dict]:
    """`count` rows of one split, each label in proportion to its share of the split, sorted by position."""
    by_label = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            by_label[row["label"]].append(row)
    total = sum(len(group) for group in by_label.values())
    generator, chosen = random.Random(seed), []
    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda row: (row["CHROM"], int(row["START"])))
        take = round(count * len(group) / total)
        chosen += generator.sample(group, min(take, len(group)))
    return sorted(chosen, key=lambda row: (row["CHROM"], int(row["START"])))


def grch38_chromosomes(fasta: Path, wanted: set):
    """(chrN, upper-case sequence) of the GRCh38.p14 primary chromosomes in `wanted`, one at a time, in file order."""
    name, parts = None, []
    with gzip.open(fasta, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if name in wanted:
                    yield name, "".join(parts).upper()
                match = REFSEQ_HEADER.match(line)
                name = f"chr{match.group(2)}" if match else None
                parts = []
            elif name in wanted:
                parts.append(line.rstrip())
    if name in wanted:
        yield name, "".join(parts).upper()


def centred_window(sequence: str, start: int, end: int, length: int):
    """The benchmark loader's window: `length` bases around the 0-based [start, end), or None off the chromosome."""
    pad = (length - (end - start)) // 2
    left, right = start - pad, end + pad + (length % 2)
    if left < 0 or right >= len(sequence):
        return None
    return sequence[left:right]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fasta", type=Path, default=ROOT / "data/GRCh38.p14.fasta.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "data/benchmarks/long_range")
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384])
    parser.add_argument("--train", type=int, default=10000, help="training rows sampled from the benchmark table")
    parser.add_argument("--test", type=int, default=10000, help="test rows sampled from the benchmark table")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    source = f"regulatory_elements/{TASK}_dataset_subset.csv"
    rows = read_table(source)
    picked = {"train": sample_split(rows, "train", arguments.train, arguments.seed),
              "test": sample_split(rows, "test", arguments.test, arguments.seed)}
    wanted = {row["CHROM"] for split_rows in picked.values() for row in split_rows}
    windows = {length: {split: [] for split in picked} for length in arguments.lengths}
    dropped = Counter()
    for chromosome, sequence in grch38_chromosomes(arguments.fasta, wanted):
        for split, split_rows in picked.items():
            for row in split_rows:
                if row["CHROM"] != chromosome:
                    continue
                # START / STOP are 1-based; the loader cuts around the 0-based [START - 1, STOP - 1)
                start, end = int(row["START"]) - 1, int(row["STOP"]) - 1
                cut = {length: centred_window(sequence, start, end, length) for length in arguments.lengths}
                if any(window is None for window in cut.values()):
                    dropped["off_chromosome"] += 1
                    continue
                if any(set(window) - set("ACGT") for window in cut.values()):
                    dropped["contains_n"] += 1
                    continue
                for length, window in cut.items():
                    windows[length][split].append((row, window))
        print(f"{chromosome}: done", flush=True)

    summary_path = arguments.output / "datasets.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    for length in arguments.lengths:
        task = f"lrb_{TASK}_{length // 1024}k"
        entry = {"repo": REPOSITORY, "revision": REVISION, "source_csv": source, "sequence_length": length, "seed": arguments.seed,
                 "dropped_elements": dict(dropped)}
        for split, items in windows[length].items():
            records = [SequenceRecord(name=f"{task}:{split}:{row['CHROM']}:{row['START']}-{row['STOP']}", sequence=window,
                                      label=int(row["label"]), task=task, chromosome=row["CHROM"]) for row, window in items]
            write_records(arguments.output / task / f"{split}.jsonl.gz", records)
            entry[split] = {"count": len(records), "classes": dict(sorted(Counter(record.label for record in records).items())),
                            "chromosomes": sorted({record.chromosome for record in records})}
        summary[task] = entry
        print(task, json.dumps({key: entry[key] for key in ("train", "test", "dropped_elements")}), flush=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
