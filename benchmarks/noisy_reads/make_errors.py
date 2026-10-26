"""NT18 test sets with nanopore-like sequencing errors at a fixed rate per base.

Every base of a clean test sequence starts an error event with probability RATE (in percent): a substitution, a
deletion of 1-3 bases, or an insertion of 1-3 bases (event model in noisy_sets.py). Errors fall anywhere in the read,
the bases that determine the label included, as in real reads. Labels are unchanged. Models are fine-tuned on the
clean training set and scored on these test sets (scripts/finetune.py --extra-test, or benchmarks/noisy_reads/score.py
for saved fine-tuned weights).

    python benchmarks/noisy_reads/make_errors.py --rates 1 3

reads <data-root>/<task>_revised/test.jsonl.gz and writes <output-root>/<task>_ont<RATE>_revised/test.jsonl.gz and
metadata.json (error counts, realized events per base) for every NT18 task, or for the tasks given with --tasks, and the
further error draws <task>_ont<RATE>-d<k>_revised, k = 1 .. --draws - 1 (noisy_sets.py). The random number generator of
each set is seeded with the string "42-<task>-test-ont<RATE>" (draw 0) or "42-<task>-test-ont<RATE>-draw<k>", so every
set can be regenerated on its own.
"""
import argparse
import json
import random
import sys
from pathlib import Path

from noisy_sets import (DEFAULT_DATA_ROOT, DRAWS, EVENT_MIX, HOMOPOLYMER_PROBABILITY, INDEL_LENGTHS, ROOT, clean_task, draw_tag,
                        pick, rate_tag, variant_task)

sys.path.insert(0, str(ROOT))

from locus.finetune.data import NT18_TASKS, SequenceRecord, load_records, sha256_file, write_records  # noqa: E402


def add_errors(sequence: str, rate: float, rng: random.Random) -> tuple[str, dict]:
    """(read with errors, number of events of each kind)."""
    read, position, counts = [], 0, {kind: 0 for kind, _ in EVENT_MIX}
    while position < len(sequence):
        if rng.random() >= rate:
            read.append(sequence[position])
            position += 1
            continue
        kind = pick(rng.random(), EVENT_MIX)
        counts[kind] += 1
        if kind == "substitution":
            read.append(rng.choice([base for base in "ACGT" if base != sequence[position]]))
            position += 1
        elif kind == "deletion":
            position += pick(rng.random(), INDEL_LENGTHS)
        else:
            length = pick(rng.random(), INDEL_LENGTHS)
            if rng.random() < HOMOPOLYMER_PROBABILITY:
                read.append(sequence[position] * length)
            else:
                read.append("".join(rng.choice("ACGT") for _ in range(length)))
            read.append(sequence[position])
            position += 1
    return "".join(read), counts


def make_set(data_root: Path, output_root: Path, task: str, rate_percent: float, draw: int = 0) -> dict:
    base = f"ont{rate_tag(rate_percent)}"
    tag = draw_tag(base, draw)
    seed = f"42-{task}-test-{base}" + (f"-draw{draw}" if draw else "")
    rate = rate_percent / 100
    rng = random.Random(seed)
    target = variant_task(task, tag)
    records = load_records(data_root / clean_task(task) / "test.jsonl.gz")
    reads, totals = [], {kind: 0 for kind, _ in EVENT_MIX}
    for record in records:
        sequence, counts = add_errors(record.sequence, rate, rng)
        for kind, count in counts.items():
            totals[kind] += count
        reads.append(SequenceRecord(name=f"{record.name}:{tag}", sequence=sequence, label=record.label, task=target,
                                    chromosome=record.chromosome))
    path = output_root / target / "test.jsonl.gz"
    write_records(path, reads)
    bases = sum(len(record.sequence) for record in records)
    metadata = {"source_task": clean_task(task), "split": "test", "rate_per_base": rate, "mix": dict(EVENT_MIX),
                "indel_lengths": dict(INDEL_LENGTHS), "insertion_homopolymer_probability": HOMOPOLYMER_PROBABILITY,
                "seed": seed, "count": len(reads), "error_events": totals, "events_per_base": sum(totals.values()) / bases,
                "sha256": sha256_file(path), **({"draw": draw} if draw else {})}
    (output_root / target / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{target}: {len(reads)} reads, events {totals}, events per base {metadata['events_per_base']:.4f}", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rates", type=float, nargs="+", default=[1.0, 3.0], help="error rates in percent per base")
    parser.add_argument("--draws", type=int, default=DRAWS, help=f"independent error draws per set (default {DRAWS})")
    parser.add_argument("--tasks", nargs="+", choices=NT18_TASKS, default=NT18_TASKS, metavar="TASK",
                        help="NT18 tasks without the _revised suffix (default: all 18)")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="directory of the clean task directories")
    parser.add_argument("--output-root", type=Path, help="directory for the new test sets (default: --data-root)")
    arguments = parser.parse_args()
    for task in arguments.tasks:
        for rate_percent in arguments.rates:
            for draw in range(arguments.draws):
                make_set(arguments.data_root, arguments.output_root or arguments.data_root, task, rate_percent, draw)


if __name__ == "__main__":
    main()
