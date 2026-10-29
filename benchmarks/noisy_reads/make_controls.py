"""Control test sets for the read-error study: substitutions only, and errors outside a protected label window.

    <task>_sub<RATE>_revised   the event positions of the nanopore-like model (noisy_sets.py), every event a single-base
                               substitution: no insertions or deletions, so the segment grid of the read is not shifted
    <task>_ontw<RATE>_revised  the full nanopore-like model, but no edit inside the task's protected window: the bases
                               that carry the label stay intact, and a change in accuracy can only come from edits
                               elsewhere in the read

Each sequence receives one error schedule, drawn per base: a uniform number u, an event kind, an indel length, the
homopolymer choice, three random bases for an insertion and a substitute base. At rate R the events with u < R are
applied, so the events at 1% are a subset of those at 3%, and both controls place their events at the same positions.
A deletion that would reach into the protected window is not applied.

The protected windows are fixed (PROTECTED_WINDOWS): [200, 400) for the NT18 splice-site tasks (600 bp, the junction
at the centre, so no edit falls within 100 bp of it) and [100, 300) for gue_splice (400 bp, junction between positions 199
and 200; the same 100 bp on each side). The other tasks have no annotated label position; promoter_tata gets [0, 160)
and the rest the central third of the sequence, and only the splice-site windows enter the result tables
(scripts/collect_results.py).

    python benchmarks/noisy_reads/make_controls.py

reads <data-root>/<task>_revised/test.jsonl.gz and writes the four sets sub1, sub3, ontw1, ontw3 (test.jsonl.gz and
metadata.json) of every NT18 task, or of the tasks and controls given with --tasks and --controls, and their further
error draws (sub1-d<k> and so on, k = 1 .. --draws - 1; noisy_sets.py). The random number generator of a task is seeded
with the string "42-<task>-test-controls" (draw 0) or "42-<task>-test-controls-draw<k>"; the four sets of one draw share
its error schedules.
"""
import argparse
import json
import random
import sys
from pathlib import Path

from noisy_sets import (DEFAULT_DATA_ROOT, DRAWS, OTHER_TASKS, EVENT_MIX, HOMOPOLYMER_PROBABILITY, INDEL_LENGTHS, ROOT, clean_task, draw_tag, pick,
                        rate_tag, variant_task)

sys.path.insert(0, str(ROOT))

from locus.finetune.data import NT18_TASKS, SequenceRecord, load_records, write_records  # noqa: E402

_CENTRAL_THIRD = {**{task: (333, 667) for task in ("H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3",
                                                   "H3K9ac", "H3K9me3", "H4K20me1")},
                  "enhancers": (133, 267), "enhancers_types": (133, 267), "promoter_all": (100, 200), "promoter_no_tata": (100, 200)}
PROTECTED_WINDOWS = {"splice_sites_acceptors": (200, 400), "splice_sites_all": (200, 400), "splice_sites_donors": (200, 400),
                     "promoter_tata": (0, 160), "gue_splice": (100, 300), **_CENTRAL_THIRD}
CONTROLS = ("sub", "ontw")


def error_schedule(sequence: str, rng: random.Random) -> list[tuple]:
    """Per base: (u, event kind, indel length, homopolymer insertion, three random bases, substitute base)."""
    return [(rng.random(), pick(rng.random(), EVENT_MIX), pick(rng.random(), INDEL_LENGTHS), rng.random() < HOMOPOLYMER_PROBABILITY,
             "".join(rng.choice("ACGT") for _ in range(3)), rng.choice([other for other in "ACGT" if other != base]))
            for base in sequence]


def apply_schedule(sequence: str, schedule: list[tuple], rate: float, window=None, substitutions_only: bool = False) -> tuple[str, int]:
    """(read, number of applied events) for the events with u < rate, none of them inside `window` [start, end)."""
    def inside(position: int) -> bool:
        return window is not None and window[0] <= position < window[1]

    read, position, events = [], 0, 0
    while position < len(sequence):
        u, kind, length, homopolymer, random_bases, substitute = schedule[position]
        if u >= rate or inside(position):
            read.append(sequence[position])
            position += 1
            continue
        if substitutions_only or kind == "substitution":
            read.append(substitute)
            position += 1
        elif kind == "deletion":
            length = min(length, len(sequence) - position)
            if any(inside(deleted) for deleted in range(position, position + length)):
                read.append(sequence[position])
                position += 1
                continue
            position += length
        else:
            read.append(sequence[position] * length if homopolymer else random_bases[:length])
            read.append(sequence[position])
            position += 1
        events += 1
    return "".join(read), events


def make_controls(data_root: Path, output_root: Path, task: str, controls: list, rates: list, draw: int = 0) -> None:
    seed = f"42-{task}-test-controls" + (f"-draw{draw}" if draw else "")
    rng = random.Random(seed)
    records = load_records(data_root / clean_task(task) / "test.jsonl.gz")
    variants = [(control, rate_percent) for control in CONTROLS if control in controls for rate_percent in rates]
    reads = {variant: [] for variant in variants}
    events = {variant: 0 for variant in variants}
    for record in records:
        schedule = error_schedule(record.sequence, rng)
        for control, rate_percent in variants:
            tag = draw_tag(f"{control}{rate_tag(rate_percent)}", draw)
            window = PROTECTED_WINDOWS[task] if control == "ontw" else None
            sequence, count = apply_schedule(record.sequence, schedule, rate_percent / 100, window, substitutions_only=control == "sub")
            reads[control, rate_percent].append(SequenceRecord(name=f"{record.name}:{tag}", sequence=sequence, label=record.label,
                                                               task=variant_task(task, tag), chromosome=record.chromosome))
            events[control, rate_percent] += count
    bases = sum(len(record.sequence) for record in records)
    for control, rate_percent in variants:
        target = variant_task(task, draw_tag(f"{control}{rate_tag(rate_percent)}", draw))
        write_records(output_root / target / "test.jsonl.gz", reads[control, rate_percent])
        window = PROTECTED_WINDOWS[task] if control == "ontw" else None
        metadata = {"source_task": clean_task(task), "control": control, "rate_per_base": rate_percent / 100, "protected_window": window,
                    "substitutions_only": control == "sub", "events_per_base": events[control, rate_percent] / bases, "seed": seed,
                    "count": len(records), **({"draw": draw} if draw else {})}
        (output_root / target / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"{target}: {len(records)} reads, events per base {metadata['events_per_base']:.4f}"
              + (f", protected window [{window[0]}, {window[1]})" if window else ""), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controls", nargs="+", choices=CONTROLS, default=list(CONTROLS))
    parser.add_argument("--rates", type=float, nargs="+", default=[1.0, 3.0], help="error rates in percent per base")
    parser.add_argument("--draws", type=int, default=DRAWS, help=f"independent error draws per set (default {DRAWS})")
    parser.add_argument("--tasks", nargs="+", choices=NT18_TASKS + list(OTHER_TASKS), default=NT18_TASKS, metavar="TASK",
                        help="NT18 tasks without the _revised suffix, or gue_splice (default: the 18 NT18 tasks)")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="directory of the clean task directories")
    parser.add_argument("--output-root", type=Path, help="directory for the new test sets (default: --data-root)")
    arguments = parser.parse_args()
    for task in arguments.tasks:
        for draw in range(arguments.draws):
            make_controls(arguments.data_root, arguments.output_root or arguments.data_root, task, arguments.controls, arguments.rates, draw)


if __name__ == "__main__":
    main()
