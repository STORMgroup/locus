"""Download the revised Nucleotide Transformer benchmark (18 tasks) and write it as labelled sequence records.

    python benchmarks/nt18/prepare.py
    python benchmarks/nt18/prepare.py --out data/benchmarks/nt18

The source is the Hugging Face dataset InstaDeepAI/nucleotide_transformer_downstream_tasks_revised, pinned to one
commit. Tasks are named after the benchmark task with the suffix _revised (H3K27ac_revised is the H3K27ac task of the
revised benchmark). For every task the script writes <out>/<task>/train.jsonl.gz and test.jsonl.gz, one SequenceRecord
per line (locus/finetune/data.py), and <out>/<task>/metadata.json with the size, class and length distribution,
ambiguous bases, chromosomes and file checksum of each split, plus the chromosomes shared by both splits and the number
of distinct sequences that occur in both splits on either strand.

Sequences are upper-cased. A base other than A, C, G, T or N, a length or label outside the task's range, a task
column that names another task, or a split size different from the published one aborts preparation. The chromosome of
a record is taken from its name (chrN:start-end) and is "unknown" when the name has no chromosome prefix."""
import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from locus.finetune.data import NT18_TASKS, SequenceRecord, get_task, sha256_file, write_records  # noqa: E402

DATASET_ID = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
DATASET_REVISION = "851f9946252e90c665cdb3cc3eedb78f1f26197c"
SUFFIX = "_revised"
ALLOWED_BASES = frozenset("ACGTN")
CHROMOSOME_PATTERN = re.compile(r"^(chr(?:[0-9]+|X|Y)):")


@dataclass(frozen=True)
class TaskSpec:
    """Expected shape of one task. primary_metric is the metric of the original benchmark definition; the paper reports
    MCC for every task (locus.finetune.data.TASKS)."""
    name: str
    category: str
    min_sequence_length: int
    max_sequence_length: int
    classes: int
    train_count: int
    test_count: int
    primary_metric: str


TASK_SPECS = {spec.name: spec for spec in (
    TaskSpec("H2AFZ_revised", "histone_revised", 1, 1000, 2, 30000, 3000, "mcc"),
    TaskSpec("H3K27ac_revised", "histone_revised", 1, 1000, 2, 30000, 1616, "mcc"),
    TaskSpec("H3K27me3_revised", "histone_revised", 1, 1000, 2, 30000, 3000, "mcc"),
    TaskSpec("H3K36me3_revised", "histone_revised", 1, 1000, 2, 30000, 3000, "mcc"),
    TaskSpec("H3K4me1_revised", "histone_revised", 1, 1000, 2, 30000, 3000, "mcc"),
    TaskSpec("H3K4me2_revised", "histone_revised", 1, 1000, 2, 30000, 2138, "mcc"),
    TaskSpec("H3K4me3_revised", "histone_revised", 1, 1000, 2, 17468, 776, "mcc"),
    TaskSpec("H3K9ac_revised", "histone_revised", 1, 1000, 2, 23274, 1004, "mcc"),
    TaskSpec("H3K9me3_revised", "histone_revised", 1, 1000, 2, 27438, 850, "mcc"),
    TaskSpec("H4K20me1_revised", "histone_revised", 1, 1000, 2, 30000, 2270, "mcc"),
    TaskSpec("enhancers_revised", "enhancer_revised", 1, 400, 2, 30000, 3000, "mcc"),
    TaskSpec("enhancers_types_revised", "enhancer_revised", 1, 400, 3, 30000, 3000, "mcc"),
    TaskSpec("promoter_all_revised", "promoter_revised", 1, 300, 2, 30000, 1584, "f1"),
    TaskSpec("promoter_no_tata_revised", "promoter_revised", 1, 300, 2, 30000, 1372, "f1"),
    TaskSpec("promoter_tata_revised", "promoter_revised", 1, 300, 2, 5062, 212, "f1"),
    TaskSpec("splice_sites_acceptors_revised", "splice_revised", 1, 600, 2, 30000, 3000, "f1"),
    TaskSpec("splice_sites_all_revised", "splice_revised", 1, 600, 3, 30000, 3000, "accuracy"),
    TaskSpec("splice_sites_donors_revised", "splice_revised", 1, 600, 2, 30000, 3000, "f1"),
)}


@dataclass
class SplitStatistics:
    count: int = 0
    class_counts: Counter = field(default_factory=Counter)
    length_counts: Counter = field(default_factory=Counter)
    ambiguous_n_bases: int = 0
    records_with_ambiguous_n: int = 0
    chromosomes: set = field(default_factory=set)
    # Lexicographically smaller of each sequence and its reverse complement, to count train/test overlap.
    canonical_sequences: set = field(default_factory=set)


def chromosome_from_name(name: str) -> str:
    match = CHROMOSOME_PATTERN.match(name)
    return match.group(1) if match else "unknown"


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1].upper()


def read_split(parquet_path: Path, spec: TaskSpec, split: str, statistics: SplitStatistics):
    """Validated records of one split, in file order; `statistics` is updated as records are produced."""
    import pyarrow.parquet as parquet

    remote_task = spec.name[:-len(SUFFIX)]
    parquet_file = parquet.ParquetFile(parquet_path)
    available = set(parquet_file.schema_arrow.names)
    missing = {"sequence", "name", "label"} - available
    if missing:
        raise ValueError(f"{parquet_path} lacks required columns: {sorted(missing)}")
    columns = ["sequence", "name", "label"] + (["task"] if "task" in available else [])
    for batch in parquet_file.iter_batches(batch_size=8192, columns=columns):
        for row in batch.to_pylist():
            if "task" in row and str(row["task"]) != remote_task:
                raise ValueError(f"Task column mismatch in {parquet_path}: {row['task']!r}")
            sequence, name, label = str(row["sequence"]).upper(), str(row["name"]), int(row["label"])
            invalid = set(sequence) - ALLOWED_BASES
            if invalid:
                raise ValueError(f"{spec.name}/{split} {name} contains unsupported bases: {''.join(sorted(invalid))}")
            if not spec.min_sequence_length <= len(sequence) <= spec.max_sequence_length:
                raise ValueError(f"{spec.name}/{split} {name} has length {len(sequence)}, expected "
                                 f"{spec.min_sequence_length}..{spec.max_sequence_length}")
            if not 0 <= label < spec.classes:
                raise ValueError(f"Invalid label {label} for {spec.name}: {name}")
            record = SequenceRecord(name=name, sequence=sequence, label=label, task=spec.name,
                                    chromosome=chromosome_from_name(name))
            ambiguous = sequence.count("N")
            statistics.count += 1
            statistics.class_counts[label] += 1
            statistics.length_counts[len(sequence)] += 1
            statistics.ambiguous_n_bases += ambiguous
            statistics.records_with_ambiguous_n += int(ambiguous > 0)
            statistics.chromosomes.add(record.chromosome)
            statistics.canonical_sequences.add(min(sequence, reverse_complement(sequence)))
            yield record


def prepare_task(spec: TaskSpec, output_root: Path) -> None:
    from huggingface_hub import hf_hub_download

    remote_task = spec.name[:-len(SUFFIX)]
    task_dir = output_root / spec.name
    statistics = {}
    for split in ("train", "test"):
        parquet_path = Path(hf_hub_download(repo_id=DATASET_ID, filename=f"{remote_task}/{split}.parquet",
                                            repo_type="dataset", revision=DATASET_REVISION))
        statistics[split] = SplitStatistics()
        write_records(task_dir / f"{split}.jsonl.gz", read_split(parquet_path, spec, split, statistics[split]))

    expected = {"train": spec.train_count, "test": spec.test_count}
    found = {split: statistics[split].count for split in ("train", "test")}
    if found != expected:
        raise ValueError(f"Split sizes of {spec.name} differ from the published ones: "
                         f"expected {expected}, found {found}")
    chromosome_overlap = sorted(statistics["train"].chromosomes & statistics["test"].chromosomes)
    sequence_overlap = len(statistics["train"].canonical_sequences & statistics["test"].canonical_sequences)
    metadata = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_format": "pinned-per-task-parquet-revised",
        "task": spec.name,
        "remote_task": remote_task,
        "task_spec": asdict(spec),
        "splits": {
            split: {
                "count": values.count,
                "class_counts": dict(sorted(values.class_counts.items())),
                "length_counts": dict(sorted(values.length_counts.items())),
                "ambiguous_n_bases": values.ambiguous_n_bases,
                "records_with_ambiguous_n": values.records_with_ambiguous_n,
                "chromosomes": sorted(values.chromosomes),
                "sha256": sha256_file(task_dir / f"{split}.jsonl.gz"),
            }
            for split, values in statistics.items()
        },
        "train_test_chromosome_overlap": chromosome_overlap,
        "train_test_sequence_or_rc_overlap": sequence_overlap,
    }
    (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{spec.name}: train={found['train']} test={found['test']} "
          f"lengths={dict(sorted(statistics['train'].length_counts.items()))} "
          f"sequence_or_rc_overlap={sequence_overlap} train_test_chromosome_overlap={chromosome_overlap}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "benchmarks" / "nt18")
    arguments = parser.parse_args()

    names = [f"{name}{SUFFIX}" for name in NT18_TASKS]
    if set(names) != set(TASK_SPECS):
        raise RuntimeError("The task specifications do not match locus.finetune.data.NT18_TASKS")
    for name in names:
        if TASK_SPECS[name].classes != get_task(name).classes:
            raise RuntimeError(f"Class count of {name} differs from locus.finetune.data.TASKS")
    for name in names:
        prepare_task(TASK_SPECS[name], arguments.out)
    print(f"Prepared {len(names)} tasks of {DATASET_ID} at revision {DATASET_REVISION} in {arguments.out}")


if __name__ == "__main__":
    main()
