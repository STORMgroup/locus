"""Download the GUE splice-site task (DNABERT-2; Zhou et al., 2024) and write it as labelled sequence records.

    python benchmarks/gue/prepare.py
    python benchmarks/gue/prepare.py --out data/benchmarks/gue --nt18 data/benchmarks/nt18

The task is GUE/splice/reconstructed (400 bp human sequences; 0 = acceptor, 1 = donor, 2 = neither; the exon-intron
junction of a splice site lies between positions 199 and 200), read from the Hugging Face dataset leannmlindsey/GUE pinned
to one commit (byte-identical to the files of the GUE archive released with DNABERT-2). The training file is the pool of
the 10-fold validation split (as for NT18), the test file the test set; GUE's dev file is not used. The script writes
<out>/gue_splice/train.jsonl.gz and test.jsonl.gz, one SequenceRecord per line (locus/finetune/data.py) named
gue_splice:<split>:<row> with chromosome "unknown", and <out>/gue_splice/metadata.json with the size, class counts and
checksum of each split.

The records carry no coordinates, so overlap with the NT18 splice tasks is found by sequence: a test record shares its
splice site with NT18 when the 32 bases around its junction (positions 184-215), on either strand, occur in a record of
splice_sites_all, _acceptors or _donors (train or test). metadata.json lists these test rows, so results can also be
reported on the test records independent of NT18.

Sequences are upper-cased; a base other than A, C, G, T or N, a length other than 400, a label outside 0-2 or a split
size different from the published one aborts preparation."""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from locus.alphabet import reverse_complement  # noqa: E402
from locus.finetune.data import SequenceRecord, get_task, load_records, sha256_file, write_records  # noqa: E402

DATASET_ID = "leannmlindsey/GUE"
DATASET_REVISION = "240df51a2cd2c0fb1d88efffaf86e8d1e477f932"
REMOTE_DIRECTORY = "GUE/splice_reconstructed"
TASK = "gue_splice"
LENGTH = 400
COUNTS = {"train": 36496, "test": 4562}
JUNCTION_CORE = (184, 216)  # 32 bases around the junction between positions 199 and 200
NT18_SPLICE = ["splice_sites_all", "splice_sites_acceptors", "splice_sites_donors"]
ALLOWED_BASES = frozenset("ACGTN")


def read_split(path: Path, split: str) -> list:
    records = []
    with open(path, newline="") as handle:
        for row, line in enumerate(csv.DictReader(handle)):
            sequence, label = line["sequence"].upper(), int(line["label"])
            if set(sequence) - ALLOWED_BASES or len(sequence) != LENGTH or not 0 <= label < get_task(TASK).classes:
                raise ValueError(f"{path} row {row}: invalid sequence or label")
            records.append(SequenceRecord(name=f"{TASK}:{split}:{row}", sequence=sequence, label=label, task=TASK, chromosome="unknown"))
    if len(records) != COUNTS[split]:
        raise ValueError(f"{path}: {len(records)} records, expected {COUNTS[split]}")
    return records


def nt18_shared_rows(test: list, nt18: Path) -> list:
    """Test rows whose junction core occurs, on either strand, in an NT18 splice record."""
    start, end = JUNCTION_CORE
    width = end - start
    cores = {}
    for row, record in enumerate(test):
        core = record.sequence[start:end]
        cores.setdefault(core, set()).add(row)
        cores.setdefault(reverse_complement(core), set()).add(row)
    shared = set()
    for task in NT18_SPLICE:
        for split in ("train", "test"):
            for record in load_records(nt18 / f"{task}_revised" / f"{split}.jsonl.gz"):
                sequence = record.sequence
                for offset in range(len(sequence) - width + 1):
                    rows = cores.get(sequence[offset:offset + width])
                    if rows:
                        shared |= rows
    return sorted(shared)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "benchmarks" / "gue")
    parser.add_argument("--nt18", type=Path, default=ROOT / "data" / "benchmarks" / "nt18",
                        help="prepared NT18 tasks (benchmarks/nt18/prepare.py), for the overlap with the NT18 splice tasks")
    arguments = parser.parse_args()
    from huggingface_hub import hf_hub_download

    task_dir = arguments.out / TASK
    splits, metadata = {}, {"dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION, "remote_directory": REMOTE_DIRECTORY,
                            "task": TASK, "length": LENGTH, "junction_between": [199, 200], "splits": {}}
    for split in ("train", "test"):
        path = Path(hf_hub_download(repo_id=DATASET_ID, filename=f"{REMOTE_DIRECTORY}/{split}.csv", repo_type="dataset",
                                    revision=DATASET_REVISION))
        splits[split] = read_split(path, split)
        write_records(task_dir / f"{split}.jsonl.gz", splits[split])
        metadata["splits"][split] = {"count": len(splits[split]), "source_sha256": sha256_file(path),
                                     "class_counts": dict(sorted(Counter(r.label for r in splits[split]).items())),
                                     "sha256": sha256_file(task_dir / f"{split}.jsonl.gz")}
    canonical = lambda records: {min(r.sequence, reverse_complement(r.sequence)) for r in records}
    metadata["train_test_sequence_or_rc_overlap"] = len(canonical(splits["train"]) & canonical(splits["test"]))
    metadata["test_rows_sharing_nt18_splice_site"] = nt18_shared_rows(splits["test"], arguments.nt18)
    (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{TASK}: train={len(splits['train'])} test={len(splits['test'])} "
          f"test rows sharing an NT18 splice site: {len(metadata['test_rows_sharing_nt18_splice_site'])}", flush=True)


if __name__ == "__main__":
    main()
