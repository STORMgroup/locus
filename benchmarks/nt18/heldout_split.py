"""NT18 with a test set on chromosomes the pretrained models never saw.

The test sets of the revised benchmark lie on chromosomes 20 and 21, which are part of the pretraining chromosomes (3-22;
locus.config.DataConfig.train_sources). This script moves every training example on a chromosome outside the pretraining
chromosomes (1, 2, X, Y) into a new test set, so that a model fine-tuned on the rest is tested on sequence that it saw
neither in pretraining nor in fine-tuning. The official test set is scored alongside (scripts/reproduce.py, plan
heldout), and the two scores of the same selected checkpoint show whether exposure to the test chromosomes during
pretraining raises the official scores.

    python benchmarks/nt18/heldout_split.py --data-root data/benchmarks/nt18 --out data/benchmarks/nt18_heldout

writes <out>/<task>/train.jsonl.gz (official training set without the held-out chromosomes), test.jsonl.gz (the
removed examples) and metadata.json (sizes, labels and chromosomes of both splits, checksums) for each of the 18 tasks.
The task names are those of the official benchmark, so the same fine-tuning recipe and task definitions apply."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from locus.config import GRCH38_CHROMOSOMES, DataConfig  # noqa: E402
from locus.finetune.data import NT18_TASKS, load_records, sha256_file, write_records  # noqa: E402

PRETRAINING_CHROMOSOMES = frozenset(f"chr{name}" for name, accession in GRCH38_CHROMOSOMES.items()
                                    if accession in DataConfig().train_sources)


def split_task(data_root: Path, out_root: Path, task: str) -> dict:
    records = load_records(data_root / task / "train.jsonl.gz")
    train = [record for record in records if record.chromosome in PRETRAINING_CHROMOSOMES]
    test = [record for record in records if record.chromosome not in PRETRAINING_CHROMOSOMES]
    task_dir = out_root / task
    write_records(task_dir / "train.jsonl.gz", train)
    write_records(task_dir / "test.jsonl.gz", test)
    describe = lambda split: {"count": len(split), "labels": dict(sorted(Counter(r.label for r in split).items())),
                              "chromosomes": dict(sorted(Counter(r.chromosome for r in split).items()))}
    metadata = {"source": f"{task}/train.jsonl.gz", "source_sha256": sha256_file(data_root / task / "train.jsonl.gz"),
                "pretraining_chromosomes": sorted(PRETRAINING_CHROMOSOMES, key=lambda c: int(c[3:])),
                "train": {**describe(train), "sha256": sha256_file(task_dir / "train.jsonl.gz")},
                "test": {**describe(test), "sha256": sha256_file(task_dir / "test.jsonl.gz")}}
    (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"{task}: train {len(train)}, held-out test {len(test)} ({', '.join(metadata['test']['chromosomes'])})", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="the official NT18 tasks (benchmarks/nt18/prepare.py)")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    for name in NT18_TASKS:
        split_task(arguments.data_root, arguments.out, f"{name}_revised")


if __name__ == "__main__":
    main()
