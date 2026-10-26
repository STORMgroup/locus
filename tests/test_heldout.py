"""benchmarks/nt18/heldout_split.py: examples on chromosomes outside the pretraining chromosomes form the new test set, all
others stay in the training set."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/nt18"))

import heldout_split  # noqa: E402
from locus.finetune.data import SequenceRecord, load_records, write_records  # noqa: E402


def test_pretraining_chromosomes():
    assert heldout_split.PRETRAINING_CHROMOSOMES == {f"chr{i}" for i in range(3, 23)}


def test_split(tmp_path):
    task = "promoter_tata_revised"
    chromosomes = ["chr1", "chr2", "chr3", "chr20", "chrX", "chrY", "chr5"]
    records = [SequenceRecord(name=f"r{i}", sequence="ACGT" * 10, label=i % 2, task=task, chromosome=c) for i, c in enumerate(chromosomes)]
    write_records(tmp_path / "nt18" / task / "train.jsonl.gz", records)
    heldout_split.split_task(tmp_path / "nt18", tmp_path / "out", task)
    train = load_records(tmp_path / "out" / task / "train.jsonl.gz")
    test = load_records(tmp_path / "out" / task / "test.jsonl.gz")
    assert [r.chromosome for r in train] == ["chr3", "chr20", "chr5"]
    assert [r.chromosome for r in test] == ["chr1", "chr2", "chrX", "chrY"]
    metadata = json.loads((tmp_path / "out" / task / "metadata.json").read_text())
    assert metadata["train"]["count"] == 3 and metadata["test"]["count"] == 4
