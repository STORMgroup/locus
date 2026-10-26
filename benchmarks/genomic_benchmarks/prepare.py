"""Download the nine Genomic Benchmarks datasets and write them as labelled sequence records.

    python benchmarks/genomic_benchmarks/prepare.py
    python benchmarks/genomic_benchmarks/prepare.py --out data/benchmarks/genomic_benchmarks

The datasets (Gresova et al., 2023) are read from the Hugging Face repositories
katarinagresova/Genomic_Benchmarks_<name>, each pinned to one commit. Dataset <name> is written to
<out>/gb_<name>/train.jsonl.gz and test.jsonl.gz, one SequenceRecord per line (locus/finetune/data.py) named
gb_<name>:<split>:<row> with chromosome "unknown", and <out>/datasets.json lists the source commit and the size, class
counts and length range of every split. Sequences are upper-cased and any symbol other than A, C, G, T or N is replaced
by N (counted in datasets.json; the pinned datasets contain none). A split whose size differs from the expected one or
a label outside the task's classes aborts preparation."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from locus.finetune.data import GB_TASKS, SequenceRecord, get_task, write_records  # noqa: E402

ALLOWED_BASES = frozenset("ACGTN")

# Dataset name: (pinned commit, training records, test records).
DATASETS = {
    "demo_coding_vs_intergenomic_seqs": ("85458172cab6cffad46cb172e64ca94780506ed6", 75000, 25000),
    "drosophila_enhancers_stark": ("b5fdfb2d398ebf80491832760a4c1b88ed551b42", 5184, 1730),
    "human_enhancers_cohn": ("49ef8d0d80087ece7441096afcef0123b0781ce0", 20843, 6948),
    "human_enhancers_ensembl": ("1e4a3993d89263e6496b64824cad9716c5f6643b", 123872, 30970),
    "human_ensembl_regulatory": ("8c014854d2aa093c37e2f161a241a35ece583fac", 231348, 57713),
    "human_nontata_promoters": ("e0003669df0a180c8570ebe091ed91f20fa080ec", 27097, 9034),
    "human_ocr_ensembl": ("3c07e4be6425be11073baa2daacf6c143c64e4ff", 139804, 34952),
    "demo_human_or_worm": ("a4f7f76b18a38f43ff3a7d5e8299408c3c121df2", 75000, 25000),
    "dummy_mouse_enhancers_ensembl": ("d4d48371e76b0c46717592c32c183c04b19ad16b", 968, 242),
}


def split_file(repo: str, revision: str, split: str) -> str:
    """Path of the single Parquet file of `split` in the dataset repository at `revision`."""
    from huggingface_hub import HfApi

    candidates = [name for name in HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
                  if name.startswith(f"data/{split}-") and name.endswith(".parquet")]
    if len(candidates) != 1:
        raise ValueError(f"Expected one {split} Parquet file in {repo}@{revision}, found {candidates}")
    return candidates[0]


def prepare_dataset(name: str, output_root: Path) -> dict:
    """Write both splits of one dataset and return its datasets.json entry."""
    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download

    revision, train_count, test_count = DATASETS[name]
    repo = f"katarinagresova/Genomic_Benchmarks_{name}"
    task = get_task(f"gb_{name}")
    entry = {"repo": repo, "revision": revision}
    for split, expected_count in (("train", train_count), ("test", test_count)):
        path = hf_hub_download(repo, split_file(repo, revision, split), repo_type="dataset", revision=revision)
        parquet_file = parquet.ParquetFile(path)
        sequence_column = "seq" if "seq" in parquet_file.schema_arrow.names else "sequence"
        labels, replaced, lengths = Counter(), Counter(), []

        def records():
            row_index = 0
            for batch in parquet_file.iter_batches(batch_size=8192, columns=[sequence_column, "label"]):
                for row in batch.to_pylist():
                    sequence = str(row[sequence_column]).upper()
                    invalid = set(sequence) - ALLOWED_BASES
                    if invalid:
                        replaced.update(base for base in sequence if base in invalid)
                        sequence = "".join(base if base in ALLOWED_BASES else "N" for base in sequence)
                    label = int(row["label"])
                    if not 0 <= label < task.classes:
                        raise ValueError(f"Invalid label {label} in {repo} {split} row {row_index}")
                    labels[label] += 1
                    lengths.append(len(sequence))
                    yield SequenceRecord(name=f"{task.name}:{split}:{row_index}", sequence=sequence, label=label,
                                         task=task.name, chromosome="unknown")
                    row_index += 1

        count = write_records(output_root / task.name / f"{split}.jsonl.gz", records())
        if count != expected_count:
            raise ValueError(f"{task.name} {split} has {count} records, expected {expected_count}")
        entry[split] = {"count": count, "classes": dict(sorted(labels.items())), "min_length": min(lengths),
                        "max_length": max(lengths), "mean_length": round(sum(lengths) / len(lengths), 1),
                        "non_acgtn_replaced_by_n": dict(replaced)}
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "benchmarks" / "genomic_benchmarks")
    arguments = parser.parse_args()

    if set(DATASETS) != set(GB_TASKS):
        raise RuntimeError("The dataset list does not match locus.finetune.data.GB_TASKS")
    summary = {}
    for name in DATASETS:
        entry = prepare_dataset(name, arguments.out)
        summary[f"gb_{name}"] = entry
        print(f"gb_{name}", json.dumps({split: entry[split] for split in ("train", "test")}), flush=True)
    (arguments.out / "datasets.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Prepared {len(summary)} Genomic Benchmarks datasets in {arguments.out}")


if __name__ == "__main__":
    main()
