"""Labelled sequence records, benchmark task definitions and validation folds."""
import gzip
import hashlib
import io
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch


@dataclass(frozen=True)
class SequenceRecord:
    name: str
    sequence: str
    label: int
    task: str
    chromosome: str


def write_records(path: Path, records: Iterable[SequenceRecord]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    # no file name or time in the gzip header, so the same records always give the same file (and checksum)
    with open(temporary, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed, \
            io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            count += 1
    temporary.replace(path)
    return count


def load_records(path: Path) -> list[SequenceRecord]:
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(SequenceRecord(**json.loads(line)))
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid record at {path}:{line_number}") from error
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Task:
    name: str
    benchmark: str
    classes: int
    metric: str  # metric reported in the paper tables


NT18_TASKS = ["H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac", "H3K9me3", "H4K20me1",
              "enhancers", "enhancers_types", "promoter_all", "promoter_no_tata", "promoter_tata", "splice_sites_acceptors",
              "splice_sites_all", "splice_sites_donors"]
GB_TASKS = ["demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "drosophila_enhancers_stark", "dummy_mouse_enhancers_ensembl",
            "human_enhancers_cohn", "human_enhancers_ensembl", "human_ensembl_regulatory", "human_nontata_promoters", "human_ocr_ensembl"]
_THREE_CLASS = {"enhancers_types_revised", "splice_sites_all_revised", "gb_human_ensembl_regulatory"}

TASKS = {}
for _name in NT18_TASKS:
    _task = f"{_name}_revised"
    TASKS[_task] = Task(_task, "nt18", 3 if _task in _THREE_CLASS else 2, "mcc")
for _name in GB_TASKS:
    _task = f"gb_{_name}"
    TASKS[_task] = Task(_task, "genomic_benchmarks", 3 if _task in _THREE_CLASS else 2, "accuracy")
# GUE splice-site task (DNABERT-2), the second splice benchmark of the tokenizer study (benchmarks/gue/prepare.py)
TASKS["gue_splice"] = Task("gue_splice", "gue", 3, "mcc")


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise ValueError(f"Unknown task {name!r}; choose one of: {', '.join(TASKS)}")
    return TASKS[name]


def stratified_folds(labels: torch.Tensor, folds: int, seed: int) -> list[torch.Tensor]:
    """`folds` disjoint, label-stratified folds of the training set (sorted indices). Fine-tuning fold k validates on
    fold k of the 10-fold split made with seed 42 and trains on the rest."""
    if labels.dim() != 1 or folds < 2:
        raise ValueError("Stratified folds require one-dimensional labels and folds >= 2")
    generator = random.Random(seed)
    fold_indices = [[] for _ in range(folds)]
    for label in sorted(set(int(value) for value in labels.tolist())):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten().tolist()
        if len(indices) < folds:
            raise ValueError(f"Class {label} has only {len(indices)} rows for {folds} folds")
        generator.shuffle(indices)
        for position, index in enumerate(indices):
            fold_indices[position % folds].append(index)
    return [torch.tensor(sorted(values), dtype=torch.long) for values in fold_indices]


def complement_indices(size: int, excluded: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(size, dtype=torch.bool)
    mask[excluded] = False
    return torch.nonzero(mask, as_tuple=False).flatten()
