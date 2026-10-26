"""Reads the genome windows written by tools/split_fasta: gzipped FASTA shards plus a read manifest
(manifests/reads.tsv with read_id, source and shard columns)."""
import csv
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class GenomeRead:
    read_id: str
    sequence: str
    source: str


def iter_fasta(path: Path) -> Iterator[tuple[str, str, str]]:
    """(identifier, upper-case sequence, header comment) of every record of a gzipped FASTA file."""
    read_id, comment, parts = None, "", []
    with gzip.open(path, mode="rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if read_id is not None:
                    yield read_id, "".join(parts), comment
                header = line[1:].split(maxsplit=1)
                read_id, comment, parts = header[0], header[1] if len(header) == 2 else "", []
            else:
                if read_id is None:
                    raise ValueError(f"Sequence found before header in {path}")
                parts.append(line.upper())
    if read_id is not None:
        yield read_id, "".join(parts), comment


def source_from_comment(comment: str) -> str:
    for item in comment.split():
        if item.startswith("source="):
            return item.removeprefix("source=")
    raise ValueError(f"FASTA comment has no source field: {comment}")


class GenomeDataset:
    """Windows of the given sources (chromosome accessions), in manifest (genome) order."""

    def __init__(self, root: Path, sources, context_length: int):
        self.root = Path(root)
        self.sources = set(sources)
        self.context_length = context_length
        if not (self.root / "_SUCCESS").is_file():
            raise FileNotFoundError(f"Incomplete genome window dataset: {self.root}")
        manifest = self.root / "manifests" / "reads.tsv"
        self.shards = []
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None or not {"read_id", "source", "shard"}.issubset(reader.fieldnames):
                raise ValueError(f"Invalid read manifest: {manifest}")
            for row in reader:
                if row["source"] in self.sources and row["shard"] not in self.shards:
                    self.shards.append(row["shard"])
        if not self.shards:
            raise ValueError(f"No windows of the requested sources in {self.root}")

    def __iter__(self) -> Iterator[GenomeRead]:
        for shard in self.shards:
            for read_id, sequence, comment in iter_fasta(self.root / shard):
                source = source_from_comment(comment)
                if source not in self.sources:
                    continue
                if len(sequence) != self.context_length:
                    raise ValueError(f"{read_id} has length {len(sequence)}, expected {self.context_length}")
                yield GenomeRead(read_id=read_id, sequence=sequence, source=source)
