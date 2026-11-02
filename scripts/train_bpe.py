"""Train the byte-pair-encoding tokenizer of the BPE twin model (twin-bpe-2m).

    python scripts/train_bpe.py --output runs/tokenizers/bpe-4096-1gb-seed42.json
    python scripts/train_bpe.py --config configs/pretrain/twin-bpe-2m-no-chr20-21.yaml \
        --output data/tokenizers/bpe-4096-1gb-seed42-no-chr20-21.json     # the split ablation's BPE twin (chromosomes 1-19, 22)

The training data are 244,140 windows of 4,096 bp (999,997,440 bases, about 1 Gb) drawn uniformly at random from the
windows of chromosomes 3-22 in the genome window dataset (scripts/prepare_genome.sh): reservoir sampling over the read
manifest, in manifest order, with random.Random(42). The selected windows are read in genome order and every window is
one training sequence (there is no pre-tokenizer). The vocabulary has 4,096 entries, including the special tokens
[UNK], [PAD] and [MASK] and the initial alphabet A, C, G, T, N, and a pair is merged only if it occurs at least twice.
A metadata file next to the tokenizer records the sample (count, bases and a sha256 over the sorted window
identifiers), the manifest checksum and the library version.

The tokenizer used in the paper is distributed as data/tokenizers/bpe-4096-1gb-seed42.json (sha256
7b25a22ea76967150c12fd1d008f8a5626aa4d5833e04dc66cb2e0896def6dd2). It was trained in this way with tokenizers 0.22.2
on the sample with identifier digest e658f2c850d521d3280fa4226f06b1c088612565810ab3d46a786525463bc425. The trainer keeps
every base of the sample in memory as a separate symbol: peak memory grows by about 64 bytes per training base, so the
full sample needs roughly 64 GB of RAM (--windows trains on a smaller sample drawn the same way).

The pretraining-split ablation's BPE twin (configs/pretrain/twin-bpe-2m-no-chr20-21.yaml) has its own tokenizer, trained the
same way with --config on windows of chromosomes 1-19 and 22: data/tokenizers/bpe-4096-1gb-seed42-no-chr20-21.json (sha256
9f12a6bae073f9a27b55cb6e2f6d97dcc745bdf8604d4bfc0c6c7d2bff607087), sample identifier digest
47e53cb0995c492d655531d4ba125fe218fd2eff6e2d3d4bcdde339afafde24b, tokenizers 0.22.2."""
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locus.config import DataConfig, load_config  # noqa: E402
from locus.data.genome import GenomeDataset  # noqa: E402
from locus.finetune.data import sha256_file  # noqa: E402

SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[MASK]"]
INITIAL_ALPHABET = list("ACGTN")


def select_windows(manifest: Path, sources, count: int, seed: int) -> set:
    """Identifiers of `count` windows of the given sources, drawn uniformly by reservoir sampling in manifest order."""
    generator = random.Random(seed)
    reservoir, eligible = [], 0
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"read_id", "source"}.issubset(reader.fieldnames):
            raise ValueError(f"Invalid read manifest: {manifest}")
        for row in reader:
            if row["source"] not in sources:
                continue
            eligible += 1
            if len(reservoir) < count:
                reservoir.append(row["read_id"])
                continue
            replacement = generator.randrange(eligible)
            if replacement < count:
                reservoir[replacement] = row["read_id"]
    if eligible < count:
        raise ValueError(f"Only {eligible} windows are available, but {count} were requested")
    return set(reservoir)


def identifier_digest(identifiers) -> str:
    digest = hashlib.sha256()
    for identifier in sorted(identifiers):
        digest.update(identifier.encode("utf-8") + b"\n")
    return digest.hexdigest()


def train_tokenizer(sequences, vocab_size: int, min_frequency: int):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=SPECIAL_TOKENS,
                         initial_alphabet=INITIAL_ALPHABET, show_progress=True)
    tokenizer.train_from_iterator(sequences, trainer=trainer)
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--genome-windows", type=Path, default=ROOT / DataConfig().genome_windows)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=244140, help="number of training windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--config", type=Path, help="pretraining configuration whose training chromosomes are sampled "
                        "(default: chromosomes 3-22, the released models'; the split ablation's BPE twin samples chromosomes 1-19 and 22)")
    arguments = parser.parse_args()

    from tokenizers import __version__ as tokenizers_version

    sources = load_config(arguments.config).data.train_sources if arguments.config else DataConfig().train_sources
    manifest = arguments.genome_windows / "manifests" / "reads.tsv"
    window_length = json.loads((arguments.genome_windows / "dataset.json").read_text())["window_length"]
    selected = select_windows(manifest, set(sources), arguments.windows, arguments.seed)
    counts = {"windows": 0, "bases": 0}

    def sequences():
        for read in GenomeDataset(arguments.genome_windows, sources, window_length):
            if read.read_id in selected:
                counts["windows"] += 1
                counts["bases"] += len(read.sequence)
                yield read.sequence

    tokenizer = train_tokenizer(sequences(), arguments.vocab_size, arguments.min_frequency)
    if counts["windows"] != len(selected):
        raise RuntimeError(f"Read {counts['windows']} of {len(selected)} selected windows")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(arguments.output))
    metadata = {
        "genome_windows": str(arguments.genome_windows),
        "reads_manifest_sha256": sha256_file(manifest),
        "training_sources": list(sources),
        "selection_seed": arguments.seed,
        "training_reads": counts["windows"],
        "training_raw_bases": counts["bases"],
        "training_read_ids_sha256": identifier_digest(selected),
        "requested_vocab_size": arguments.vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "min_frequency": arguments.min_frequency,
        "tokenizer_sha256": sha256_file(arguments.output),
        "tokenizers_version": tokenizers_version,
    }
    metadata_path = arguments.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Trained on {counts['windows']} windows ({counts['bases']} bases); vocabulary {tokenizer.get_vocab_size()}; "
          f"tokenizer {arguments.output}, metadata {metadata_path}")


if __name__ == "__main__":
    main()
