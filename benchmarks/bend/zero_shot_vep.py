"""Zero-shot noncoding variant effect prediction of BEND (Marin et al., ICLR 2024), with BEND's protocol.

For each SNP the reference window is genome[start - 256 : start + 256] (BED start, 0-based, 512 bp) and the alternative
window is the same with the base at index 256 replaced by the alternative allele. The score of a variant is the cosine
distance between the per-base embeddings of the two windows at the variant; there is no training and every variant is
scored; the metric is AUROC. Tasks: expression (DeepSEA eQTLs against 1000 Genomes background SNPs, 105,263 variants)
and disease (ClinVar noncoding pathogenic against benign, 295,495 variants).

LoCUS models: the per-base output (the mean of the segments covering the base), both strands averaged (on the reverse
strand the variant is at index 255). Also recorded, never used for the main number: the forward strand alone and the
distance between window-mean embeddings. AUROC is also reported for the variants on chromosomes that were not used for
pretraining (1, 2, X, Y) and by variant consequence.

    python benchmarks/bend/zero_shot_vep.py prepare            # data/bend/variant_effects_{expression,disease}.bed + GRCh38
    python benchmarks/bend/zero_shot_vep.py embed --pretrained locus-2m
    python benchmarks/bend/zero_shot_vep.py analyze            # results/bend/results.json

The BED files of the BEND data release (variant_effects_expression.bed, variant_effects_disease.bed) are downloaded
into --bed-dir when missing and checked against their SHA-256. embed writes results/bend/dist_<name>.npz and is
resumable: it works in shards of 25,000 variants kept under results/bend/parts."""
import argparse
import gzip
import json
import sys
import urllib.request
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from locus.finetune.collate import Collator, SegmentCache  # noqa: E402
from locus.finetune.data import sha256_file  # noqa: E402

TASKS = ("expression", "disease")
FLANK, SHARD, BATCH = 256, 25_000, 256
HELD_OUT = {"chr1", "chr2", "chrX", "chrY"}  # chromosomes not used for pretraining (chromosomes 3-22 only)
BED_URL = "https://sid.erda.dk/share_redirect/aNQa0Oz2lY/data/variant_effects/variant_effects_{task}.bed"
BED_SHA256 = {"expression": "c881ae2e7bc32e62f411893fc0e1de1524fd7e1accd20df9cc2c6bb0e9b8efca",
              "disease": "ea3ce8dcc0274e0f4f0412eecf47f4a9faec77ef6702f1ec85a7e7adc8371246"}
LABELS = {"locus-0.5m": "LoCUS-0.5M", "locus-2m": "LoCUS-2M", "locus-7m": "LoCUS-7M", "locus-2m-16k": "LoCUS-2M-16k",
          "twin-fixed-2m": "fixed-segment twin (2M)", "twin-bpe-2m": "BPE twin (2M)"}
# BEND Table 3: zero-shot AUROC over all variants
PUBLISHED = {"expression": {"DNABERT": 0.60, "GROVER": 0.56, "ResNet-LM": 0.55, "NT-H": 0.55, "NT-MS": 0.54, "AWD-LSTM": 0.53,
                            "HyenaDNA large": 0.51, "DNABERT-2": 0.49, "GENA-LM BERT": 0.49, "GENA-LM BigBird": 0.49, "NT-V2": 0.48,
                            "HyenaDNA tiny": 0.47, "NT-1000G": 0.45, "DeepSEA (supervised)": 0.70},
             "disease": {"NT-MS": 0.77, "DNABERT": 0.56, "ResNet-LM": 0.55, "GENA-LM BERT": 0.55, "GENA-LM BigBird": 0.52,
                         "DNABERT-2": 0.51, "GROVER": 0.51, "NT-1000G": 0.49, "NT-H": 0.48, "NT-V2": 0.48, "AWD-LSTM": 0.45,
                         "HyenaDNA large": 0.45, "HyenaDNA tiny": 0.44, "DeepSEA (supervised)": 0.56}}


class OnDemandSegments(SegmentCache):
    """Segment boundaries computed when a batch is built and not kept: every sequence is embedded once per strand."""

    def bounds(self, index: int, orientation: str) -> list:
        return self.segmenter.bounds(self.sequence(index, orientation))


def bed_file(directory: Path, task: str) -> Path:
    """BEND's variant_effects_<task>.bed, downloaded when missing; its SHA-256 must be the one of the paper's file."""
    path = directory / f"variant_effects_{task}.bed"
    if not path.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        print("downloading", BED_URL.format(task=task), flush=True)
        urllib.request.urlretrieve(BED_URL.format(task=task), path.with_suffix(".part"))
        path.with_suffix(".part").rename(path)
    if sha256_file(path) != BED_SHA256[task]:
        raise ValueError(f"{path} is not the BEND file used in the paper (SHA-256 {BED_SHA256[task]})")
    return path


def read_bed(path: Path) -> list[dict]:
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in handle if line.strip()]


def read_genome(fasta: Path) -> dict:
    """{chr1..chr22, chrX, chrY: sequence} from the RefSeq GRCh38.p14 FASTA (NC_000001 - NC_000024)."""
    contigs, name, parts = {}, None, []
    with gzip.open(fasta, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    contigs[name] = "".join(parts)
                accession, name, parts = line[1:].split()[0], None, []
                if accession.startswith("NC_0000") and 1 <= int(accession[3:9]) <= 24:
                    number = int(accession[3:9])
                    name = "chrX" if number == 23 else "chrY" if number == 24 else f"chr{number}"
            elif name is not None:
                parts.append(line.strip())
    if name is not None:
        contigs[name] = "".join(parts)
    return contigs


def prepare(arguments) -> None:
    from locus.finetune.data import SequenceRecord, write_records

    genome = read_genome(arguments.fasta)
    print("genome", len(genome), "chromosomes", flush=True)
    summary = {}
    for task in arguments.tasks:
        rows, ref, alt, meta = read_bed(bed_file(arguments.bed_dir, task)), [], [], []
        stats = {"variants": len(rows), "ref_match_0based": 0, "ref_match_1based": 0, "short_window": 0, "windows_with_N": 0}
        for i, row in enumerate(rows):
            chromosome, start = row["chromosome"], int(row["start"])
            sequence = genome[chromosome][start - FLANK:start + FLANK].upper()
            stats["ref_match_0based"] += genome[chromosome][start].upper() == row["ref"].upper()
            stats["ref_match_1based"] += genome[chromosome][start - 1].upper() == row["ref"].upper()
            if len(sequence) != 2 * FLANK:
                stats["short_window"] += 1
                sequence = sequence.ljust(2 * FLANK, "N")
            stats["windows_with_N"] += "N" in sequence
            mutated = sequence[:FLANK] + row["alt"].upper() + sequence[FLANK + 1:]
            name = f"{task}:{i}:{chromosome}:{start}:{row['ref']}>{row['alt']}"
            ref.append(SequenceRecord(name=name, sequence=sequence, label=int(row["label"]), task=task, chromosome=chromosome))
            alt.append(SequenceRecord(name=name + ":alt", sequence=mutated, label=int(row["label"]), task=task, chromosome=chromosome))
            meta.append({"chromosome": chromosome, "start": start, "ref": row["ref"], "alt": row["alt"], "label": int(row["label"]),
                         "consequence": row["Consequence"]})
        write_records(arguments.data / task / "ref.jsonl.gz", ref)
        write_records(arguments.data / task / "alt.jsonl.gz", alt)
        with gzip.open(arguments.data / task / "rows.json.gz", "wt") as handle:
            json.dump(meta, handle)
        summary[task] = stats
        print(task, stats, flush=True)
    (arguments.data / "prepared.json").write_text(json.dumps(summary, indent=1) + "\n")


def embed(arguments) -> None:
    import torch

    from locus.config import with_paths_relative_to
    from locus.finetune.data import load_records
    from locus.segmentation import Segmenter

    from locus.finetune.model import load_pretrained
    from locus.models.locus_model import parameter_count

    device = torch.device(arguments.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    name = arguments.name or Path(arguments.pretrained).stem
    final = arguments.results / f"dist_{name}.npz"
    if final.is_file():
        print(name, "done", flush=True)
        return
    model, config, _ = load_pretrained(arguments.pretrained)
    model = model.to(device).eval()
    config = with_paths_relative_to(config, ROOT)
    if arguments.bpe_tokenizer is not None:
        config = replace(config, data=replace(config.data, bpe_tokenizer=str(arguments.bpe_tokenizer)))
    segmenter, large, orientations = Segmenter(config.data), parameter_count(model) > 5_000_000, ("forward", "reverse")

    def per_base(batch):
        return model.encode(batch.base_ids, batch.valid_mask, batch.segment_starts, batch.nucleotide_mask, batch.segment_mask)[0]

    batch_size = arguments.batch_size or (BATCH // 4 if large else BATCH)
    parts = arguments.results / "parts"
    parts.mkdir(parents=True, exist_ok=True)

    def cosine_distance(a, b):
        return 1 - (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)

    counts = {}
    for task in arguments.tasks:
        collates = {}
        for allele in ("ref", "alt"):
            records = load_records(arguments.data / task / f"{allele}.jsonl.gz")
            cache = OnDemandSegments(records, segmenter)
            collates[allele] = (records, {orientation: Collator(cache, orientation) for orientation in orientations})
        n = counts[task] = min(len(collates["ref"][0]), arguments.limit or len(collates["ref"][0]))
        for shard_start in range(0, n, SHARD):
            part = parts / f"{name}_{task}_{shard_start // SHARD:03d}.npz"
            if part.is_file():
                continue
            vectors = {}  # (allele, orientation) -> (vectors of the variant base, window means)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                for allele, (records, collate) in collates.items():
                    for orientation, function in collate.items():
                        at = FLANK if orientation == "forward" else 2 * FLANK - 1 - FLANK  # the variant base on the reverse strand
                        variant_base, window_mean = [], []
                        for start in range(shard_start, min(shard_start + SHARD, n), batch_size):
                            batch = function(list(range(start, min(start + batch_size, shard_start + SHARD, n)))).to(device)
                            hidden = per_base(batch).float()
                            keep = (batch.valid_mask.bool() & (hidden.norm(dim=-1) > 0)).unsqueeze(-1).float()
                            variant_base.append(hidden[:, at])
                            window_mean.append((hidden * keep).sum(1) / keep.sum(1).clamp_min(1))
                        vectors[allele, orientation] = (torch.cat(variant_base), torch.cat(window_mean))

            def both(allele, kind):
                return sum(vectors[allele, orientation][kind] for orientation in orientations) / len(orientations)

            np.savez(part, position=cosine_distance(both("ref", 0), both("alt", 0)).cpu().numpy(),
                     position_forward=cosine_distance(vectors["ref", "forward"][0], vectors["alt", "forward"][0]).cpu().numpy(),
                     window_mean=cosine_distance(both("ref", 1), both("alt", 1)).cpu().numpy(),
                     uncovered=((both("ref", 0).norm(dim=-1) == 0) | (both("alt", 0).norm(dim=-1) == 0)).cpu().numpy())
            print(f"{name} {task} variants {shard_start}-{min(shard_start + SHARD, n)} of {n}", flush=True)
    result = {}
    for task, n in counts.items():
        files = [np.load(parts / f"{name}_{task}_{k:03d}.npz") for k in range((n + SHARD - 1) // SHARD)]
        for key in files[0].files:
            result[f"{task}|{key}"] = np.concatenate([item[key] for item in files])
    np.savez_compressed(final, **result)
    for path in parts.glob(f"{name}_*.npz"):
        path.unlink()
    print(name, "wrote", final, flush=True)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUROC; tied scores share their average rank (equal to sklearn's roc_auc_score)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    for a, b in zip(starts, ends):
        ranks[order[a:b]] = (a + b - 1) / 2 + 1
    positives = labels == 1
    n1, n0 = positives.sum(), (~positives).sum()
    return float((ranks[positives].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def analyze(arguments) -> None:
    rows = {task: json.load(gzip.open(arguments.data / task / "rows.json.gz", "rt")) for task in arguments.tasks}
    result = {"protocol": "BEND zero-shot: cosine distance of the per-base embeddings at the variant (512 bp window), AUROC over all variants",
              "published": PUBLISHED, "prepared": json.loads((arguments.data / "prepared.json").read_text()), "models": {}}
    for path in sorted(arguments.results.glob("dist_*.npz")):
        name, distances = path.stem[5:], np.load(path)
        entry = result["models"][name] = {"label": LABELS.get(name, name)}
        for task, meta in rows.items():
            labels = np.array([row["label"] for row in meta])
            score = distances[f"{task}|position"]
            finite = np.isfinite(score)
            held_out = np.array([row["chromosome"] in HELD_OUT for row in meta])
            consequence = np.array([row["consequence"] for row in meta])
            by_consequence = {kind: {"auroc": auroc(labels[finite & (consequence == kind)], score[finite & (consequence == kind)]),
                                     "n": int((consequence == kind).sum()), "positives": int(labels[consequence == kind].sum())}
                              for kind in sorted(set(consequence))}
            entry[task] = {"auroc": auroc(labels[finite], score[finite]), "n": int(finite.sum()), "not_finite": int((~finite).sum()),
                           "uncovered": int(distances[f"{task}|uncovered"].sum()),
                           "held_out_chromosomes": auroc(labels[finite & held_out], score[finite & held_out]),
                           "pretraining_chromosomes": auroc(labels[finite & ~held_out], score[finite & ~held_out]),
                           "forward_only": auroc(labels, distances[f"{task}|position_forward"]),
                           "window_mean": auroc(labels, distances[f"{task}|window_mean"]), "by_consequence": by_consequence}
            print(f"{entry['label']:32s} {task:10s} AUROC {entry[task]['auroc']:.3f}  (held-out chr {entry[task]['held_out_chromosomes']:.3f}, "
                  f"pretraining chr {entry[task]['pretraining_chromosomes']:.3f}; forward only {entry[task]['forward_only']:.3f}, "
                  f"window mean {entry[task]['window_mean']:.3f})", flush=True)
    arguments.results.mkdir(parents=True, exist_ok=True)
    (arguments.results / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", arguments.results / "results.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage in ("prepare", "embed", "analyze"):
        sub = stages.add_parser(stage)
        sub.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
        sub.add_argument("--data", type=Path, default=ROOT / "data/benchmarks/bend", help="prepared windows")
        sub.add_argument("--results", type=Path, default=ROOT / "results/bend")
        if stage == "prepare":
            sub.add_argument("--bed-dir", type=Path, default=ROOT / "data/bend", help="BEND's variant_effects_<task>.bed files")
            sub.add_argument("--fasta", type=Path, default=ROOT / "data/GRCh38.p14.fasta.gz")
        if stage == "embed":
            sub.add_argument("--pretrained", required=True,
                             help="released model name (e.g. locus-2m), Hub-layout directory or scripts/pretrain.py checkpoint")
            sub.add_argument("--name", help="output name (default: the model name, or the file name of a checkpoint)")
            sub.add_argument("--bpe-tokenizer", type=Path, help="BPE model of a BPE checkpoint (default: the path in its config)")
            sub.add_argument("--batch-size", type=int, help=f"default: {BATCH}, {BATCH // 4} for models above 5M parameters")
            sub.add_argument("--limit", type=int, default=0, help="score only the first N variants of each task (quick test)")
            sub.add_argument("--device", help="default: cuda when available, else cpu")
    arguments = parser.parse_args()
    {"prepare": prepare, "embed": embed, "analyze": analyze}[arguments.stage](arguments)


if __name__ == "__main__":
    main()
