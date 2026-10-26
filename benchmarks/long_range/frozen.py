"""Genomics Long-Range Benchmark tasks scored with the benchmark paper's frozen-embedding protocol (Kao et al. 2024, Sec. 2.4):
frozen embeddings -> MLP; five folds whose validation sets are five randomly chosen training chromosomes; in each fold the
epoch with the lowest validation loss is kept and scored once on the test set; results are means over the five folds. The
MLP's hidden width is twice the input embedding dimension.

    eqtl      variant_effect_causal_eqtl (GRCh38; test chromosomes 9 and 10, all rows): the mean embedding of the 1,500 bp
              centred on the variant, for the reference and the alternative sequence, concatenated with a one-hot tissue
              -> MLP with two hidden layers, cross-entropy. Test AUROC and accuracy, and AUROC by distance to the nearest
              TSS (< 3 kb, 3-12 kb, 12-35 kb, > 35 kb).
    bulk_rna  bulk_rna_expression (hg19; test chromosome 8): the sequence centred on the gene's CAGE representative TSS,
              read on the gene's strand; the mean embedding of the 384 bp upstream and 256 bp downstream of the TSS -> MLP
              with one hidden layer, mean squared error on the 218 tissue values. Test Spearman correlation per tissue
              across genes (mean over tissues), per gene across tissues (mean over genes), and R^2 (mean over tissues).

Choices where the benchmark paper is silent: features are standardized with training-fold statistics; Adam (lr 1e-3),
batch 256, ReLU, at most 100 epochs, stopping after 10 epochs without a lower validation loss; embeddings are averaged
over both strands; inputs are the benchmark loader's windows at 4 kb and 16 kb, and an item whose window contains an N
or runs off the chromosome is dropped (counts in rows.json.gz). Model "tss" (eqtl only) is the sequence-free reference:
one-hot tissue and log distance to the TSS. Fits on several CPU threads (--threads, default 4) can differ in the last digits
between repeated runs; --threads 1 is reproducible.

    python benchmarks/long_range/frozen.py prepare --task eqtl
    python benchmarks/long_range/frozen.py embed --task eqtl --length 4096 --pretrained locus-2m --name locus-2m
    python benchmarks/long_range/frozen.py fit --task eqtl --length 4096 --name locus-2m

prepare writes <data>/lrb_<task>_<4k|16k>/{ref,alt|seq}.jsonl.gz and rows.json.gz; embed writes
<results>/emb_<name>_lrb_<task>_<4k|16k>.npz; fit writes <results>/fit_<name>_lrb_<task>_<4k|16k>.json."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

from locus.alphabet import reverse_complement  # noqa: E402
from locus.finetune.collate import Collator, SegmentCache  # noqa: E402
from locus.finetune.data import SequenceRecord, load_records, write_records  # noqa: E402
from prepare_enhancer import REPOSITORY, REVISION, benchmark_url, grch38_chromosomes  # noqa: E402

LENGTHS = (4096, 16384)
SEED, FOLDS = 42, 5
DISTANCE_BINS = (("<3k", 0, 3_000), ("3k-12k", 3_000, 12_000), ("12k-35k", 12_000, 35_000), (">35k", 35_000, float("inf")))
ALLELES = {"eqtl": ("ref", "alt"), "bulk_rna": ("seq",)}
HG19_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz"
HG19_MD5 = "806c02398f5ac5da8ffd6da2d1d5d1a9"  # UCSC md5sum.txt; the file used for the paper
BASES_PER_BATCH = 131_072


class OnDemandSegments(SegmentCache):
    """Segment boundaries computed when a batch is built and not kept: every sequence is embedded once per strand."""

    def bounds(self, index: int, orientation: str) -> list:
        return self.segmenter.bounds(self.sequence(index, orientation))


def task_directory(task: str, length: int) -> str:
    return f"lrb_{task}_{length // 1024}k"


def read_csv_rows(path: str) -> list[list[str]]:
    with urllib.request.urlopen(benchmark_url(path)) as response:
        return list(csv.reader(io.TextIOWrapper(response, encoding="utf-8")))


def ucsc_chromosomes(fasta: Path, wanted: set):
    """(name, upper-case sequence) of the UCSC-style records (>chrN) in `wanted`, one at a time, in file order."""
    name, parts = None, []
    with gzip.open(fasta, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if name in wanted:
                    yield name, "".join(parts).upper()
                name, parts = line[1:].split()[0], []
            elif name in wanted:
                parts.append(line.rstrip())
    if name in wanted:
        yield name, "".join(parts).upper()


def cut(sequence: str, start: int, length: int):
    """The benchmark loader's window: `length` bases with `start` at index length // 2, or None off the chromosome."""
    left, right = start - length // 2, start + length // 2 + length % 2
    return sequence[left:right] if left >= 0 and right < len(sequence) else None


def prepare(arguments) -> None:
    task = arguments.task
    dropped = {"off_chromosome": 0, "contains_n": 0}
    if task == "eqtl":
        header, *body = read_csv_rows("variant_effect_causal_eqtl/All_Tissues.csv")
        rows = [dict(zip(header, values)) for values in body]
        keys = sorted({(row["CHROM"], int(row["POS"]), row["REF"], row["ALT"]) for row in rows}, key=lambda key: (key[0], key[1], key[3]))
        genome = grch38_chromosomes(arguments.fasta, {key[0] for key in keys})
    else:
        header, *body = read_csv_rows("bulk_rna_expression/gene_coordinates.csv")
        rows = [dict(zip(header, values)) for values in body]
        labels = read_csv_rows("bulk_rna_expression/rna_expression_values.csv")[1:]
        if len(labels) != len(rows):
            raise ValueError("The expression table and the gene table have different lengths")
        for row, values in zip(rows, labels):
            row["labels"] = [float(value) for value in values]
        keys = sorted({(row["chrom"], int(row["CAGE_representative_TSS"]), row["strand"]) for row in rows})
        if not arguments.hg19.is_file():
            arguments.hg19.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(HG19_URL, arguments.hg19.with_suffix(".part"))
            arguments.hg19.with_suffix(".part").rename(arguments.hg19)
        digest = hashlib.md5()
        with open(arguments.hg19, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 24), b""):
                digest.update(chunk)
        if digest.hexdigest() != HG19_MD5:
            raise ValueError(f"{arguments.hg19} is not UCSC's hg19.fa.gz (MD5 {HG19_MD5})")
        genome = ucsc_chromosomes(arguments.hg19, {key[0] for key in keys})
    by_chromosome = {}
    for key in keys:
        by_chromosome.setdefault(key[0], []).append(key)
    windows, mismatched = {}, 0
    for chromosome, sequence in genome:
        for key in by_chromosome[chromosome]:
            # positions are 1-based
            cuts = {length: cut(sequence, key[1] - 1, length) for length in LENGTHS}
            if any(window is None for window in cuts.values()):
                dropped["off_chromosome"] += 1
            elif any(set(window) - set("ACGT") for window in cuts.values()):
                dropped["contains_n"] += 1
            else:
                if task == "eqtl":
                    mismatched += cuts[LENGTHS[0]][LENGTHS[0] // 2] != key[2]
                else:  # genes on the minus strand are read on their own strand, as the benchmark's loader does
                    cuts = {length: reverse_complement(window) if key[2] == "-" else window for length, window in cuts.items()}
                windows[key] = cuts
        print(f"{chromosome}: done", flush=True)
    kept = [key for key in keys if key in windows]
    index = {key: position for position, key in enumerate(kept)}
    if task == "eqtl":
        table = [{"item": index[key], "split": row["split"], "chromosome": row["CHROM"], "label": int(row["label"]), "tissue": row["tissue"],
                  "distance": abs(int(float(row["distance_to_nearest_TSS"])))}
                 for row in rows if (key := (row["CHROM"], int(row["POS"]), row["REF"], row["ALT"])) in index]
    else:
        table = [{"item": index[key], "split": row["split"], "chromosome": row["chrom"], "gene": row["id"], "strand": row["strand"],
                  "labels": row["labels"]} for row in rows if (key := (row["chrom"], int(row["CAGE_representative_TSS"]), row["strand"])) in index]
    for length in LENGTHS:
        folder = arguments.data / task_directory(task, length)
        for allele in ALLELES[task]:
            if allele == "alt":  # the alternative base replaces the variant base at index length // 2
                sequences = [windows[key][length][:length // 2] + key[3] + windows[key][length][length // 2 + 1:] for key in kept]
            else:
                sequences = [windows[key][length] for key in kept]
            records = [SequenceRecord(name=f"{task_directory(task, length)}:{allele}:{':'.join(map(str, key))}", sequence=sequence, label=0,
                                      task=task_directory(task, length), chromosome=key[0]) for key, sequence in zip(kept, sequences)]
            write_records(folder / f"{allele}.jsonl.gz", records)
        with gzip.open(folder / "rows.json.gz", "wt") as handle:
            json.dump({"repo": REPOSITORY, "revision": REVISION, "items": len(kept), "dropped_items": dropped,
                       "ref_mismatches": mismatched, "rows": table}, handle)
    print(json.dumps({"task": task, "items": len(kept), "rows": len(table), "dropped_items": dropped, "ref_mismatches": mismatched}))


def pooled_window(task: str, length: int, strand: str) -> tuple[int, int]:
    """[start, end) of the pooled bases in the stored sequence (read on the gene's strand for bulk_rna)."""
    if task == "eqtl":
        return length // 2 - 750, length // 2 + 750
    tss = length // 2 if strand == "+" else length // 2 - 1  # reverse-complementing moves the TSS base from L/2 to L/2 - 1
    return tss - 384, tss + 256


def embed(arguments) -> None:
    import torch

    from locus.config import with_paths_relative_to
    from locus.segmentation import Segmenter

    from locus.finetune.model import load_pretrained
    from locus.models.locus_model import parameter_count

    device = torch.device(arguments.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    length, task = arguments.length, arguments.task
    model, config, _ = load_pretrained(arguments.pretrained)
    model = model.to(device).eval()
    segmenter = Segmenter(with_paths_relative_to(config, ROOT).data)
    orientations = ("forward", "reverse")
    large = parameter_count(model) > 5_000_000

    def per_base(batch):
        return model.encode(batch.base_ids, batch.valid_mask, batch.segment_starts, batch.nucleotide_mask, batch.segment_mask)[0]

    batch_size = arguments.batch_size or max(1, BASES_PER_BATCH // length // (2 if large else 1))
    folder = arguments.data / task_directory(task, length)
    with gzip.open(folder / "rows.json.gz", "rt") as handle:
        strands = {row["item"]: row.get("strand", "+") for row in json.load(handle)["rows"]}
    result = {}
    for allele in ALLELES[task]:
        records = load_records(folder / f"{allele}.jsonl.gz")
        total = min(arguments.limit or len(records), len(records))
        cache = OnDemandSegments(records, segmenter)
        bounds = np.array([pooled_window(task, length, strands[item]) for item in range(len(records))])
        strand_vectors = []
        for orientation in orientations:
            collate, vectors = Collator(cache, orientation), []
            # the reverse-complement strand holds the same bases at mirrored indices
            low, high = (bounds[:, 0], bounds[:, 1]) if orientation == "forward" else (length - bounds[:, 1], length - bounds[:, 0])
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                for start in range(0, total, batch_size):
                    batch = collate(list(range(start, min(start + batch_size, total)))).to(device)
                    hidden = per_base(batch).float()
                    positions = torch.arange(hidden.shape[1], device=device)
                    keep = ((positions >= torch.as_tensor(low[start:start + len(hidden)], device=device)[:, None])
                            & (positions < torch.as_tensor(high[start:start + len(hidden)], device=device)[:, None])
                            & batch.valid_mask.bool()).unsqueeze(-1).float()
                    vectors.append(((hidden * keep).sum(1) / keep.sum(1).clamp_min(1)).cpu().numpy())
            strand_vectors.append(np.concatenate(vectors))
            print(f"{task} {arguments.name} {length} {allele} {orientation}: {total} embedded", flush=True)
        result[allele] = np.mean(strand_vectors, axis=0).astype(np.float32)
    output = arguments.output or arguments.results / f"emb_{arguments.name}_{task_directory(task, length)}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **result)
    print(f"wrote {output}", flush=True)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUROC; tied scores share their average rank."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    positives = labels == 1
    n1, n0 = positives.sum(), (~positives).sum()
    return float((ranks[positives].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks along each row; tied values share their average rank (scipy.stats.rankdata, method "average")."""
    ranks = np.empty(values.shape, dtype=np.float64)
    for row, row_values in enumerate(values):
        order = np.argsort(row_values, kind="stable")
        ordered = row_values[order]
        starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
        ends = np.r_[starts[1:], len(ordered)]
        ranks[row, order] = np.repeat((starts + ends - 1) / 2.0, ends - starts)
    return ranks


def spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman correlation of each row of `a` with the same row of `b` (tie-averaged ranks: many expression values are
    tied at the floor of a tissue)."""
    ra, rb = average_ranks(a), average_ranks(b)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))


def fit(arguments) -> None:
    import torch

    torch.set_num_threads(arguments.threads)
    task, length, name = arguments.task, arguments.length, arguments.name
    with gzip.open(arguments.data / task_directory(task, length) / "rows.json.gz", "rt") as handle:
        rows = json.load(handle)["rows"]
    item = np.array([row["item"] for row in rows])
    embeddings_path = arguments.embeddings or arguments.results / f"emb_{name}_{task_directory(task, length)}.npz"
    if task == "eqtl":
        tissues = sorted({row["tissue"] for row in rows})
        onehot = np.eye(len(tissues), dtype=np.float32)[[tissues.index(row["tissue"]) for row in rows]]
        if name == "tss":
            features = np.concatenate([onehot, np.log10(1 + np.array([[row["distance"]] for row in rows], dtype=np.float32))], 1)
            width = 64
        else:
            embeddings = np.load(embeddings_path)
            features, width = np.concatenate([embeddings["ref"][item], embeddings["alt"][item], onehot], 1), 2 * embeddings["ref"].shape[1]
        targets = np.array([row["label"] for row in rows])
        depth, outputs = 2, 2
    else:
        embeddings = np.load(embeddings_path)
        features, width = embeddings["seq"][item], 2 * embeddings["seq"].shape[1]
        targets = np.array([row["labels"] for row in rows], dtype=np.float32)
        depth, outputs = 1, targets.shape[1]
    chromosomes = np.array([row["chromosome"] for row in rows])
    in_training = np.array([row["split"] == "train" for row in rows])
    test = np.where(~in_training)[0]
    validation_chromosomes = random.Random(SEED).sample(sorted(set(chromosomes[in_training])), FOLDS)
    report = {"task": task, "model": name, "length": length, "validation_chromosomes": validation_chromosomes, "folds": []}
    for fold, held_out in enumerate(validation_chromosomes):
        torch.manual_seed(SEED + fold)
        train = np.where(in_training & (chromosomes != held_out))[0]
        validation = np.where(in_training & (chromosomes == held_out))[0]
        mean, std = features[train].mean(0), features[train].std(0) + 1e-6
        inputs = torch.tensor((features - mean) / std, dtype=torch.float32)
        labels = torch.tensor(targets, dtype=torch.long if task == "eqtl" else torch.float32)
        layers, size = [], features.shape[1]
        for _ in range(depth):
            layers += [torch.nn.Linear(size, width), torch.nn.ReLU()]
            size = width
        network = torch.nn.Sequential(*layers, torch.nn.Linear(size, outputs))
        loss_function = torch.nn.CrossEntropyLoss() if task == "eqtl" else torch.nn.MSELoss()
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        best, best_state, best_epoch, waited = float("inf"), None, 0, 0
        for epoch in range(1, 101):
            network.train()
            for chunk in torch.randperm(len(train)).split(256):
                batch = torch.as_tensor(train[chunk.numpy()])
                optimizer.zero_grad()
                loss_function(network(inputs[batch]), labels[batch]).backward()
                optimizer.step()
            network.eval()
            with torch.no_grad():
                validation_loss = float(loss_function(network(inputs[torch.as_tensor(validation)]), labels[torch.as_tensor(validation)]))
            if validation_loss < best:
                best, best_state, best_epoch, waited = validation_loss, {k: v.clone() for k, v in network.state_dict().items()}, epoch, 0
            else:
                waited += 1
                if waited >= 10:
                    break
        network.load_state_dict(best_state)
        with torch.no_grad():
            predictions = network(inputs[torch.as_tensor(test)]).numpy()
        result = {"validation_chromosome": held_out, "best_epoch": best_epoch, "validation_loss": best}
        if task == "eqtl":
            score = predictions[:, 1] - predictions[:, 0]
            result.update(auroc=auroc(targets[test], score), accuracy=float(((score > 0) == targets[test]).mean()))
            distance = np.array([rows[i]["distance"] for i in test])
            result["auroc_by_distance"] = {label: auroc(targets[test][(distance >= low) & (distance < high)], score[(distance >= low) & (distance < high)])
                                           for label, low, high in DISTANCE_BINS}
        else:
            truth = targets[test]
            r2 = 1 - ((truth - predictions) ** 2).sum(0) / ((truth - truth.mean(0)) ** 2).sum(0)
            result.update(spearman_genes=float(spearman_rows(predictions.T, truth.T).mean()),
                          spearman_tissues=float(spearman_rows(predictions, truth).mean()), r2=float(r2.mean()))
        report["folds"].append(result)
        print(f"{task} {name} {length} fold {fold} (validation {held_out}): "
              + json.dumps({key: value for key, value in result.items() if key != "auroc_by_distance"}), flush=True)
    keys = ("auroc", "accuracy") if task == "eqtl" else ("spearman_genes", "spearman_tissues", "r2")
    report["mean"] = {key: float(np.mean([fold[key] for fold in report["folds"]])) for key in keys}
    if task == "eqtl":
        report["mean"]["auroc_by_distance"] = {label: float(np.mean([fold["auroc_by_distance"][label] for fold in report["folds"]]))
                                               for label, _, _ in DISTANCE_BINS}
    output = arguments.output or arguments.results / f"fit_{name}_{task_directory(task, length)}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["mean"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage in ("prepare", "embed", "fit"):
        sub = stages.add_parser(stage)
        sub.add_argument("--task", required=True, choices=sorted(ALLELES))
        sub.add_argument("--data", type=Path, default=ROOT / "data/benchmarks/long_range")
        if stage == "prepare":
            sub.add_argument("--fasta", type=Path, default=ROOT / "data/GRCh38.p14.fasta.gz", help="GRCh38.p14 (eqtl)")
            sub.add_argument("--hg19", type=Path, default=ROOT / "data/hg19.fa.gz", help="UCSC hg19, downloaded when missing (bulk_rna)")
            continue
        sub.add_argument("--length", type=int, required=True, choices=LENGTHS)
        sub.add_argument("--name", required=True, help="model name used in the output file names (fit: 'tss' = no sequence, eqtl only)")
        sub.add_argument("--results", type=Path, default=ROOT / "results/long_range/frozen")
        sub.add_argument("--output", type=Path, help="output file (default: in --results, named after the model and the task)")
        if stage == "embed":
            sub.add_argument("--pretrained", required=True,
                             help="released model name (e.g. locus-2m), Hub-layout directory or scripts/pretrain.py checkpoint")
            sub.add_argument("--batch-size", type=int, help=f"default: {BASES_PER_BATCH:,} bases per batch, half for models above 5M parameters")
            sub.add_argument("--limit", type=int, default=0, help="embed only the first N items (quick test)")
            sub.add_argument("--device", help="default: cuda when available, else cpu")
        else:
            sub.add_argument("--embeddings", type=Path, help="embedding file (default: the embed stage's output in --results)")
            sub.add_argument("--threads", type=int, default=4, help="CPU threads of torch")
    arguments = parser.parse_args()
    {"prepare": prepare, "embed": embed, "fit": fit}[arguments.stage](arguments)


if __name__ == "__main__":
    main()
