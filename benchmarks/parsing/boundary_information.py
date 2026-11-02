"""How much the segment boundaries tell a pretrained model about the bases it has to predict (masked bases).

Pretraining segments the clean crop and then hides whole segments (locus/data/examples.py), so the model sees where the
hidden segments start and end. LCP places those boundaries by rules on the bases themselves, so they carry information
about the hidden bases; fixed segments carry none. On uniform random sequences the context carries no information either:
a model that recovers hidden bases there better than chance (25%) reads them from the boundaries. For each released model,
crops are built exactly as in pretraining (length uniform in [256, 4096], forward strand, the model's own segmenter, 15% of
the segments hidden: MASK 80%, random bases 10%, unchanged 10%, the overlap symbol where two hidden segments disagree), from
--sequences random sequences of 4,096 bp and from as many chromosome-1 windows (held out of pretraining; every k-th
window), and the masked-base accuracy is reported per corruption: bases replaced by MASK (the leak measure), by the overlap
symbol, by random bases, left unchanged, and all targets (as in train_metrics.jsonl).

    python benchmarks/parsing/boundary_information.py --windows data/GRCh38.p14.acgt.w4096.s4096 \
        --output results/parsing/boundary_information.json

Sequences, crops and masks come from fixed seeds and the models run in float32 in evaluation mode, so on the CPU the
result is deterministic."""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from locus.config import GRCH38_CHROMOSOMES, with_paths_relative_to  # noqa: E402
from locus.data.batching import build_model_batch  # noqa: E402
from locus.data.examples import build_example  # noqa: E402
from locus.data.genome import GenomeDataset, GenomeRead  # noqa: E402

MODELS = ["locus-0.5m", "locus-2m", "twin-fixed-2m", "twin-bpe-2m"]
ACTIONS = ["mask", "overlap_conflict", "random", "unchanged"]
LENGTH, SEED = 4096, 42


def sequences(windows: Path, count: int) -> dict:
    """{set name: [GenomeRead]}: uniform random sequences and every k-th chromosome-1 window."""
    rng = random.Random(SEED)
    random_reads = [GenomeRead(f"random_{i:05d}", "".join(rng.choice("ACGT") for _ in range(LENGTH)), "random") for i in range(count)]
    chromosome1 = list(GenomeDataset(windows, [GRCH38_CHROMOSOMES["1"]], LENGTH))
    step = max(1, len(chromosome1) // count)
    return {"random": random_reads, "chromosome 1": chromosome1[::step][:count]}


def accuracy(model, config, reads: list, batch_size: int) -> dict:
    import torch

    from locus.segmentation import Segmenter

    data = with_paths_relative_to(config, ROOT).data
    segmenter = Segmenter(data)
    correct = {action: 0 for action in ACTIONS}
    total = {action: 0 for action in ACTIONS}
    for begin in range(0, len(reads), batch_size):
        examples = [build_example(read, segmenter, data, SEED, 0) for read in reads[begin:begin + batch_size]]
        batch = build_model_batch(examples)
        with torch.no_grad():
            logits = model(batch.base_ids, batch.valid_mask, batch.segment_starts, batch.nucleotide_mask, batch.segment_mask,
                           batch.target_positions)
        hits = (logits.argmax(dim=-1) == batch.labels).tolist()
        for row, example in enumerate(examples):
            for column, action in enumerate(example.corruption_actions):
                correct[action] += hits[row][column]
                total[action] += 1
    out = {action: {"targets": total[action], "accuracy": correct[action] / total[action] if total[action] else None} for action in ACTIONS}
    out["all"] = {"targets": sum(total.values()), "accuracy": sum(correct.values()) / sum(total.values())}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", type=Path, required=True, help="genome window dataset (scripts/prepare_genome.sh)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=1000, help="random sequences and chromosome-1 windows (each)")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--batch-size", type=int, default=16)
    arguments = parser.parse_args()

    import torch

    from locus.finetune.model import load_pretrained

    torch.set_grad_enabled(False)
    sets = sequences(arguments.windows, arguments.sequences)
    result = {"sequences_per_set": arguments.sequences, "length": LENGTH, "seed": SEED, "chance": 0.25, "models": {}}
    for name in arguments.models:
        model, config, _ = load_pretrained(name)
        model.eval()
        result["models"][name] = {set_name: accuracy(model, config, reads, arguments.batch_size) for set_name, reads in sets.items()}
        for set_name, r in result["models"][name].items():
            print(f"{name:14s} {set_name:12s} MASK {r['mask']['accuracy']:.3f}  all targets {r['all']['accuracy']:.3f}", flush=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", arguments.output)


if __name__ == "__main__":
    main()
