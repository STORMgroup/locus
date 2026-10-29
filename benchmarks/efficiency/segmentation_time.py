"""Dataset creation cost of fine-tuning: the time to segment every NT18 record (training and test set, both strands) with
the released segmenters, on one CPU core, as scripts/finetune.py does through SegmentCache (once per record and strand,
then reused in every epoch).

Segmenters: LCP level 2 (LoCUS), BPE with a vocabulary of 4,096 (twin-bpe-2m) and fixed 10 bp segments every 5 bp
(twin-fixed-2m), each built from its pretraining configuration; and, for the single-base baselines (Caduceus, HyenaDNA),
the per-record work they need (base encoding of both strands), so that every model is charged the same way. Each
segmenter runs over the 18 tasks in one pass after a warm-up on 200 records; the time is recorded per task and split.

    python benchmarks/efficiency/segmentation_time.py [--data-root data/benchmarks/nt18] [--output results/efficiency/segmentation_time.json]

Run it on an otherwise idle machine; a single thread is used (set OMP_NUM_THREADS=1 and the like for other libraries)."""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from locus.alphabet import reverse_complement  # noqa: E402
from locus.config import load_config, with_paths_relative_to  # noqa: E402
from locus.finetune.collate import encode_bases  # noqa: E402
from locus.finetune.data import NT18_TASKS, load_records  # noqa: E402
from locus.segmentation import Segmenter  # noqa: E402

CONFIGS = {"lcp": "configs/pretrain/locus-2m.yaml", "bpe": "configs/pretrain/twin-bpe-2m.yaml", "fixed": "configs/pretrain/twin-fixed-2m.yaml"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/benchmarks/nt18")
    parser.add_argument("--output", type=Path, default=ROOT / "results/efficiency/segmentation_time.json")
    parser.add_argument("--repeats", type=int, default=1, help="passes per task and split; the fastest is kept")
    arguments = parser.parse_args()
    records = {(task, split): [r.sequence for r in load_records(arguments.data_root / f"{task}_revised" / f"{split}.jsonl.gz")]
               for task in NT18_TASKS for split in ("train", "test")}
    both = {key: [(s, reverse_complement(s)) for s in sequences] for key, sequences in records.items()}
    runners = {name: Segmenter(with_paths_relative_to(load_config(ROOT / path), ROOT).data).bounds for name, path in CONFIGS.items()}
    runners["single-base"] = encode_bases

    result = {"processor": platform.processor(), "python": platform.python_version(),
              "unit": "seconds, both strands of every train and test record", "tasks": {}}
    for name, run in runners.items():
        for forward, reverse in both[(NT18_TASKS[0], "test")][:200]:
            run(forward), run(reverse)
        for task in NT18_TASKS:
            entry = result["tasks"].setdefault(task, {})
            for split in ("train", "test"):
                best = None
                for _ in range(arguments.repeats):
                    start = time.perf_counter()
                    for forward, reverse in both[(task, split)]:
                        run(forward)
                        run(reverse)
                    elapsed = time.perf_counter() - start
                    best = elapsed if best is None else min(best, elapsed)
                entry.setdefault(name, {})[split] = best
                entry.setdefault("records", {})[split] = len(records[(task, split)])
                entry.setdefault("bases", {})[split] = sum(len(s) for s in records[(task, split)])
            print(name, task, {k: round(v, 2) for k, v in entry[name].items()}, flush=True)
        total = sum(sum(result["tasks"][t][name].values()) for t in NT18_TASKS)
        bases = sum(sum(result["tasks"][t]["bases"].values()) for t in NT18_TASKS) * 2
        result.setdefault("totals", {})[name] = {"seconds": total, "bases_per_second": bases / total}
        print(name, "total", round(total, 1), "s", flush=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
