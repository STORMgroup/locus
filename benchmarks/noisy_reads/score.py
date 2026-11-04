"""Score saved fine-tuned weights on clean and noisy-read test sets, without further training.

A fold fine-tuned with scripts/finetune.py --save-best leaves its result <dir>/fold-XX.json and the selected weights
<dir>/fold-XX.best.pt. The classifier is rebuilt from the pretrained checkpoint named in the result (model
configuration and segmenter; its SHA-256 must match the one recorded at fine-tuning), the selected weights are loaded,
and every test set is scored as during fine-tuning: evaluation mode, logits averaged over both strands (the forward strand
only for a recipe with evaluation_orientation forward), the evaluation batch size and autocast precision of the fine-tuning run. Segments of the noisy reads are computed on the reads
themselves.

    python benchmarks/noisy_reads/score.py results/locus-2m/nt18/H3K27ac_revised/fold-00.json --sets clean ont3 pbsim97

Folds of the baselines (scripts/finetune_baseline.py --save-best; their result names the baseline) are scored the same way: the
baseline's encoder is rebuilt at its pinned revision (or from --baseline-path) and reads single bases, as during fine-tuning.

A set name refers to a directory under --data-root: "clean" is the task's own test set, any other name is the variant
<task>_<name>_revised (gue_splice_<name> for the GUE task) written by make_errors.py, make_controls.py or simulate_pbsim.py, further error draws included
("ont3-d1", ...; scripts/reproduce.py scores every draw). Scores are merged into
<dir>/fold-XX.scored.json as {"best_step": ..., "sets": {name: {metrics..., "loss": mean cross-entropy}}}; sets that
are already scored are kept and skipped unless --overwrite is given. The step stored with the weights must equal the
selected step of the fold result. --output-dir DIR writes the score file to DIR instead, next to a copy of the fold result,
so that sets of another study (for example the real reads of real_reads.py) are kept apart from the fold's own score file.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

from noisy_sets import DEFAULT_DATA_ROOT, OTHER_TASKS, ROOT, variant_task

sys.path.insert(0, str(ROOT))

from locus.config import with_paths_relative_to  # noqa: E402
from locus.finetune.collate import SegmentCache  # noqa: E402
from locus.finetune.data import get_task, load_records, sha256_file  # noqa: E402
from locus.finetune.model import LoCUSEncoder, SequenceClassifier, load_pretrained, set_dropout  # noqa: E402
from locus.hub import MODELS, weights_sha256  # noqa: E402
from locus.finetune.train import effective_autocast_dtype, evaluation_logits, scores  # noqa: E402
from locus.segmentation import Segmenter  # noqa: E402

SET_NAMES = ("clean", "ont1", "ont3", "ontw1", "ontw3", "sub1", "sub3", "pbsim97")


def test_set_path(data_root: Path, task: str, set_name: str) -> Path:
    if set_name == "clean":
        return data_root / task / "test.jsonl.gz"
    if task in OTHER_TASKS:
        return data_root / variant_task(task, set_name) / "test.jsonl.gz"
    if not task.endswith("_revised"):
        raise ValueError(f"Noisy-read test sets exist for the NT18 tasks and {', '.join(OTHER_TASKS)} only, not for {task}")
    return data_root / variant_task(task.removesuffix("_revised"), set_name) / "test.jsonl.gz"


def _save(path: Path, best_step: int, set_name: str, scores: dict) -> None:
    """Adds one set to the score file, re-reading it first so that sets written meanwhile are kept."""
    scored = json.loads(path.read_text()) if path.is_file() else {"best_step": best_step, "sets": {}}
    if scored["best_step"] != best_step:
        raise RuntimeError(f"{path} holds scores of step {scored['best_step']}, the saved weights are from step {best_step}")
    scored["sets"][set_name] = scores
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(scored, indent=2) + "\n")
    os.replace(temporary, path)


def build_classifier(result: dict, weights: dict, pretrained: Path):
    """(fine-tuned classifier, segmenter of its pretrained model)."""
    if weights_sha256(pretrained) != result["pretrained_sha256"]:
        raise RuntimeError(f"{pretrained} is not the pretrained checkpoint the fold was fine-tuned from (SHA-256 differs)")
    recipe = result["recipe"]
    model, config, _ = load_pretrained(pretrained)
    config = with_paths_relative_to(config, ROOT)
    encoder = LoCUSEncoder(model)
    set_dropout(encoder, recipe["dropout"])
    classifier = SequenceClassifier(encoder, get_task(result["task"]).classes, dropout=recipe["dropout"])
    classifier.load_state_dict(weights, strict=True)
    return classifier, Segmenter(config.data)


def build_baseline_classifier(result: dict, weights: dict, baseline_path: str = None):
    """(fine-tuned baseline classifier, its whole-sequence segmenter) for a fold of scripts/finetune_baseline.py."""
    from locus.finetune.baselines import WholeSequenceSegmenter, load_baseline

    encoder = load_baseline(result["baseline"], baseline_path or result.get("baseline_path"))
    dropout = result["recipe"].get("dropout", 0.0)  # inactive when scoring; recipes that keep the model's own dropout have none
    set_dropout(encoder, dropout)
    classifier = SequenceClassifier(encoder, get_task(result["task"]).classes, dropout=dropout)
    classifier.load_state_dict(weights, strict=True)
    return classifier, WholeSequenceSegmenter()


def score_fold(result_path: Path, set_names: list, data_root: Path, device: torch.device, pretrained: Path = None,
               overwrite: bool = False, baseline_path: str = None, output_dir: Path = None) -> dict:
    result = json.loads(result_path.read_text())
    stem = result_path.with_suffix("")
    saved = torch.load(Path(f"{stem}.best.pt"), map_location="cpu", weights_only=True)
    if saved["best_step"] != result["best_step"]:
        raise RuntimeError(f"{stem}.best.pt holds step {saved['best_step']}, but {result_path} selected step {result['best_step']}")
    output = Path(f"{stem}.scored.json")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result_path, output_dir / result_path.name)
        output = output_dir / output.name
    existing = json.loads(output.read_text()) if output.is_file() else {"best_step": saved["best_step"], "sets": {}}
    if existing["best_step"] != saved["best_step"]:
        raise RuntimeError(f"{output} holds scores of step {existing['best_step']}, the saved weights are from step {saved['best_step']}")
    todo = [name for name in set_names if overwrite or name not in existing["sets"]]
    if not todo:
        return existing
    if "baseline" in result:
        classifier, segmenter = build_baseline_classifier(result, saved["model"], baseline_path)
    else:
        if pretrained is None:
            pretrained = result["pretrained"]
            if pretrained not in MODELS and not Path(pretrained).is_absolute() and not Path(pretrained).exists():
                pretrained = ROOT / pretrained
        classifier, segmenter = build_classifier(result, saved["model"], pretrained)
    classifier = classifier.to(device).eval()
    batch_size = max(1, result["recipe"]["batch_size"] // result.get("micro_batches", 1))
    autocast_dtype = effective_autocast_dtype(result["recipe"]["autocast"], device)
    orientation = result["recipe"].get("evaluation_orientation", "both")  # as in the fold's validation and test
    for set_name in todo:
        records = load_records(test_set_path(data_root, result["task"], set_name))
        logits, labels = evaluation_logits(classifier, SegmentCache(records, segmenter), list(range(len(records))), batch_size, device,
                                           autocast_dtype, orientation)
        metrics, loss = scores(logits, labels)
        _save(output, saved["best_step"], set_name, {**metrics, "loss": loss})
        note = f" (fold result: {result['test_metrics']['mcc']:.6f})" if set_name == "clean" else ""
        print(f"{result['task']} fold {result['fold']} {set_name}: mcc {metrics['mcc']:.6f}{note}, loss {loss:.5f}", flush=True)
    return json.loads(output.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="+", help="fold results (fold-XX.json) with fold-XX.best.pt next to them")
    parser.add_argument("--sets", nargs="+", default=list(SET_NAMES), help=f"test sets to score (default: {' '.join(SET_NAMES)})")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="directory of the task directories")
    parser.add_argument("--pretrained", type=Path, help="pretrained checkpoint (default: the path recorded in the fold result, "
                                                        "relative to the current directory or the repository root)")
    parser.add_argument("--overwrite", action="store_true", help="score sets again that are already in the score file")
    parser.add_argument("--baseline-path", help="local copy of a baseline's Hugging Face repository (baseline folds; default: download)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, help="write the score file (and a copy of the fold result) here; one fold result only")
    arguments = parser.parse_args()
    if arguments.output_dir and len(arguments.results) > 1:
        parser.error("--output-dir takes one fold result")
    device = torch.device(arguments.device)
    for result_path in arguments.results:
        score_fold(result_path, arguments.sets, arguments.data_root, device, arguments.pretrained, arguments.overwrite, arguments.baseline_path,
                   arguments.output_dir)


if __name__ == "__main__":
    main()
