"""Download pretrained models into the local Hugging Face cache: the LoCUS models and, with --baselines, the public models
fine-tuned as same-protocol baselines (at the revisions pinned in locus/finetune/baselines.py).

    python scripts/download_models.py                  # the four LoCUS models (same as --locus)
    python scripts/download_models.py locus-0.5m locus-2m
    python scripts/download_models.py --split          # the four models of the pretraining-split ablation (no chr20/21)
    python scripts/download_models.py --baselines      # the two baselines (Caduceus-PS 0.47M and 1.9M)
    python scripts/download_models.py --locus --split --baselines

The models are then loaded by name: locus.load("locus-2m"), scripts/finetune.py --pretrained locus-2m, or
scripts/finetune_baseline.py --baseline caduceus-ps-1.9m."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from locus.finetune.baselines import BASELINES  # noqa: E402
from locus.hub import MODELS, RELEASED, SPLIT_MODELS, download  # noqa: E402


def download_baseline(name: str) -> Path:
    from huggingface_hub import snapshot_download

    baseline = BASELINES[name]
    return Path(snapshot_download(baseline.repository, revision=baseline.revision))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", metavar="MODEL",
                        help=f"models to download (default: {', '.join(RELEASED)}; also {', '.join(SPLIT_MODELS)}; baselines: {', '.join(BASELINES)})")
    parser.add_argument("--locus", action="store_true", help="the four LoCUS models")
    parser.add_argument("--split", action="store_true", help="the four models of the pretraining-split ablation")
    parser.add_argument("--baselines", action="store_true", help="both baselines")
    arguments = parser.parse_args()
    unknown = [name for name in arguments.models if name not in MODELS and name not in BASELINES]
    if unknown:
        parser.error(f"unknown model(s) {', '.join(unknown)}; choose from {', '.join(list(MODELS) + list(BASELINES))}")
    names = list(arguments.models)
    if arguments.locus or not (names or arguments.baselines or arguments.split):
        names += [name for name in RELEASED if name not in names]
    if arguments.split:
        names += [name for name in SPLIT_MODELS if name not in names]
    if arguments.baselines:
        names += [name for name in BASELINES if name not in names]
    for name in names:
        path = download_baseline(name) if name in BASELINES else download(name)
        print(f"{name}: {path}", flush=True)


if __name__ == "__main__":
    main()
