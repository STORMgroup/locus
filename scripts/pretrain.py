"""Pretrain a LoCUS model.

    python scripts/pretrain.py --config configs/pretrain/locus-2m.yaml --output-dir runs/locus-2m
    python scripts/pretrain.py --config configs/pretrain/locus-2m.yaml --output-dir runs/locus-2m --resume

--resume continues from the newest step checkpoint in the output directory (or from the given checkpoint). Options can
be overridden with --set section.option=value, for example --set training.device=cpu."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locus.config import load_config, with_paths_relative_to  # noqa: E402
from locus.training.pretrain import newest_checkpoint, train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", nargs="?", const="auto", help="resume from PATH, or from the newest step checkpoint")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.OPTION=VALUE")
    parser.add_argument("--stop-after-step", type=int, help="stop early without changing the learning-rate schedule")
    arguments = parser.parse_args()
    config = with_paths_relative_to(load_config(arguments.config, arguments.set), ROOT)
    resume = None
    if arguments.resume == "auto":
        resume = newest_checkpoint(arguments.output_dir / "checkpoints")
    elif arguments.resume is not None:
        resume = Path(arguments.resume)
    train(config, arguments.output_dir, resume=resume, stop_after_step=arguments.stop_after_step)


if __name__ == "__main__":
    main()
