"""Publish pretrained LoCUS models to the Hugging Face Hub (for the authors). Each checkpoint of scripts/pretrain.py is
written in the Hub layout (config.json, model.safetensors, a model card) and, with --upload, pushed to the folder
<model name> of one repository (--repository, e.g. ORG/locus). Without --upload the files are only written to --staging,
for inspection.

    python scripts/upload_to_hub.py --checkpoint locus-2m=runs/locus-2m/checkpoints/final.pt [...] --repository ORG/locus [--upload]

Uploading needs a Hugging Face token with write access (huggingface-cli login)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from locus.hub import MODELS, export, load_state  # noqa: E402
from locus.models.locus_model import LoCUSModel  # noqa: E402
from locus.config import config_from_dict  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--repository", required=True, help="Hub repository of all models, e.g. ORG/locus")
    parser.add_argument("--staging", type=Path, default=Path("runs/hub"))
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--private", action="store_true", help="create the repositories as private")
    arguments = parser.parse_args()
    for item in arguments.checkpoint:
        name, _, path = item.partition("=")
        if name not in MODELS:
            parser.error(f"unknown model {name!r}; choose one of: {', '.join(MODELS)}")
        directory = export(Path(path), arguments.staging / name, name)
        state, config, _ = load_state(directory)  # the exported files load strictly into the model
        LoCUSModel(config_from_dict(config).model).load_state_dict(state, strict=True)
        print(f"{name}: written to {directory}", flush=True)
        if arguments.upload:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(arguments.repository, exist_ok=True, private=arguments.private)
            api.upload_folder(repo_id=arguments.repository, folder_path=str(directory), path_in_repo=name, commit_message=f"Release {name}")
            print(f"{name}: uploaded to https://huggingface.co/{arguments.repository}/tree/main/{name}", flush=True)


if __name__ == "__main__":
    main()
