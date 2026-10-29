"""Pretrained LoCUS models on the Hugging Face Hub.

Each model is one Hub repository holding `config.json` (the pretraining configuration and step) and `model.safetensors`
(the weights, including the masked-base head). `download` fetches a model into the local Hugging Face cache and returns
its directory; `export` writes that layout from a checkpoint of scripts/pretrain.py.

The Hub organization is set in HUB_ORGANIZATION (or the environment variable LOCUS_HUB_ORGANIZATION). LOCUS_HUB_DIR
points to a local directory with one sub-directory per model in the same layout, for use without network access."""
import hashlib
import json
import os
from pathlib import Path

import torch

MODELS = {
    "locus-0.5m": "LoCUS-0.5M: width 64, 4 kb context",
    "locus-2m": "LoCUS-2M: width 128, 4 kb context",
    "twin-fixed-2m": "Tokenizer study: LoCUS-2M with fixed 10 bp segments every 5 bp",
    "twin-bpe-2m": "Tokenizer study: LoCUS-2M with BPE tokens (vocabulary 4,096) as segments",
}
HUB_ORGANIZATION = os.environ.get("LOCUS_HUB_ORGANIZATION", "<huggingface-organization>")


def repository(name: str) -> str:
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; choose one of: {', '.join(MODELS)}")
    return f"{HUB_ORGANIZATION}/{name}"


def download(name: str, revision: str = None) -> Path:
    """Local directory of model `name` (config.json and model.safetensors), downloaded on first use."""
    local = os.environ.get("LOCUS_HUB_DIR")
    if local:
        directory = Path(local) / name
        if not (directory / "model.safetensors").is_file():
            raise FileNotFoundError(f"{directory} has no model.safetensors")
        return directory
    if HUB_ORGANIZATION.startswith("<"):
        raise RuntimeError("Set the Hugging Face organization of the LoCUS models (locus/hub.py HUB_ORGANIZATION or "
                           "LOCUS_HUB_ORGANIZATION), or point LOCUS_HUB_DIR to a local copy")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repository(name), revision=revision, allow_patterns=["config.json", "model.safetensors", "README.md"]))


def export(checkpoint_path: Path, output_directory: Path, name: str) -> Path:
    """Write the Hub layout of a pretraining checkpoint (scripts/pretrain.py): config.json, model.safetensors, README.md."""
    from safetensors.torch import save_file

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    save_file({key: value.contiguous() for key, value in checkpoint["model_state"].items()}, str(output_directory / "model.safetensors"))
    (output_directory / "config.json").write_text(json.dumps({"config": checkpoint["config"], "step": int(checkpoint["step"])}, indent=2) + "\n")
    (output_directory / "README.md").write_text(model_card(name, checkpoint["config"], int(checkpoint["step"])))
    return output_directory


def model_card(name: str, config: dict, step: int) -> str:
    model, data = config["model"], config["data"]
    return (f"---\nlibrary_name: locus\ntags:\n- dna\n- genomics\n- locally-consistent-parsing\n---\n\n# {name}\n\n{MODELS.get(name, name)}.\n\n"
            f"- Segments: {data['segmenter']}" + (f" (level {data['lcp_level']})" if data["segmenter"] == "lcp" else "") + "\n"
            f"- Width {model['model_dim']}, {model['attention_heads']} heads, base attention radii {list(model['base_attention_radii'])}, "
            f"{model['segment_layers']} segment layers\n"
            f"- Pretrained for {step:,} steps on GRCh38 chromosomes 3-22, crops of {data['min_context_length']}-{data['context_length']} bp\n\n"
            "## Use\n\n```python\nimport locus\n\n"
            f"model = locus.load(\"{name}\")\nvectors = model.embed([\"ACGT\" * 100])\n```\n\n"
            "Code, fine-tuning recipes and the scripts that reproduce every result of the paper: see the LoCUS repository.\n")


def load_state(source) -> tuple[dict, dict, int]:
    """(weights, config dict, pretraining step) from a model name, a Hub-layout directory or a scripts/pretrain.py checkpoint."""
    path = Path(source)
    if not path.exists() and str(source) in MODELS:
        path = download(str(source))
    if path.is_dir():
        from safetensors.torch import load_file

        meta = json.loads((path / "config.json").read_text())
        return load_file(str(path / "model.safetensors")), meta["config"], int(meta["step"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint["model_state"], checkpoint["config"], int(checkpoint["step"])


def weights_sha256(source) -> str:
    """SHA-256 of the weights (names, dtypes, shapes and values, in name order): the same for a model whether it is read
    from the Hub layout or from a scripts/pretrain.py checkpoint."""
    state, _, _ = load_state(source)
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(f"{key}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        digest.update(tensor.numpy().tobytes() if tensor.dtype != torch.bfloat16 else tensor.view(torch.int16).numpy().tobytes())
    return digest.hexdigest()
