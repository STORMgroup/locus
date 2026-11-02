"""Pretrained LoCUS models on the Hugging Face Hub.

All models are in one Hub repository, one folder per model, each holding `config.json` (the pretraining configuration
and step) and `model.safetensors` (the weights, including the masked-base head). `download` fetches a model's folder into
the local Hugging Face cache and returns its directory; `export` writes that layout from a checkpoint of scripts/pretrain.py.

The repository is set in HUB_REPOSITORY (or the environment variable LOCUS_HUB_REPOSITORY). LOCUS_HUB_DIR points to a
local directory with the same folders, for use without network access."""
import hashlib
import json
import os
from pathlib import Path

import torch

from locus.config import GRCH38_CHROMOSOMES

# the paper's models, and the same four pretrained without chromosomes 20 and 21 (the pretraining-split ablation)
MODELS = {
    "locus-0.5m": "LoCUS-0.5M: width 64, 4 kb context",
    "locus-2m": "LoCUS-2M: width 128, 4 kb context",
    "twin-fixed-2m": "Tokenizer study: LoCUS-2M with fixed 10 bp segments every 5 bp",
    "twin-bpe-2m": "Tokenizer study: LoCUS-2M with BPE tokens (vocabulary 4,096) as segments",
    "locus-0.5m-no-chr20-21": "Pretraining-split ablation: LoCUS-0.5M pretrained without chromosomes 20 and 21 (the NT18 test chromosomes)",
    "locus-2m-no-chr20-21": "Pretraining-split ablation: LoCUS-2M pretrained without chromosomes 20 and 21 (the NT18 test chromosomes)",
    "twin-fixed-2m-no-chr20-21": "Pretraining-split ablation: twin-fixed-2m pretrained without chromosomes 20 and 21",
    "twin-bpe-2m-no-chr20-21": "Pretraining-split ablation: twin-bpe-2m pretrained without chromosomes 20 and 21, with a BPE vocabulary "
                               "trained on its own pretraining chromosomes",
}
RELEASED = ["locus-0.5m", "locus-2m", "twin-fixed-2m", "twin-bpe-2m"]
SPLIT_MODELS = [f"{name}-no-chr20-21" for name in RELEASED]
HUB_REPOSITORY = os.environ.get("LOCUS_HUB_REPOSITORY", "STORMResearch/locus")


def download(name: str, revision: str = None) -> Path:
    """Local directory of model `name` (config.json and model.safetensors), downloaded on first use."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; choose one of: {', '.join(MODELS)}")
    local = os.environ.get("LOCUS_HUB_DIR")
    if local:
        directory = Path(local) / name
        if not (directory / "model.safetensors").is_file():
            raise FileNotFoundError(f"{directory} has no model.safetensors")
        return directory
    if HUB_REPOSITORY.startswith("<"):
        raise RuntimeError("Set the Hugging Face repository of the LoCUS models (locus/hub.py HUB_REPOSITORY or "
                           "LOCUS_HUB_REPOSITORY), or point LOCUS_HUB_DIR to a local copy")
    from huggingface_hub import snapshot_download

    files = [f"{name}/{file}" for file in ("config.json", "model.safetensors", "README.md")]
    return Path(snapshot_download(HUB_REPOSITORY, revision=revision, allow_patterns=files)) / name


def export(checkpoint_path: Path, output_directory: Path, name: str) -> Path:
    """Write the Hub layout of a pretraining checkpoint (scripts/pretrain.py): config.json, model.safetensors, README.md."""
    from safetensors.torch import save_file

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    save_file({key: value.contiguous() for key, value in checkpoint["model_state"].items()}, str(output_directory / "model.safetensors"))
    config = repository_paths(checkpoint["config"])
    (output_directory / "config.json").write_text(json.dumps({"config": config, "step": int(checkpoint["step"])}, indent=2) + "\n")
    (output_directory / "README.md").write_text(model_card(name, config, int(checkpoint["step"])))
    return output_directory


def repository_paths(config: dict) -> dict:
    """The configuration with its data paths relative to the repository (data/..., build/...), as the released models record
    them, so that an exported model carries no path of the machine it was trained on (loading resolves them against the
    repository, locus.config.with_paths_relative_to)."""
    data = dict(config["data"])
    for key in ("genome_windows", "lcp_library", "bpe_tokenizer"):
        path = Path(data[key])
        anchors = [i for i, part in enumerate(path.parts) if part in ("data", "build")]
        if path.is_absolute() and anchors:
            data[key] = str(Path(*path.parts[anchors[-1]:]))
    return {**config, "data": data}


def chromosome_text(sources) -> str:
    """'3-22', '1-19 and 22', ...: the GRCh38 chromosomes of a list of RefSeq accessions."""
    names = {accession: name for name, accession in GRCH38_CHROMOSOMES.items()}
    numbers = sorted(int(names[s]) for s in sources if names.get(s, "").isdigit())
    ranges, start = [], None
    for i, n in enumerate(numbers):
        start = n if start is None else start
        if i + 1 == len(numbers) or numbers[i + 1] != n + 1:
            ranges.append(f"{start}-{n}" if n > start else str(n))
            start = None
    ranges += [names[s] for s in sources if s in names and not names[s].isdigit()]
    return ranges[0] if len(ranges) == 1 else ", ".join(ranges[:-1]) + " and " + ranges[-1]


def model_card(name: str, config: dict, step: int) -> str:
    model, data = config["model"], config["data"]
    return (f"---\nlibrary_name: locus\ntags:\n- dna\n- genomics\n- locally-consistent-parsing\n---\n\n# {name}\n\n{MODELS.get(name, name)}.\n\n"
            f"- Segments: {data['segmenter']}" + (f" (level {data['lcp_level']})" if data["segmenter"] == "lcp" else "") + "\n"
            f"- Width {model['model_dim']}, {model['attention_heads']} heads, base attention radii {list(model['base_attention_radii'])}, "
            f"{model['segment_layers']} segment layers\n"
            f"- Pretrained for {step:,} steps on GRCh38 chromosomes {chromosome_text(data['train_sources'])}, crops of "
            f"{data['min_context_length']}-{data['context_length']} bp\n\n"
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
