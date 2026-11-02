"""Pretraining configuration: model, data and training sections, read from one YAML file."""
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import yaml

# GRCh38.p14 RefSeq accessions of chromosomes 1-22. Pretraining uses chromosomes 3-22; chromosomes 1 and 2, unplaced
# scaffolds, alternate loci, X, Y and the mitochondrial genome are not used.
GRCH38_CHROMOSOMES = {
    "1": "NC_000001.11", "2": "NC_000002.12", "3": "NC_000003.12", "4": "NC_000004.12", "5": "NC_000005.10",
    "6": "NC_000006.12", "7": "NC_000007.14", "8": "NC_000008.11", "9": "NC_000009.12", "10": "NC_000010.11",
    "11": "NC_000011.10", "12": "NC_000012.12", "13": "NC_000013.11", "14": "NC_000014.9", "15": "NC_000015.10",
    "16": "NC_000016.10", "17": "NC_000017.11", "18": "NC_000018.10", "19": "NC_000019.10", "20": "NC_000020.11",
    "21": "NC_000021.9", "22": "NC_000022.11",
}


@dataclass(frozen=True)
class ModelConfig:
    model_dim: int = 128
    attention_heads: int = 4
    ffn_dim: int = 512
    # Base-level local attention layers, one per radius (each base attends to bases at most `radius` away).
    base_attention_radii: tuple = (32, 128)
    # Transformer layers over the segment sequence.
    segment_layers: int = 6
    base_embedding_dim: int = 16
    dropout: float = 0.1
    # Size of the segment-length and within-segment-offset embeddings; longer segments share the last entry.
    max_segment_length: int = 512
    rope_theta: float = 10000.0
    # Segment positions for rotary attention are segment centers in base pairs divided by this value.
    rope_position_scale: float = 5.0


@dataclass(frozen=True)
class DataConfig:
    # Directory written by tools/split_fasta (scripts/prepare_genome.sh): N-free windows of `context_length` bases.
    genome_windows: str = "data/GRCh38.p14.acgt.w4096.s4096"
    context_length: int = 4096
    # Each crop has a length drawn uniformly from [min_context_length, context_length].
    min_context_length: int = 256
    # Crops per window and pass: crop 0 is read on the forward strand and crop 1 on the reverse strand.
    crops_per_read: int = 2
    train_sources: tuple = tuple(GRCH38_CHROMOSOMES[str(c)] for c in range(3, 23))
    # Recorded in every configuration (and in the released models' config.json, so they stay readable) but not used:
    # pretraining has no validation or test pass; chromosomes 1 and 2 (and X, Y) are simply not among train_sources.
    validation_sources: tuple = (GRCH38_CHROMOSOMES["2"],)
    test_sources: tuple = (GRCH38_CHROMOSOMES["1"],)
    # Bases this close to either crop end are never prediction targets.
    target_margin: int = 32
    # Fraction of segments masked; every masked segment is hidden with MASK (mask_replacement_probability), replaced by
    # random bases (random_replacement_probability) or left unchanged.
    mask_probability: float = 0.15
    mask_replacement_probability: float = 0.8
    random_replacement_probability: float = 0.1
    segmenter: str = "lcp"
    lcp_level: int = 2
    lcp_library: str = "build/lcptools/lib/liblcptools.so"
    fixed_length: int = 10
    fixed_stride: int = 5
    bpe_tokenizer: str = "data/tokenizers/bpe-4096-1gb-seed42.json"  # the released BPE twin's vocabulary
    # Worker processes that build training examples (segmentation and masking); the example order does not depend on it.
    example_workers: int = 3


@dataclass(frozen=True)
class TrainingConfig:
    run_name: str = "locus"
    seed: int = 42
    batch_size: int = 16
    max_steps: int = 359466
    learning_rate: float = 6e-4
    warmup_steps: int = 3600
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-6
    gradient_clip_norm: float = 1.0
    # Windows are read in genome order through a shuffle buffer of this many (window, crop) items.
    shuffle_buffer_size: int = 512
    mixed_precision: bool = True
    device: str = "cuda"
    log_every: int = 10
    checkpoint_every: int = 1000
    # Checkpoints kept besides the newest one: every `keep_checkpoint_every` steps and the listed steps.
    keep_checkpoint_every: int = 10000
    keep_checkpoint_steps: tuple = ()


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _section(cls, values: dict):
    known = {item.name: item for item in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} option(s): {', '.join(sorted(unknown))}")
    converted = {}
    for name, value in values.items():
        if isinstance(known[name].default, tuple) and isinstance(value, list):
            value = tuple(value)
        converted[name] = value
    return cls(**converted)


def config_from_dict(values: dict) -> Config:
    unknown = set(values) - {"model", "data", "training"}
    if unknown:
        raise ValueError(f"Unknown config section(s): {', '.join(sorted(unknown))}")
    return Config(
        model=_section(ModelConfig, values.get("model") or {}),
        data=_section(DataConfig, values.get("data") or {}),
        training=_section(TrainingConfig, values.get("training") or {}),
    )


def load_config(path: Path, overrides: list = ()) -> Config:
    """Read a YAML config; `overrides` are "section.option=value" strings (value parsed as YAML)."""
    values = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides:
        key, _, raw = item.partition("=")
        section, _, option = key.partition(".")
        if not option:
            raise ValueError(f"Override must look like section.option=value: {item}")
        values.setdefault(section, {})[option] = yaml.safe_load(raw)
    config = config_from_dict(values)
    validate(config)
    return config


def validate(config: Config) -> None:
    model, data = config.model, config.data
    if model.model_dim % model.attention_heads or (model.model_dim // model.attention_heads) % 2:
        raise ValueError("model_dim / attention_heads must be an even integer (rotary embeddings)")
    if data.segmenter not in {"lcp", "fixed", "bpe"}:
        raise ValueError(f"Unknown segmenter: {data.segmenter}")
    if not 0 < data.min_context_length <= data.context_length:
        raise ValueError("min_context_length must be in (0, context_length]")
    if data.mask_replacement_probability + data.random_replacement_probability > 1.0:
        raise ValueError("mask and random replacement probabilities exceed 1")


def with_paths_relative_to(config: Config, root: Path) -> Config:
    """Resolve the data paths of a config against `root` (the repository root). A checkpoint stores the paths of the
    machine it was trained on; when its LCP library or BPE tokenizer is not found there, the copy of this repository
    (build/lcptools, data/tokenizers) is used."""
    root = Path(root)

    def resolve(path: str, fallback_directory: str = None) -> str:
        resolved = Path(path) if Path(path).is_absolute() else root / path
        if not resolved.exists() and fallback_directory is not None and (root / fallback_directory / resolved.name).exists():
            resolved = root / fallback_directory / resolved.name
        return str(resolved)

    return replace(config, data=replace(config.data, genome_windows=resolve(config.data.genome_windows),
                                        lcp_library=resolve(config.data.lcp_library, "build/lcptools/lib"),
                                        bpe_tokenizer=resolve(config.data.bpe_tokenizer, "data/tokenizers")))
