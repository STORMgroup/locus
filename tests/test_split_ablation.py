"""The pretraining-split ablation (benchmarks/README.md): the four configurations differ from their released counterparts only
in their training chromosomes (and, for the BPE twin, its tokenizer, trained on those chromosomes), the split plans of
scripts/reproduce.py fine-tune each model with its released counterpart's recipe, and fine-tuning starts only from complete
pretraining runs."""
import argparse
import hashlib
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import reproduce  # noqa: E402
from locus.config import GRCH38_CHROMOSOMES, load_config  # noqa: E402

WITHOUT_20_21 = tuple(GRCH38_CHROMOSOMES[str(name)] for name in list(range(1, 20)) + [22])


@pytest.mark.parametrize("name, released", list(reproduce.SPLIT.items()))
def test_configs_differ_only_in_the_training_chromosomes(name, released):
    base = asdict(load_config(ROOT / "configs/pretrain" / f"{released}.yaml"))
    config = asdict(load_config(ROOT / "configs/pretrain" / f"{name}.yaml"))
    assert config["model"] == base["model"]
    assert {k: v for k, v in config["training"].items() if k != "run_name"} == {k: v for k, v in base["training"].items() if k != "run_name"}
    unused = {"validation_sources", "test_sources"}  # recorded only (locus/config.py)
    tokenizer = {"bpe_tokenizer"} if base["data"]["segmenter"] == "bpe" else set()  # trained on the same chromosomes
    assert {key for key in base["data"] if config["data"][key] != base["data"][key]} - unused == {"train_sources"} | tokenizer
    assert config["data"]["train_sources"] == WITHOUT_20_21


def test_split_tokenizer_is_the_papers():
    assert hashlib.sha256(reproduce.SPLIT_TOKENIZER.read_bytes()).hexdigest() == "9f12a6bae073f9a27b55cb6e2f6d97dcc745bdf8604d4bfc0c6c7d2bff607087"
    assert load_config(ROOT / "configs/pretrain/twin-bpe-2m-no-chr20-21.yaml").data.bpe_tokenizer == str(reproduce.SPLIT_TOKENIZER.relative_to(ROOT))


def test_split_plans(tmp_path):
    arguments = argparse.Namespace(results=tmp_path / "results", data=tmp_path / "data", runs=tmp_path / "runs", models_from=None)
    plans = reproduce.build_plans(arguments)
    pretrain = plans["split-pretrain"]
    assert pretrain.exclusive and pretrain.data == ["genome-windows", "bpe-no-chr20-21"]
    assert [job.name for job in pretrain.stages[0]] == [f"pretrain.{name}" for name in reproduce.SPLIT]
    finetune = plans["split-nt18"].stages[0]
    assert len(finetune) == len(reproduce.SPLIT) * 10 * 18
    folds = [int(job.command[job.command.index("--fold") + 1]) for job in finetune]
    assert folds == sorted(folds)  # fold by fold
    setup = {job.name.split(".nt18.")[0]: (job.command[job.command.index("--pretrained") + 1], Path(job.command[job.command.index("--recipe") + 1]).name,
                                           job.command[job.command.index("--micro-batches") + 1], "--save-best" in job.command) for job in finetune}
    assert setup == {name: (name, f"{reproduce.RECIPE[released]}.yaml",  # the released split models, by name
                            str(reproduce.MICRO_BATCHES[released]), True) for name, released in reproduce.SPLIT.items()}
    # local checkpoints: --models-from DIR, or the --runs of a command that also pretrains them
    for extra, directory in (({"models_from": tmp_path / "own"}, tmp_path / "own"), ({"plans": ["split-pretrain", "split-nt18"]}, tmp_path / "runs")):
        local = reproduce.build_plans(argparse.Namespace(**{**vars(arguments), **extra}))["split-nt18"].stages[0]
        assert {job.command[job.command.index("--pretrained") + 1] for job in local} == {str(directory / name / "checkpoints/final.pt") for name in reproduce.SPLIT}
    score = plans["split-score"]
    assert [len(stage) for stage in score.stages] == [len(reproduce.SPLIT_SCORED) * 180, len(reproduce.SPLIT_SCORED) * 180 * 2]
    assert {job.name.split(".")[0] for job in score.stages[1]} == set(reproduce.SPLIT_SCORED)
    assert not set(reproduce.SPLIT_PLANS) & set(reproduce.PLANS)  # not part of "all"
    assert plans["pretrain"].exclusive  # one pretraining per GPU, as split-pretrain
    # the checkpoints to check: those the selected fine-tunes start from, and those split-pretrain would skip as done
    assert reproduce.split_models_needed(plans, ["split-nt18"], ["locus-2m-*"]) == ["locus-2m-no-chr20-21"]
    assert reproduce.split_models_needed(plans, ["split-pretrain"]) == list(reproduce.SPLIT)
    assert reproduce.split_models_needed(plans, ["split-score"]) == reproduce.SPLIT_SCORED
    assert reproduce.split_models_needed(plans, ["nt18"]) == []


def test_split_checkpoints_must_be_complete(tmp_path):
    """Fine-tuning starts only from final.pt files of the full schedule and the model's own configuration."""
    def save(name, step, **training):
        config = load_config(ROOT / "configs/pretrain" / f"{name}.yaml").to_dict()
        config["training"].update(training)
        path = tmp_path / name / "checkpoints/final.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"step": step, "config": config}, path)

    for name in reproduce.SPLIT:
        save(name, 359466)
    reproduce.check_split_checkpoints(tmp_path, list(reproduce.SPLIT))  # complete: passes
    save("locus-2m-no-chr20-21", 4)
    with pytest.raises(SystemExit, match="stopped at step 4 of 359466"):
        reproduce.check_split_checkpoints(tmp_path, list(reproduce.SPLIT))
    save("locus-2m-no-chr20-21", 359466, learning_rate=1e-3)
    with pytest.raises(SystemExit, match="training settings differ"):
        reproduce.check_split_checkpoints(tmp_path, list(reproduce.SPLIT))
    with pytest.raises(SystemExit, match="missing"):
        reproduce.check_split_checkpoints(tmp_path / "elsewhere", ["locus-2m-no-chr20-21"])
