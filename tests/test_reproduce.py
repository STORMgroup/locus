"""scripts/reproduce.py: every command it runs is accepted by the argument parser of its script, and no two different jobs
write the same result."""
import argparse
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import reproduce  # noqa: E402


class Parsed(Exception):
    pass


def plans(tmp_path: Path, models_from=None) -> dict:
    arguments = argparse.Namespace(results=tmp_path / "results", data=tmp_path / "data", runs=tmp_path / "runs", models_from=models_from)
    return reproduce.build_plans(arguments)


def python_commands(tmp_path: Path) -> list:
    """One command per script, subcommand and set of options, from the jobs and the data preparation steps."""
    commands = [job.command for models_from in (None, tmp_path / "runs") for plan in plans(tmp_path, models_from).values()
                for stage in plan.stages for job in stage]
    commands += [command for _, steps in reproduce.data_steps(tmp_path / "data").values() for command in steps]
    distinct = {}
    for command in commands:
        if command[0] != reproduce.PYTHON:
            continue
        subcommand = command[2] if len(command) > 2 and not command[2].startswith("-") and not command[2].endswith(".json") else None
        distinct.setdefault((command[1], subcommand, tuple(a for a in command[2:] if a.startswith("--"))), command)
    return list(distinct.values())


def test_commands_parse(tmp_path, monkeypatch):
    original = argparse.ArgumentParser.parse_args

    def parse_only(self, args=None, namespace=None):
        original(self, args, namespace)
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_only)
    commands = python_commands(tmp_path)
    assert len(commands) > 15
    for command in commands:
        script = Path(command[1])
        assert script.is_file(), script
        monkeypatch.setattr(sys, "argv", command[1:])
        monkeypatch.setattr(sys, "path", [str(script.parent)] + sys.path)
        with pytest.raises(Parsed):
            runpy.run_path(str(script), run_name="__main__")


def test_jobs_have_distinct_outputs(tmp_path):
    jobs = {}
    for plan in plans(tmp_path).values():
        for stage in plan.stages:
            for job in stage:
                if job.name in jobs:  # a job shared by two plans (LoCUS-2M's NT18 runs) must be the same job
                    assert jobs[job.name].command == job.command
                jobs[job.name] = job
    outputs = [job.output for job in jobs.values()]
    assert len(outputs) == len(set(outputs))
    counts = {name: len({job.name for stage in plan.stages for job in stage}) for name, plan in plans(tmp_path).items()}
    assert counts == {"pretrain": 4, "nt18": 360, "random-init": 180, "baselines": 360, "baselines-fp32": 360, "baselines-authors": 360, "baselines-authors-bf16": 0, "genomic-benchmarks": 180, "tokenizer-study": 1080, "gue-splice": 90,
                      "real-reads": 1080, "single-nucleotide": 540, "single-nucleotide-fp32": 540, "heldout": 360, "parsing": 3, "stability": 1, "efficiency": 2, "timing": 108,
                      "split-pretrain": 4, "split-nt18": 720, "split-score": 1620}


def test_nt18_folds(tmp_path):
    """--nt18-folds 3 restricts the NT18 plans (LoCUS, random init, baselines, tokenizer study, real reads) and the GUE splice study
    to folds 0-2 of the split (default: all 10) and leaves the other plans unchanged."""
    arguments = argparse.Namespace(results=tmp_path / "results", data=tmp_path / "data", runs=tmp_path / "runs", models_from=None, nt18_folds=3)
    counts = {name: len({job.name for stage in plan.stages for job in stage}) for name, plan in reproduce.build_plans(arguments).items()}
    assert counts == {"pretrain": 4, "nt18": 108, "random-init": 54, "baselines": 108, "baselines-fp32": 108, "baselines-authors": 108, "baselines-authors-bf16": 0, "genomic-benchmarks": 180, "tokenizer-study": 324, "gue-splice": 27,
                      "real-reads": 324, "single-nucleotide": 162, "single-nucleotide-fp32": 162, "heldout": 360, "parsing": 3, "stability": 1, "efficiency": 2, "timing": 108,
                      "split-pretrain": 4, "split-nt18": 216, "split-score": 486}


def test_other_folds(tmp_path):
    """--other-folds 3 restricts the Genomic Benchmarks and the held-out-chromosome check to folds 0-2."""
    arguments = argparse.Namespace(results=tmp_path / "results", data=tmp_path / "data", runs=tmp_path / "runs", models_from=None, other_folds=3)
    counts = {name: len({job.name for stage in plan.stages for job in stage}) for name, plan in reproduce.build_plans(arguments).items()}
    assert counts["genomic-benchmarks"] == 54 and counts["heldout"] == 108 and counts["nt18"] == 360


def test_one_fold_tables():
    """--nt18-folds 1: a one-fold cell is complete and has no standard deviation, so the table step does not fail."""
    import collect_results

    one, two = collect_results.cell([0.5], folds=1), collect_results.cell([0.5, 0.6], folds=2)
    assert one["complete"] and one["sd"] is None and collect_results.text(one) == "0.500"
    assert collect_results.text(two) == "0.550 ± 0.071"


def test_baselines_run_at_the_precision_of_locus(tmp_path):
    """The baselines of the main tables (plans baselines and single-nucleotide) fine-tune with the bfloat16 recipes, the
    precision of LoCUS; plan baselines-fp32 runs the same recipes in float32, and the two recipe files differ only there.
    scripts/finetune_baseline.py defaults to the bfloat16 recipe."""
    import yaml

    from locus.finetune.baselines import BASELINES

    built = plans(tmp_path)
    recipe = lambda job: Path(job.command[job.command.index("--recipe") + 1]).name
    assert {recipe(job) for job in built["baselines"].stages[0]} == {"caduceus-ps-0.47m-bf16.yaml", "caduceus-ps-1.9m-bf16.yaml"}
    assert {recipe(job) for job in built["baselines-fp32"].stages[0]} == {"caduceus-ps-0.47m.yaml", "caduceus-ps-1.9m.yaml"}
    assert all("/caduceus-ps-1.9m-bf16/" in str(job.output) for job in built["single-nucleotide"].stages[1])
    assert {recipe(job) for job in built["single-nucleotide-fp32"].stages[0]} == {"caduceus-ps-1.9m.yaml"}
    assert all("/caduceus-ps-1.9m/" in str(job.output) for job in built["single-nucleotide-fp32"].stages[1])
    load = lambda name: yaml.safe_load((ROOT / "configs/finetune" / f"{name}.yaml").read_text())
    for size in ("0.47m", "1.9m"):
        bf16, fp32 = load(f"caduceus-ps-{size}-bf16"), load(f"caduceus-ps-{size}")
        assert bf16.pop("autocast") == "bfloat16" and fp32.pop("autocast") == "float32" and bf16 == fp32
    assert {baseline.recipe for baseline in BASELINES.values()} == {"caduceus-ps-0.47m-bf16", "caduceus-ps-1.9m-bf16"}


def test_precision_table(tmp_path, monkeypatch):
    """The main NT18 table shows the bfloat16 baselines; the precision table pairs them with the float32 runs fold by fold."""
    import json

    import collect_results

    monkeypatch.setattr(collect_results, "NT18_FOLDS", 2)
    results = tmp_path / "results"
    tasks = [task for _, group in collect_results.NT18_GROUPS for task in group]
    for run, base in (("caduceus-ps-1.9m-bf16", 0.6), ("caduceus-ps-1.9m", 0.5)):
        for task in tasks:
            for fold in range(2):
                path = results / run / "nt18" / f"{task}_revised" / f"fold-{fold:02d}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                mcc = base + 0.01 * fold
                path.write_text(json.dumps({"history": [{"validation_loss": 0.5, "validation_metrics": {"mcc": 0.0}, "test_metrics": {"mcc": mcc}}],
                                            "test_metrics": {"mcc": mcc}}))
    numbers = {}
    main = collect_results.nt18_table(results, numbers)
    assert "caduceus-ps-1.9m-bf16" in main.splitlines()[2] and "| caduceus-ps-1.9m |" not in main
    table = collect_results.precision_table(results, numbers)
    difference = numbers["nt18_precision"]["H2AFZ"]["caduceus-ps-1.9m-bf16 - caduceus-ps-1.9m"]
    assert difference["values"] == pytest.approx([0.1, 0.1]) and "+0.100 ± 0.000" in table
    assert numbers["nt18_precision_tasks_bf16_higher"] == {"caduceus-ps-1.9m-bf16": 18}


def test_rows_over_several_tasks_wait_for_every_task(tmp_path, monkeypatch):
    """Like the NT18 means, the paired drop tests, the real-read rows and the single-nucleotide rows of a group or of all 18
    tasks stay empty until every task-fold of their tasks is in, so a partial rerun is not compared with 18-task means."""
    import json

    import collect_results

    monkeypatch.setattr(collect_results, "NT18_FOLDS", 1)
    monkeypatch.setattr(collect_results, "WARNINGS", [])
    draws = lambda name, value: {f"{name}-d{k}" if k else name: {"mcc": value} for k in range(collect_results.NOISY_DRAWS)}
    noisy = {}
    for _, key in collect_results.STUDY_SETS:
        noisy.update(draws(key, 0.5))
    real = {"realref": {"mcc": 0.7}, **draws("realsup", 0.6), **draws("realhac", 0.55)}

    def write(results, tasks):
        for task in tasks:
            for _, directory in collect_results.TOKENIZERS:
                path = results / "tokenizer_study" / directory / "nt18" / f"{task}_revised" / "fold-00.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"best_step": 1, "test_metrics": {"mcc": 0.6}, "history": [
                    {"validation_loss": 0.5, "validation_metrics": {"mcc": 0.6}, "test_metrics": {"mcc": 0.6}, "extra_test_metrics": noisy}]}))
                path = results / "real_reads" / directory / "nt18" / f"{task}_revised" / "fold-00.scored.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"best_step": 1, "sets": real}))
            path = results / "single_nucleotide" / "caduceus-ps-1.9m-bf16" / "nt18" / f"{task}_revised" / "fold-00.scored.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"best_step": 1, "sets": {"clean": {"mcc": 0.6}, **noisy}}))
            path = results / "real_reads" / "caduceus-ps-1.9m-bf16" / "nt18" / f"{task}_revised" / "fold-00.scored.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"best_step": 1, "sets": real}))

    def collect(results):
        numbers = {}
        tables = (collect_results.study_tables(results, numbers) + collect_results.real_reads_table(results, numbers)
                  + collect_results.single_nucleotide_table(results, numbers))
        return numbers, tables

    tasks = [task for _, group in collect_results.NT18_GROUPS for task in group]
    write(tmp_path / "one", ["promoter_tata"])
    numbers, tables = collect(tmp_path / "one")
    assert all("LCP drop" not in test for test in numbers["tokenizer_study"]["drop tests"])
    assert len(numbers["tokenizer_study"]["drop tests"]) == len(collect_results.STUDY_SETS) * (len(collect_results.NT18_GROUPS) + 1)
    assert not numbers["real_reads"] and all(not values for values in numbers["single_nucleotide"].values())
    assert "| 1% | all 18 | 1 of 18 |" in tables and "| promoter | 1 of 3 |" in tables
    write(tmp_path / "all", tasks)
    numbers, _ = collect(tmp_path / "all")
    assert all(test["LCP drop"] == pytest.approx(-0.1) for test in numbers["tokenizer_study"]["drop tests"])
    assert numbers["real_reads"]["all 18"]["LCP"]["drop sup"] == pytest.approx(-0.1)
    simulated = numbers["single_nucleotide"]["NT18, simulated errors (drop = set - clean test set)"]
    assert simulated["1%"]["all 18"]["task-folds"] == 18
