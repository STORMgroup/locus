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
    outputs = [job.output for job in jobs.values() if job.name != "bend.analyze"]
    assert len(outputs) == len(set(outputs))
    counts = {name: len({job.name for stage in plan.stages for job in stage}) for name, plan in plans(tmp_path).items()}
    assert counts == {"pretrain": 6, "nt18": 162, "random-init": 54, "baselines": 162, "genomic-benchmarks": 81, "tokenizer-study": 324, "heldout": 108, "context": 18,
                      "long-range": 19, "parsing": 1, "bend": 6, "efficiency": 1, "timing": 108}


def test_nt18_folds(tmp_path):
    """--nt18-folds 10 extends the NT18 plans (LoCUS, random init, baselines, tokenizer study) to the 10 folds of the split and
    leaves the other plans, the held-out-chromosome check included, at 3 folds."""
    arguments = argparse.Namespace(results=tmp_path / "results", data=tmp_path / "data", runs=tmp_path / "runs", models_from=None, nt18_folds=10)
    counts = {name: len({job.name for stage in plan.stages for job in stage}) for name, plan in reproduce.build_plans(arguments).items()}
    assert counts == {"pretrain": 6, "nt18": 540, "random-init": 180, "baselines": 540, "genomic-benchmarks": 81, "tokenizer-study": 1080, "heldout": 108, "context": 18,
                      "long-range": 19, "parsing": 1, "bend": 6, "efficiency": 1, "timing": 108}
