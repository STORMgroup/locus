"""Reproduce the experiments of the paper: prepare the data, run every job, and collect the result tables.

    python scripts/reproduce.py --list                    # the plans, their jobs, and how many are done
    python scripts/reproduce.py nt18 --gpus 0,1,2,3       # one plan on four GPUs, one job per GPU at a time
    python scripts/reproduce.py all --gpus 0,1            # every downstream plan
    python scripts/reproduce.py nt18 --dry-run            # print the commands without running them
    python scripts/reproduce.py nt18 --only "locus-2m.*"  # only the jobs whose name matches (see --dry-run --verbose)
    python scripts/reproduce.py tables                    # only rebuild results/tables.md and results/tables.json
    python scripts/reproduce.py nt18 baselines --nt18-folds 10 --jobs-per-gpu 3   # all 10 folds, three jobs per GPU

Plans:
    pretrain            the six pretraining runs (one GPU each); then use them with --models-from runs
    nt18                LoCUS-0.5M, LoCUS-2M and LoCUS-7M on the 18 NT18 tasks, 3 folds each (--nt18-folds: up to 10)
    random-init         the pretraining ablation: the LoCUS-2M architecture fine-tuned on NT18 from random initialization
    baselines           Caduceus-PS 1.9M, HyenaDNA-tiny 0.44M and HyenaDNA-small 3.3M on NT18, same protocol
                        (needs bash install.sh --baselines)
    genomic-benchmarks  LoCUS-0.5M, LoCUS-2M and LoCUS-7M on the 9 Genomic Benchmarks tasks
    tokenizer-study     the 2M twins (LoCUS-2M, twin-fixed-2m, twin-bpe-2m) on NT18, then scored on the modified test
                        sets (errors at 1% / 3%, controls, PBSIM3 reads; needs make pbsim3), each in 10 independent error draws
    heldout             LoCUS-2M on NT18 tested on chromosomes outside the pretraining chromosomes (benchmarks/nt18/heldout_split.py),
                        with the official test set scored alongside, and the same with random initialization (a model that saw
                        neither test set); folds 0-2
    context             context-length ablation: LoCUS-2M and LoCUS-2M-16k on the long-range enhancer task at 4 kb and
                        16 kb input, and LoCUS-2M-16k with a constant learning rate
    long-range          frozen embeddings on the eQTL and bulk RNA tasks of the long-range benchmark
    parsing             parse-level tests of the three segmenters: edit locality and context independence (CPU)
    bend                BEND zero-shot variant effect prediction
    efficiency          fine-tuning throughput of LoCUS and the baselines (needs bash install.sh --baselines)
    timing              wall clock of full fine-tunes: fold 0 of every NT18 task, LoCUS and the baselines, always one
                        job per GPU
    all                 every plan except pretrain

The released models are downloaded by name; --models-from DIR uses DIR/<name>/checkpoints/final.pt instead (the output
of the pretrain plan). Data preparation runs first, for the data a plan needs that does not exist yet. A job whose result
exists is skipped, so a plan can be started again after an interruption; an interrupted fine-tune resumes from its last
finished epoch. Jobs of a plan run in stages (for example embeddings before the fits on them); a stage starts when the
previous one has finished without failures. Logs: <results>/logs/<job>.log.

LoCUS-2M's NT18 runs are those of the tokenizer study (same checkpoint, recipe and seeds), so the two plans share them.
--nt18-folds sets the folds of the NT18 plans (nt18, random-init, baselines, tokenizer-study): fold k validates on fold k of
a stratified 10-fold split of the training set and trains with seed 42 + k. The paper's fine-tunes shared their GPU with up
to two others (--jobs-per-gpu 3). Repeated runs on GPU differ slightly (floating-point atomics in the covering-segment
mean), alone or sharing a GPU alike: about 0.013 MCC per task and fold on average, more on small tasks such as
promoter_tata, and about 0.01 on 18-task means."""
import argparse
import fnmatch
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locus.finetune.data import GB_TASKS, NT18_TASKS  # noqa: E402

PYTHON = sys.executable
FOLDS = 3
LOCUS = ["locus-0.5m", "locus-2m", "locus-7m"]
TWINS = {"lcp": "locus-2m", "fixed": "twin-fixed-2m", "bpe": "twin-bpe-2m"}
BASELINES = ["caduceus-ps-1.9m", "hyenadna-tiny-0.44m", "hyenadna-small-3.3m"]
RECIPE = {"locus-0.5m": "locus-0.5m", "locus-2m": "locus-2m", "locus-7m": "locus-7m", "locus-2m-16k": "locus-2m",
          "twin-fixed-2m": "locus-2m", "twin-bpe-2m": "locus-2m"}
# Micro-batches of the reported runs (gradient accumulation; the update is that of the full batch, only memory changes):
# per model, times 4 for the two Genomic Benchmarks tasks with long sequences and for 16 kb input.
MICRO_BATCHES = {"locus-0.5m": 1, "locus-2m": 2, "locus-7m": 4, "locus-2m-16k": 2, "twin-fixed-2m": 2, "twin-bpe-2m": 2}
LONG_GB_TASKS = {"drosophila_enhancers_stark", "dummy_mouse_enhancers_ensembl"}
NOISY_DRAWS = 10  # independent error draws per noisy set (benchmarks/noisy_reads/noisy_sets.py)


def drawn(tags: list) -> list:
    """The set names of every error draw of the sets `tags`: <set> (draw 0), <set>-d1, ..."""
    return [tag if draw == 0 else f"{tag}-d{draw}" for tag in tags for draw in range(NOISY_DRAWS)]


NOISY_SETS = drawn(["ont1", "ont3", "ontw1", "ontw3", "sub1", "sub3", "pbsim97"])
FROZEN = [("eqtl", 4096, ["locus-0.5m", "locus-2m", "locus-7m", "locus-2m-16k"]), ("bulk_rna", 4096, ["locus-0.5m", "locus-2m", "locus-7m"]),
          ("eqtl", 16384, ["locus-2m", "locus-2m-16k"])]
# LoCUS-2M-16k is used only where the context length is the question (context ablation, frozen eQTL at 4 kb and 16 kb)
BEND_MODELS = ["locus-0.5m", "locus-2m", "locus-7m", "twin-fixed-2m", "twin-bpe-2m"]
PRETRAIN = ["locus-0.5m", "locus-2m", "locus-7m", "locus-2m-16k", "twin-fixed-2m", "twin-bpe-2m"]
PLANS = ["pretrain", "nt18", "random-init", "baselines", "genomic-benchmarks", "tokenizer-study", "heldout", "context", "long-range", "parsing", "bend",
         "efficiency", "timing"]


@dataclass
class Job:
    name: str
    command: list
    output: Path
    gpu: bool = True
    done: Optional[Callable[[], bool]] = None  # default: the output file exists

    def is_done(self) -> bool:
        return self.done() if self.done else self.output.is_file()


def speed_complete(path: Path) -> bool:
    """The throughput run measured every model (a model that failed is stored with its error)."""
    return path.is_file() and not any("error" in entry for entry in json.loads(path.read_text()).values())


@dataclass
class Plan:
    name: str
    data: list = field(default_factory=list)    # names of the data preparations it needs (DATA)
    stages: list = field(default_factory=list)  # lists of jobs; a stage starts when the previous one has finished
    exclusive: bool = False                     # one job per GPU regardless of --jobs-per-gpu


def data_steps(data: Path) -> dict:
    """name -> (files that exist when it is complete, commands)."""
    nt18, script = data / "benchmarks/nt18", lambda path, *args: [PYTHON, str(ROOT / path), *map(str, args)]
    sets = lambda tags: [nt18 / f"{task}_{tag}_revised/metadata.json" for task in NT18_TASKS for tag in tags]
    return {
        "genome-fasta": ([data / "GRCh38.p14.fasta.gz"], [["bash", str(ROOT / "scripts/prepare_genome.sh"), "--download-only", str(data)]]),
        "genome-windows": ([data / f"GRCh38.p14.acgt.w{w}.s{w}/_SUCCESS" for w in (4096, 16384)],
                           [["bash", str(ROOT / "scripts/prepare_genome.sh"), str(data)]]),
        "nt18": ([nt18 / f"{task}_revised/metadata.json" for task in NT18_TASKS], [script("benchmarks/nt18/prepare.py", "--out", nt18)]),
        "nt18-heldout": ([data / f"benchmarks/nt18_heldout/{task}_revised/metadata.json" for task in NT18_TASKS],
                         [script("benchmarks/nt18/heldout_split.py", "--data-root", nt18, "--out", data / "benchmarks/nt18_heldout")]),
        "genomic-benchmarks": ([data / "benchmarks/genomic_benchmarks/datasets.json"],
                               [script("benchmarks/genomic_benchmarks/prepare.py", "--out", data / "benchmarks/genomic_benchmarks")]),
        "error-sets": (sets(drawn(["ont1", "ont3"])), [script("benchmarks/noisy_reads/make_errors.py", "--rates", 1, 3, "--draws", NOISY_DRAWS,
                                                              "--data-root", nt18)]),
        "control-sets": (sets(drawn(["sub1", "sub3", "ontw1", "ontw3"])), [script("benchmarks/noisy_reads/make_controls.py", "--draws", NOISY_DRAWS,
                                                                                  "--data-root", nt18)]),
        "pbsim-sets": (sets(drawn(["pbsim97"])), [script("benchmarks/noisy_reads/simulate_pbsim.py", "--accuracies", 97, "--draws", NOISY_DRAWS,
                                                         "--data-root", nt18)]),
        "enhancer": ([data / "benchmarks/long_range/datasets.json"],
                     [script("benchmarks/long_range/prepare_enhancer.py", "--fasta", data / "GRCh38.p14.fasta.gz", "--output", data / "benchmarks/long_range")]),
        "long-range-frozen": ([data / f"benchmarks/long_range/lrb_{task}_{length}/rows.json.gz" for task in ("eqtl", "bulk_rna") for length in ("4k", "16k")],
                              [script("benchmarks/long_range/frozen.py", "prepare", "--task", task, "--data", data / "benchmarks/long_range",
                                      "--fasta", data / "GRCh38.p14.fasta.gz", "--hg19", data / "hg19.fa.gz") for task in ("eqtl", "bulk_rna")]),
        "bend": ([data / "benchmarks/bend/prepared.json"],
                 [script("benchmarks/bend/zero_shot_vep.py", "prepare", "--data", data / "benchmarks/bend", "--bed-dir", data / "bend",
                         "--fasta", data / "GRCh38.p14.fasta.gz")]),
    }


DATA_ORDER = ["genome-fasta", "genome-windows", "nt18", "nt18-heldout", "genomic-benchmarks", "error-sets", "control-sets", "pbsim-sets",
              "enhancer", "long-range-frozen", "bend"]


def build_plans(arguments) -> dict:
    results, data = arguments.results, arguments.data
    nt18, gb, lrb = data / "benchmarks/nt18", data / "benchmarks/genomic_benchmarks", data / "benchmarks/long_range"
    script = lambda path, *args: [PYTHON, str(ROOT / path), *map(str, args)]
    pretrained = lambda name: str(arguments.models_from / name / "checkpoints/final.pt") if arguments.models_from else name
    nt18_folds = range(getattr(arguments, "nt18_folds", FOLDS))

    def finetune(model, data_root, task, fold, output, micro_batches, recipe=None, save_best=False, random_init=False) -> Job:
        command = script("scripts/finetune.py", "--pretrained", pretrained(model), "--recipe", ROOT / "configs/finetune" / f"{recipe or RECIPE[model]}.yaml",
                         "--data-root", data_root, "--task", task, "--fold", fold, "--output", output, "--micro-batches", micro_batches)
        return Job(f"{model}.{output.parent.parent.name}.{task}.fold{fold}",
                   command + (["--save-best"] if save_best else []) + (["--random-init"] if random_init else []), output)

    def baseline(name, task, fold, output, save_best=False) -> Job:
        return Job(f"{name}.nt18.{task}.fold{fold}", script("scripts/finetune_baseline.py", "--baseline", name, "--data-root", nt18, "--task", task,
                                                             "--fold", fold, "--output", output, *(["--save-best"] if save_best else [])), output)

    def twin_job(twin, task, fold) -> Job:
        model = TWINS[twin]
        output = results / "tokenizer_study" / twin / "nt18" / f"{task}_revised" / f"fold-{fold:02d}.json"
        job = finetune(model, nt18, f"{task}_revised", fold, output, MICRO_BATCHES[model], save_best=True)
        job.name = f"{model}.tokenizer_study.{task}.fold{fold}"
        return job

    def score_job(twin, task, fold) -> Job:
        result = results / "tokenizer_study" / twin / "nt18" / f"{task}_revised" / f"fold-{fold:02d}.json"
        scored = result.with_name(f"fold-{fold:02d}.scored.json")
        complete = lambda: scored.is_file() and set(NOISY_SETS) <= set(json.loads(scored.read_text())["sets"])
        return Job(f"{TWINS[twin]}.score.{task}.fold{fold}", script("benchmarks/noisy_reads/score.py", result, "--sets", *NOISY_SETS, "--data-root", nt18),
                   scored, done=complete)

    plans = {}
    plans["pretrain"] = Plan("pretrain", ["genome-windows"], [[Job(f"pretrain.{name}", script("scripts/pretrain.py", "--config", ROOT / "configs/pretrain" / f"{name}.yaml",
                                                                                         "--output-dir", arguments.runs / name, "--resume"),
                                                                    arguments.runs / name / "checkpoints/final.pt") for name in PRETRAIN]])
    nt18_jobs = [finetune(model, nt18, f"{task}_revised", fold, results / model / "nt18" / f"{task}_revised" / f"fold-{fold:02d}.json", MICRO_BATCHES[model])
                 for model in ("locus-0.5m", "locus-7m") for task in NT18_TASKS for fold in nt18_folds]
    lcp_jobs = [twin_job("lcp", task, fold) for task in NT18_TASKS for fold in nt18_folds]
    plans["nt18"] = Plan("nt18", ["nt18"], [nt18_jobs + lcp_jobs])
    random_init = [finetune("locus-2m", nt18, f"{task}_revised", fold, results / "locus-2m-random-init" / "nt18" / f"{task}_revised" / f"fold-{fold:02d}.json",
                            MICRO_BATCHES["locus-2m"], random_init=True) for task in NT18_TASKS for fold in nt18_folds]
    for job in random_init:
        job.name = job.name.replace("locus-2m.", "locus-2m-random-init.", 1)
    plans["random-init"] = Plan("random-init", ["nt18"], [random_init])
    # the selected weights are kept, so the baselines can also be scored on the modified test sets (benchmarks/noisy_reads)
    plans["baselines"] = Plan("baselines", ["nt18"], [[baseline(name, f"{task}_revised", fold, results / name / "nt18" / f"{task}_revised" / f"fold-{fold:02d}.json",
                                                                save_best=True)
                                                       for name in BASELINES for task in NT18_TASKS for fold in nt18_folds]])
    plans["genomic-benchmarks"] = Plan("genomic-benchmarks", ["genomic-benchmarks"], [[
        finetune(model, gb, f"gb_{task}", fold, results / model / "genomic_benchmarks" / f"gb_{task}" / f"fold-{fold:02d}.json",
                 MICRO_BATCHES[model] * (4 if task in LONG_GB_TASKS else 1))
        for model in LOCUS for task in GB_TASKS for fold in range(FOLDS)]])
    plans["tokenizer-study"] = Plan("tokenizer-study", ["nt18", "error-sets", "control-sets", "pbsim-sets"], [
        lcp_jobs + [twin_job(twin, task, fold) for twin in ("fixed", "bpe") for task in NT18_TASKS for fold in nt18_folds],
        [score_job(twin, task, fold) for twin in TWINS for task in NT18_TASKS for fold in nt18_folds]])
    heldout = []
    for name, random in (("locus-2m", False), ("locus-2m-random-init", True)):  # random init: saw neither test set in pretraining
        for task in NT18_TASKS:
            for fold in range(FOLDS):
                job = finetune("locus-2m", data / "benchmarks/nt18_heldout", f"{task}_revised", fold,
                               results / name / "nt18_heldout" / f"{task}_revised" / f"fold-{fold:02d}.json", MICRO_BATCHES["locus-2m"],
                               random_init=random)
                job.command += ["--extra-test", f"official={nt18 / f'{task}_revised'}"]
                job.name = job.name.replace("locus-2m.", f"{name}.", 1)
                heldout.append(job)
    plans["heldout"] = Plan("heldout", ["nt18", "nt18-heldout"], [heldout])
    context = []
    for model, recipe, folder in (("locus-2m", "long-range-enhancer-ep20", "long_range_ep20"), ("locus-2m-16k", "long-range-enhancer-ep20", "long_range_ep20"),
                                  ("locus-2m-16k", "long-range-enhancer-ep20-constant", "long_range_ep20_constant")):
        for length, factor in (("4k", 1), ("16k", 4)):
            for fold in range(FOLDS):
                output = results / model / folder / f"lrb_enhancer_{length}" / f"fold-{fold:02d}.json"
                job = finetune(model, lrb, f"lrb_enhancer_{length}", fold, output, MICRO_BATCHES[model] * factor, recipe=recipe)
                job.name = f"{model}.{folder}.{length}.fold{fold}"
                context.append(job)
    plans["context"] = Plan("context", ["genome-fasta", "enhancer"], [context])
    frozen_dir = results / "long_range/frozen"
    tag = lambda length: "4k" if length == 4096 else "16k"
    embeds = [Job(f"embed.{name}.{task}.{tag(length)}", script("benchmarks/long_range/frozen.py", "embed", "--task", task, "--length", length, "--pretrained", pretrained(name),
                                                              "--name", name, "--data", lrb, "--results", frozen_dir),
                  frozen_dir / f"emb_{name}_lrb_{task}_{tag(length)}.npz") for task, length, names in FROZEN for name in names]
    fits = [Job(f"fit.{name}.{task}.{tag(length)}", script("benchmarks/long_range/frozen.py", "fit", "--task", task, "--length", length, "--name", name,
                                                          "--data", lrb, "--results", frozen_dir),
                frozen_dir / f"fit_{name}_lrb_{task}_{tag(length)}.json", gpu=False) for task, length, names in FROZEN for name in names + (["tss"] if task == "eqtl" and length == 4096 else [])]
    plans["long-range"] = Plan("long-range", ["genome-fasta", "long-range-frozen"], [embeds, fits])
    locality = results / "parsing/locality.json"
    plans["parsing"] = Plan("parsing", ["genome-windows"], [[Job("parsing.locality", script("benchmarks/parsing/locality.py", "--windows",
                                                                                          data / "GRCh38.p14.acgt.w4096.s4096", "--output", locality),
                                                                 locality, gpu=False)]])
    bend_dir = results / "bend"
    plans["bend"] = Plan("bend", ["genome-fasta", "bend"], [
        [Job(f"bend.{name}", script("benchmarks/bend/zero_shot_vep.py", "embed", "--pretrained", pretrained(name), "--name", name, "--data", data / "benchmarks/bend",
                                    "--results", bend_dir), bend_dir / f"dist_{name}.npz") for name in BEND_MODELS],
        [Job("bend.analyze", script("benchmarks/bend/zero_shot_vep.py", "analyze", "--data", data / "benchmarks/bend", "--results", bend_dir),
             bend_dir / "results.json", gpu=False, done=lambda: False)]])
    speed = results / "efficiency/finetune_speed.json"
    plans["efficiency"] = Plan("efficiency", ["nt18"], [[Job("efficiency", script("benchmarks/efficiency/finetune_speed.py", "--data-root", nt18, "--output", speed,
                                                                                  *(["--models-from", arguments.models_from] if arguments.models_from else [])),
                                                             speed, done=lambda: speed_complete(speed))]])
    timing = [finetune(model, nt18, f"{task}_revised", 0, results / "timing" / model / "nt18" / f"{task}_revised" / "fold-00.json", MICRO_BATCHES[model])
              for model in LOCUS for task in NT18_TASKS]
    timing += [baseline(name, f"{task}_revised", 0, results / "timing" / name / "nt18" / f"{task}_revised" / "fold-00.json") for name in BASELINES for task in NT18_TASKS]
    for job in timing:
        job.name = "timing." + job.name
    plans["timing"] = Plan("timing", ["nt18"], [timing], exclusive=True)
    return plans


def run_data(names: list, data: Path, dry_run: bool, logs: Path) -> None:
    steps = data_steps(data)
    for name in [n for n in DATA_ORDER if n in names]:
        markers, commands = steps[name]
        if all(path.is_file() for path in markers):
            continue
        for command in commands:
            print(f"[data] {name}: {shlex.join(command)}", flush=True)
            if not dry_run:
                logs.mkdir(parents=True, exist_ok=True)
                with open(logs / f"data.{name}.log", "a") as log:
                    if subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode != 0:
                        sys.exit(f"data preparation {name} failed; see {logs / f'data.{name}.log'}")


def run_stage(jobs: list, gpu_slots: list, cpu_slots: int, dry_run: bool, logs: Path, verbose: bool = False) -> list:
    """Runs the jobs that are not done; returns the failed ones."""
    pending = [job for job in jobs if not job.is_done()]
    if dry_run:
        for job in pending:
            print(f"[{'gpu' if job.gpu else 'cpu'}] " + (f"{job.name}: " if verbose else "") + shlex.join(job.command))
        return []
    logs.mkdir(parents=True, exist_ok=True)
    free = {"gpu": list(gpu_slots), "cpu": [""] * cpu_slots}
    running, failed = [], []
    while pending or running:
        for job in list(pending):
            kind = "gpu" if job.gpu else "cpu"
            if not free[kind]:
                continue
            slot = free[kind].pop(0)
            pending.remove(job)
            log = open(logs / f"{job.name}.log", "a")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot)
            # as in the paper's runs: expandable allocator segments, so that three fine-tunes fit on one 24 GB GPU (memory layout
            # only; the computation is unchanged)
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            print(f"start {job.name}" + (f" on GPU {slot}" if job.gpu else ""), flush=True)
            running.append((job, kind, slot, log, subprocess.Popen(job.command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)))
        time.sleep(2)
        for item in list(running):
            job, kind, slot, log, process = item
            if process.poll() is None:
                continue
            running.remove(item)
            log.close()
            free[kind].append(slot)
            ok = process.returncode == 0 and job.is_done()
            print(f"{'done' if ok else 'FAILED'} {job.name}" + ("" if ok else f" (exit {process.returncode}; log {logs / f'{job.name}.log'})"), flush=True)
            if not ok:
                failed.append(job)
    return failed


def collect(results: Path, dry_run: bool, nt18_folds: int = FOLDS) -> None:
    command = [PYTHON, str(ROOT / "scripts/collect_results.py"), "--results", str(results), "--output", str(results / "tables"),
               "--nt18-folds", str(nt18_folds)]
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plans", nargs="*", metavar="PLAN", help=f"{', '.join(PLANS)}, all, or tables")
    parser.add_argument("--list", action="store_true", help="list the plans with their number of jobs and how many are done")
    parser.add_argument("--gpus", help="comma-separated GPU indices (default: every visible GPU)")
    parser.add_argument("--jobs-per-gpu", type=int, default=1, help="fine-tunes sharing one GPU (the paper: up to 3)")
    parser.add_argument("--nt18-folds", type=int, default=FOLDS, choices=range(1, 11), metavar="{1..10}",
                        help=f"folds of the NT18 plans (default {FOLDS}; the split has 10)")
    parser.add_argument("--cpu-jobs", type=int, default=2, help="parallel CPU jobs (frozen-embedding fits, parsing tests, BEND analysis)")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs", help="output directory of the pretrain plan")
    parser.add_argument("--models-from", type=Path, help="use DIR/<name>/checkpoints/final.pt instead of the released models")
    parser.add_argument("--only", nargs="+", metavar="PATTERN", help="run only the jobs whose name matches one of these shell patterns")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="with --dry-run: also print the job names")
    arguments = parser.parse_args()
    arguments.results, arguments.data, arguments.runs = arguments.results.resolve(), arguments.data.resolve(), arguments.runs.resolve()
    if arguments.models_from:
        arguments.models_from = arguments.models_from.resolve()
    plans = build_plans(arguments)

    if arguments.list or not arguments.plans:
        for name in PLANS:
            jobs = [job for stage in plans[name].stages for job in stage]
            print(f"{name:20s} {sum(job.is_done() for job in jobs):4d} / {len(jobs):4d} jobs done")
        return
    names = PLANS[1:] if "all" in arguments.plans else [n for n in arguments.plans if n != "tables"]
    unknown = [n for n in names if n not in plans]
    if unknown:
        parser.error(f"unknown plan(s) {', '.join(unknown)}; choose from {', '.join(PLANS)}, all, tables")
    if arguments.gpus is not None:
        gpus = [g for g in arguments.gpus.split(",") if g]
    else:
        import torch

        gpus = [str(i) for i in range(torch.cuda.device_count())]
    if not gpus and any(job.gpu for n in names for stage in plans[n].stages for job in stage):
        sys.exit("no GPU found; pass --gpus")

    logs = arguments.results / "logs"
    run_data(sorted({d for n in names for d in plans[n].data}, key=DATA_ORDER.index), arguments.data, arguments.dry_run, logs)
    seen, failed = set(), []
    for name in names:
        per_gpu = 1 if plans[name].exclusive else arguments.jobs_per_gpu
        for stage in plans[name].stages:
            jobs = [job for job in stage if job.name not in seen]  # jobs shared by two plans run once
            seen.update(job.name for job in jobs)
            if arguments.only:
                jobs = [job for job in jobs if any(fnmatch.fnmatchcase(job.name, pattern) for pattern in arguments.only)]
            stage_failed = run_stage(jobs, [g for g in gpus for _ in range(per_gpu)], arguments.cpu_jobs, arguments.dry_run, logs, arguments.verbose)
            failed += stage_failed
            if stage_failed:
                print(f"{name}: {len(stage_failed)} job(s) failed; the plan's later stages are skipped", flush=True)
                break
    if names != ["pretrain"]:
        collect(arguments.results, arguments.dry_run, arguments.nt18_folds)
    if failed:
        sys.exit(f"{len(failed)} job(s) failed: " + ", ".join(job.name for job in failed))


if __name__ == "__main__":
    main()
