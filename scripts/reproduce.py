"""Reproduce the experiments of the paper: prepare the data, run every job, and collect the result tables.

    python scripts/reproduce.py --list                    # the plans, their jobs, and how many are done
    python scripts/reproduce.py nt18 --gpus 0,1,2,3       # one plan on four GPUs, one job per GPU at a time
    python scripts/reproduce.py all --gpus 0,1            # every downstream plan
    python scripts/reproduce.py nt18 --dry-run            # print the commands without running them
    python scripts/reproduce.py nt18 --only "locus-2m.*"  # only the jobs whose name matches (see --dry-run --verbose)
    python scripts/reproduce.py tables                    # only rebuild results/tables.md and results/tables.json
    python scripts/reproduce.py nt18 baselines --jobs-per-gpu 3   # three fine-tunes per GPU, as in the paper
    python scripts/reproduce.py nt18 --nt18-folds 3                # only folds 0-2 of the NT18 plans (quicker check)

Plans:
    pretrain            the four pretraining runs (one GPU each); then use them with --models-from runs
    nt18                LoCUS-0.5M and LoCUS-2M on the 18 NT18 tasks, 10 folds each (--nt18-folds: fewer)
    random-init         the pretraining ablation: the LoCUS-2M architecture fine-tuned on NT18 from random initialization
    baselines           Caduceus-PS 0.47M and 1.9M (the sizes of LoCUS-0.5M and LoCUS-2M) on NT18 with the protocol of LoCUS,
                        in bfloat16 like LoCUS (configs/finetune/caduceus-ps-<size>-bf16.yaml), 10 folds, fold by fold (needs
                        bash install.sh --baselines)
    baselines-fp32      the same runs in float32 (configs/finetune/caduceus-ps-<size>.yaml): the precision check of the
                        supplementary table
    baselines-authors   Caduceus-PS 0.47M and 1.9M (the sizes of LoCUS-0.5M and LoCUS-2M) on NT18 with Caduceus's own fine-tuning
                        settings (configs/finetune/caduceus-ps-<size>-authors.yaml; the supplementary table), 10 folds, fold by fold
    baselines-authors-bf16
                        the folds of Caduceus-PS (own settings) whose fp16 training overflowed (non-finite steps or
                        validation losses), again in bfloat16 (configs/finetune/caduceus-ps-<size>-authors-bf16.yaml); after
                        baselines-authors, which it reads; the table uses these runs for those folds
    genomic-benchmarks  LoCUS-0.5M and LoCUS-2M on the 9 Genomic Benchmarks tasks, 10 folds each (--other-folds: fewer)
    tokenizer-study     the 2M twins (LoCUS-2M, twin-fixed-2m, twin-bpe-2m) on NT18, then scored on the modified test
                        sets (errors at 1% / 3%, controls, PBSIM3 reads; needs make pbsim3), each in 10 independent error draws
    gue-splice          the tokenizer study repeated on the GUE splice-site task (benchmarks/gue/prepare.py; needs make pbsim3): the 2M twins
                        fine-tuned on its clean training data (folds as --nt18-folds), then scored on the same kinds of modified
                        test sets, also without the test rows that share a splice site with NT18 (benchmarks/gue/independent_sets.py)
    real-reads          the tokenizer study's NT18 folds scored on real HG002 nanopore reads of their test windows
                        (benchmarks/noisy_reads/real_reads.py; streams the public CRAM files, needs network; stops unless
                        the sets are identical to the paper's, benchmarks/noisy_reads/real_reads_sha256.tsv)
    single-nucleotide   Caduceus-PS 1.9M's runs (single nucleotides, folds of the baselines plan) scored on the same modified test
                        sets and real reads: the loss that errors cause without a tokenizer (needs bash install.sh --baselines)
    single-nucleotide-fp32
                        the same for Caduceus-PS 1.9M's float32 runs (folds of the baselines-fp32 plan): the precision check of the
                        single-nucleotide drops (supplementary table)
    heldout             LoCUS-2M on NT18 tested on chromosomes outside the pretraining chromosomes (benchmarks/nt18/heldout_split.py),
                        with the official test set scored alongside, and the same with random initialization (a model that saw
                        neither test set); 10 folds (--other-folds: fewer)
    parsing             parse-level tests of the three segmenters (edit locality, context independence), the statistics
                        of the LCP parse (segment lengths and coverage, worst-case edit locality at levels 1 and 2), and how
                        much the segment boundaries tell the pretrained models about masked bases (CPU)
    stability           embedding stability: how far one edit propagates through the frozen 2M twins (one GPU)
    efficiency          fine-tuning throughput of LoCUS and Caduceus-PS (needs bash install.sh --baselines), and the time to
                        segment every NT18 record with each segmenter (dataset creation, CPU)
    timing              wall clock of full fine-tunes: fold 0 of every NT18 task, LoCUS and Caduceus-PS, always one
                        job per GPU; Caduceus-PS with the recipe of plan baselines (bfloat16, the precision of LoCUS) and of
                        plan baselines-fp32 (float32)
    split-pretrain      the pretraining-split ablation: LoCUS-0.5M, LoCUS-2M and the two 2M twins pretrained on chromosomes 1-19 and
                        22, without the NT18 test chromosomes 20 and 21 (configs/pretrain/<model>-no-chr20-21.yaml), one GPU each;
                        the BPE twin's tokenizer is data/tokenizers/bpe-4096-1gb-seed42-no-chr20-21.json (retrained by the data
                        step only if it is missing, about 64 GB of RAM)
    split-nt18          NT18 for the four split-ablation models, 10 folds, with the recipe and micro-batches of their released
                        counterparts: the released split models (locus/hub.py, downloaded by name), or the checkpoints of your own
                        split-pretrain (--models-from runs, or in the same command as split-pretrain); the tables compare them
                        with the released models' runs (plans nt18 and tokenizer-study)
    split-score         the three 2M models of split-nt18 scored on the tokenizer study's modified test sets and real reads
    all                 every plan except pretrain and the split-ablation plans

The released models are downloaded by name; --models-from DIR uses DIR/<name>/checkpoints/final.pt instead (the output
of the pretrain plan). Data preparation runs first, for the data a plan needs that does not exist yet. A job whose result
exists is skipped, so a plan can be started again after an interruption; an interrupted fine-tune resumes from its last
finished epoch. Jobs of a plan run in stages (for example embeddings before the fits on them); a stage starts when the
previous one has finished without failures. Logs: <results>/logs/<job>.log.

LoCUS-2M's NT18 runs are those of the tokenizer study (same checkpoint, recipe and seeds), so the two plans share them.
--nt18-folds (default 10, as in the paper) sets the folds of the NT18 plans (nt18, random-init, baselines, baselines-fp32,
baselines-authors, tokenizer-study, real-reads, single-nucleotide, split-nt18, split-score) and of gue-splice: fold k validates on fold k of a
stratified 10-fold split of the training set and trains with seed 42 + k. --other-folds (default 10) does the same for genomic-benchmarks and heldout. The paper's fine-tunes shared their GPU with up
to two others (--jobs-per-gpu 3). Repeated runs on GPU differ slightly (floating-point atomics in the covering-segment
mean), alone or sharing a GPU alike: about 0.013 MCC per task and fold on average, more on small tasks such as
promoter_tata, and about 0.01 on 18-task means."""
import argparse
import fnmatch
import json
import math
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
OTHER_FOLDS = 10  # Genomic Benchmarks and the held-out-chromosome check (--other-folds)
NT18_FOLDS = 10  # NT18 plans and gue-splice (--nt18-folds)
LOCUS = ["locus-0.5m", "locus-2m"]
TWINS = {"lcp": "locus-2m", "fixed": "twin-fixed-2m", "bpe": "twin-bpe-2m"}
# Caduceus-PS at the sizes of LoCUS-0.5M and LoCUS-2M with the protocol of LoCUS, in bfloat16 like LoCUS (plan baselines;
# configs/finetune/<run>.yaml): run name -> baseline model
BASELINES = ["caduceus-ps-0.47m-bf16", "caduceus-ps-1.9m-bf16"]
BASELINE_MODEL = {name: name.removesuffix("-bf16") for name in BASELINES}
# the same recipes in float32 (plan baselines-fp32, the precision check; configs/finetune/<model>.yaml, the model's own recipe)
FP32_BASELINES = ["caduceus-ps-1.9m", "caduceus-ps-0.47m"]
# Caduceus-PS at the sizes of LoCUS-0.5M and LoCUS-2M with Caduceus's own fine-tuning settings (configs/finetune/<run>.yaml;
# supplementary table)
AUTHORS_BASELINES = ["caduceus-ps-0.47m-authors", "caduceus-ps-1.9m-authors"]
BASELINE_MODEL.update({name: name.removesuffix("-authors") for name in AUTHORS_BASELINES})
# folds of the own-settings runs whose fp16 training overflowed, run again in bfloat16 (plan baselines-authors-bf16)
AUTHORS_BF16 = {run: f"{run}-bf16" for run in AUTHORS_BASELINES}
BASELINE_MODEL.update({rerun: BASELINE_MODEL[run] for run, rerun in AUTHORS_BF16.items()})
# single-nucleotide models scored on the modified test sets and real reads (plan single-nucleotide): Caduceus-PS 1.9M's runs of
# plan baselines
SINGLE_NUCLEOTIDE = ["caduceus-ps-1.9m-bf16"]
# the same for the float32 runs of plan baselines-fp32 (plan single-nucleotide-fp32, the precision check)
SINGLE_NUCLEOTIDE_FP32 = ["caduceus-ps-1.9m"]
RECIPE = {"locus-0.5m": "locus-0.5m", "locus-2m": "locus-2m", "twin-fixed-2m": "locus-2m", "twin-bpe-2m": "locus-2m"}
# Micro-batches of the reported runs (gradient accumulation; the update is that of the full batch, only memory changes):
# per model, times 4 for the two Genomic Benchmarks tasks with long sequences.
MICRO_BATCHES = {"locus-0.5m": 1, "locus-2m": 2, "twin-fixed-2m": 2, "twin-bpe-2m": 2}
LONG_GB_TASKS = {"drosophila_enhancers_stark", "dummy_mouse_enhancers_ensembl"}
NOISY_DRAWS = 10  # independent error draws per noisy set (benchmarks/noisy_reads/noisy_sets.py)


def drawn(tags: list) -> list:
    """The set names of every error draw of the sets `tags`: <set> (draw 0), <set>-d1, ..."""
    return [tag if draw == 0 else f"{tag}-d{draw}" for tag in tags for draw in range(NOISY_DRAWS)]


NOISY_SETS = drawn(["ont1", "ont3", "ontw1", "ontw3", "sub1", "sub3", "pbsim97"])
# real reads (benchmarks/noisy_reads/real_reads.py): the reference sequences of the kept windows, then 10 reads per window in two basecalls
REAL_TASKS = NT18_TASKS  # every NT18 test record names its window on chromosome 20 or 21
REAL_SETS = ["realref"] + drawn(["realsup", "realhac"])
PRETRAIN = ["locus-0.5m", "locus-2m", "twin-fixed-2m", "twin-bpe-2m"]
# Pretraining-split ablation: the four models pretrained without chromosomes 20 and 21 (the NT18 test chromosomes), each
# fine-tuned with the recipe and micro-batches of its released counterpart (model -> released model); the 2M ones are also
# scored on the tokenizer study's modified test sets and real reads.
SPLIT = {"locus-0.5m-no-chr20-21": "locus-0.5m", "locus-2m-no-chr20-21": "locus-2m", "twin-fixed-2m-no-chr20-21": "twin-fixed-2m",
         "twin-bpe-2m-no-chr20-21": "twin-bpe-2m"}
SPLIT_SCORED = ["locus-2m-no-chr20-21", "twin-fixed-2m-no-chr20-21", "twin-bpe-2m-no-chr20-21"]
SPLIT_PLANS = ["split-pretrain", "split-nt18", "split-score"]
SPLIT_TOKENIZER = ROOT / "data/tokenizers/bpe-4096-1gb-seed42-no-chr20-21.json"
PLANS = ["pretrain", "nt18", "random-init", "baselines", "baselines-fp32", "baselines-authors", "baselines-authors-bf16", "genomic-benchmarks", "tokenizer-study", "gue-splice", "real-reads", "single-nucleotide", "single-nucleotide-fp32", "heldout", "parsing",
         "stability", "efficiency", "timing"]


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
    gue, gue_independent = data / "benchmarks/gue", data / "benchmarks/gue_independent"
    return {
        "genome-fasta": ([data / "GRCh38.p14.fasta.gz"], [["bash", str(ROOT / "scripts/prepare_genome.sh"), "--download-only", str(data)]]),
        "genome-windows": ([data / "GRCh38.p14.acgt.w4096.s4096/_SUCCESS"],
                           [["bash", str(ROOT / "scripts/prepare_genome.sh"), str(data)]]),
        "bpe-no-chr20-21": ([SPLIT_TOKENIZER], [script("scripts/train_bpe.py", "--config", ROOT / "configs/pretrain/twin-bpe-2m-no-chr20-21.yaml",
                                                       "--genome-windows", data / "GRCh38.p14.acgt.w4096.s4096", "--output", SPLIT_TOKENIZER)]),
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
        "gue": ([gue / "gue_splice/metadata.json"], [script("benchmarks/gue/prepare.py", "--out", gue, "--nt18", nt18)]),
        "gue-sets": ([gue / f"gue_splice_{tag}/metadata.json" for tag in drawn(["ont1", "ont3", "sub1", "sub3", "ontw1", "ontw3", "pbsim97"])],
                     [script("benchmarks/noisy_reads/make_errors.py", "--tasks", "gue_splice", "--rates", 1, 3, "--draws", NOISY_DRAWS, "--data-root", gue),
                      script("benchmarks/noisy_reads/make_controls.py", "--tasks", "gue_splice", "--draws", NOISY_DRAWS, "--data-root", gue),
                      script("benchmarks/noisy_reads/simulate_pbsim.py", "--tasks", "gue_splice", "--accuracies", 97, "--draws", NOISY_DRAWS,
                             "--data-root", gue)]),
        "gue-independent": ([gue_independent / "metadata.json"], [script("benchmarks/gue/independent_sets.py", "--data-root", gue, "--out", gue_independent)]),
        "real-reads": ([data / "benchmarks/real_reads/summary.json"],
                       [script("benchmarks/noisy_reads/real_reads.py", "--nt18", nt18, "--genome", data / "GRCh38.p14.fasta.gz",
                               "--out", data / "benchmarks/real_reads", "--expected", ROOT / "benchmarks/noisy_reads/real_reads_sha256.tsv")]),
    }


DATA_ORDER = ["genome-fasta", "genome-windows", "bpe-no-chr20-21", "nt18", "nt18-heldout", "genomic-benchmarks", "error-sets", "control-sets", "pbsim-sets", "gue", "gue-sets",
              "gue-independent", "real-reads"]


def fp16_overflowed(directory: Path, folds) -> list:
    """(task, fold) of the finished fp16 runs in `directory` (<run>/nt18) that overflowed: a skipped non-finite training step,
    or a non-finite validation loss in some epoch (no_finite_validation_loss: in every epoch), fold by fold."""
    found = []
    for fold in folds:
        for task in NT18_TASKS:
            path = directory / f"{task}_revised" / f"fold-{fold:02d}.json"
            if path.is_file():
                report = json.loads(path.read_text())
                losses = [entry["validation_loss"] for entry in report.get("history", [])]
                if report.get("nonfinite_steps") or report.get("no_finite_validation_loss") or not all(math.isfinite(x) for x in losses):
                    found.append((task, fold))
    return found


def build_plans(arguments) -> dict:
    results, data = arguments.results, arguments.data
    nt18, gb = data / "benchmarks/nt18", data / "benchmarks/genomic_benchmarks"
    script = lambda path, *args: [PYTHON, str(ROOT / path), *map(str, args)]
    pretrained = lambda name: str(arguments.models_from / name / "checkpoints/final.pt") if arguments.models_from else name
    nt18_folds = range(getattr(arguments, "nt18_folds", NT18_FOLDS))
    other_folds = range(getattr(arguments, "other_folds", OTHER_FOLDS))

    def finetune(model, data_root, task, fold, output, micro_batches, recipe=None, save_best=False, random_init=False) -> Job:
        command = script("scripts/finetune.py", "--pretrained", pretrained(model), "--recipe", ROOT / "configs/finetune" / f"{recipe or RECIPE[model]}.yaml",
                         "--data-root", data_root, "--task", task, "--fold", fold, "--output", output, "--micro-batches", micro_batches)
        return Job(f"{model}.{output.parent.parent.name}.{task}.fold{fold}",
                   command + (["--save-best"] if save_best else []) + (["--random-init"] if random_init else []), output)

    def baseline(name, task, fold, output, save_best=False) -> Job:
        # every run names its recipe: configs/finetune/<run>.yaml (the float32 runs: the model's name)
        return Job(f"{name}.nt18.{task}.fold{fold}", script("scripts/finetune_baseline.py", "--baseline", BASELINE_MODEL.get(name, name), "--data-root", nt18,
                                                             "--task", task, "--fold", fold, "--output", output,
                                                             "--recipe", ROOT / "configs/finetune" / f"{name}.yaml",
                                                             *(["--save-best"] if save_best else [])), output)

    def twin_job(twin, task, fold) -> Job:
        model = TWINS[twin]
        output = results / "tokenizer_study" / twin / "nt18" / f"{task}_revised" / f"fold-{fold:02d}.json"
        job = finetune(model, nt18, f"{task}_revised", fold, output, MICRO_BATCHES[model], save_best=True)
        job.name = f"{model}.tokenizer_study.{task}.fold{fold}"
        return job

    def score_job(twin, task, fold, benchmark="nt18", data_root=nt18, sets=NOISY_SETS, study=None) -> Job:
        """Scores a saved twin fold on modified test sets; with `study`, the score file goes to results/<study>/... (score.py --output-dir)."""
        folder = f"{task}_revised" if benchmark == "nt18" else task
        result = results / "tokenizer_study" / twin / benchmark / folder / f"fold-{fold:02d}.json"
        output_dir = results / study / twin / benchmark / folder if study else None
        scored = (output_dir or result.parent) / f"fold-{fold:02d}.scored.json"
        complete = lambda: scored.is_file() and set(sets) <= set(json.loads(scored.read_text())["sets"])
        kind = "score" + ("" if benchmark == "nt18" else f"_{benchmark}") + (f"_{study}" if study else "")
        command = script("benchmarks/noisy_reads/score.py", result, "--sets", *sets, "--data-root", data_root)
        return Job(f"{TWINS[twin]}.{kind}.{task}.fold{fold}", command + (["--output-dir", str(output_dir)] if study else []), scored, done=complete)

    def baseline_score_job(name, task, fold, data_root, sets, study) -> Job:
        """Scores a saved single-nucleotide baseline fold (baselines plan) into results/<study>/<name>/nt18/<task>_revised."""
        result = results / name / "nt18" / f"{task}_revised" / f"fold-{fold:02d}.json"
        output_dir = results / study / name / "nt18" / f"{task}_revised"
        scored = output_dir / f"fold-{fold:02d}.scored.json"
        complete = lambda: scored.is_file() and set(sets) <= set(json.loads(scored.read_text())["sets"])
        return Job(f"{name}.score_{study}.{task}.fold{fold}", script("benchmarks/noisy_reads/score.py", result, "--sets", *sets, "--data-root", data_root,
                                                                     "--output-dir", output_dir), scored, done=complete)

    def gue_twin_job(twin, fold) -> Job:
        model = TWINS[twin]
        output = results / "tokenizer_study" / twin / "gue" / "gue_splice" / f"fold-{fold:02d}.json"
        job = finetune(model, data / "benchmarks/gue", "gue_splice", fold, output, MICRO_BATCHES[model], save_best=True)
        job.name = f"{model}.tokenizer_study_gue.gue_splice.fold{fold}"
        return job

    plans = {}
    plans["pretrain"] = Plan("pretrain", ["genome-windows"], [[
        Job(f"pretrain.{name}", script("scripts/pretrain.py", "--config", ROOT / "configs/pretrain" / f"{name}.yaml", "--output-dir", arguments.runs / name,
                                       "--resume", "--set", f"data.genome_windows={data / 'GRCh38.p14.acgt.w4096.s4096'}"),
            arguments.runs / name / "checkpoints/final.pt") for name in PRETRAIN]], exclusive=True)
    nt18_jobs = [finetune(model, nt18, f"{task}_revised", fold, results / model / "nt18" / f"{task}_revised" / f"fold-{fold:02d}.json", MICRO_BATCHES[model])
                 for model in ("locus-0.5m",) for task in NT18_TASKS for fold in nt18_folds]
    lcp_jobs = [twin_job("lcp", task, fold) for task in NT18_TASKS for fold in nt18_folds]
    plans["nt18"] = Plan("nt18", ["nt18"], [nt18_jobs + lcp_jobs])
    random_init = [finetune("locus-2m", nt18, f"{task}_revised", fold, results / "locus-2m-random-init" / "nt18" / f"{task}_revised" / f"fold-{fold:02d}.json",
                            MICRO_BATCHES["locus-2m"], random_init=True) for task in NT18_TASKS for fold in nt18_folds]
    for job in random_init:
        job.name = job.name.replace("locus-2m.", "locus-2m-random-init.", 1)
    plans["random-init"] = Plan("random-init", ["nt18"], [random_init])
    # the selected weights are kept, so the baselines can also be scored on the modified test sets (benchmarks/noisy_reads); fold by
    # fold (every model and task of fold 0, then fold 1, ...), so an unfinished run leaves complete folds
    baseline_runs = [baseline(name, f"{task}_revised", fold, results / name / "nt18" / f"{task}_revised" / f"fold-{fold:02d}.json", save_best=True)
                     for fold in nt18_folds for name in BASELINES for task in NT18_TASKS]
    plans["baselines"] = Plan("baselines", ["nt18"], [baseline_runs])
    fp32_runs = [baseline(name, f"{task}_revised", fold, results / name / "nt18" / f"{task}_revised" / f"fold-{fold:02d}.json", save_best=True)
                 for name in FP32_BASELINES for task in NT18_TASKS for fold in nt18_folds]
    plans["baselines-fp32"] = Plan("baselines-fp32", ["nt18"], [fp32_runs])
    # fold by fold (every model and task of fold 0, then fold 1, ...), so an unfinished run leaves complete folds
    plans["baselines-authors"] = Plan("baselines-authors", ["nt18"], [[
        baseline(name, f"{task}_revised", fold, results / name / "nt18" / f"{task}_revised" / f"fold-{fold:02d}.json", save_best=True)
        for fold in nt18_folds for name in AUTHORS_BASELINES for task in NT18_TASKS]])
    plans["baselines-authors-bf16"] = Plan("baselines-authors-bf16", ["nt18"], [[
        baseline(rerun, f"{task}_revised", fold, results / rerun / "nt18" / f"{task}_revised" / f"fold-{fold:02d}.json", save_best=True)
        for run, rerun in AUTHORS_BF16.items() for task, fold in fp16_overflowed(results / run / "nt18", nt18_folds)]])
    plans["genomic-benchmarks"] = Plan("genomic-benchmarks", ["genomic-benchmarks"], [[
        finetune(model, gb, f"gb_{task}", fold, results / model / "genomic_benchmarks" / f"gb_{task}" / f"fold-{fold:02d}.json",
                 MICRO_BATCHES[model] * (4 if task in LONG_GB_TASKS else 1))
        for model in LOCUS for task in GB_TASKS for fold in other_folds]])
    plans["tokenizer-study"] = Plan("tokenizer-study", ["nt18", "error-sets", "control-sets", "pbsim-sets"], [
        lcp_jobs + [twin_job(twin, task, fold) for twin in ("fixed", "bpe") for task in NT18_TASKS for fold in nt18_folds],
        [score_job(twin, task, fold) for twin in TWINS for task in NT18_TASKS for fold in nt18_folds]])
    plans["gue-splice"] = Plan("gue-splice", ["nt18", "gue", "gue-sets", "gue-independent"], [
        [gue_twin_job(twin, fold) for twin in TWINS for fold in nt18_folds],
        [score_job(twin, "gue_splice", fold, "gue", data / "benchmarks/gue") for twin in TWINS for fold in nt18_folds]
        + [score_job(twin, "gue_splice", fold, "gue", data / "benchmarks/gue_independent", ["clean"] + NOISY_SETS, "gue_independent")
           for twin in TWINS for fold in nt18_folds]])
    # the twins' NT18 folds (shared with tokenizer-study) scored on the real reads
    plans["real-reads"] = Plan("real-reads", ["genome-fasta", "nt18", "real-reads"], [
        [twin_job(twin, task, fold) for twin in TWINS for task in REAL_TASKS for fold in nt18_folds],
        [score_job(twin, task, fold, data_root=data / "benchmarks/real_reads", sets=REAL_SETS, study="real_reads")
         for twin in TWINS for task in REAL_TASKS for fold in nt18_folds]])
    # single-nucleotide models (no merging of nucleotides into tokens) on the same modified test sets and real reads, as the floor
    # of the loss that errors cause without a tokenizer; their NT18 folds are those of the baselines plan
    plans["single-nucleotide"] = Plan("single-nucleotide", ["genome-fasta", "nt18", "error-sets", "control-sets", "pbsim-sets", "real-reads"], [
        [job for job in baseline_runs if job.name.split(".nt18.")[0] in SINGLE_NUCLEOTIDE],
        [baseline_score_job(name, task, fold, nt18, ["clean"] + NOISY_SETS, "single_nucleotide")
         for name in SINGLE_NUCLEOTIDE for task in NT18_TASKS for fold in nt18_folds]
        + [baseline_score_job(name, task, fold, data / "benchmarks/real_reads", REAL_SETS, "real_reads")
           for name in SINGLE_NUCLEOTIDE for task in REAL_TASKS for fold in nt18_folds]])
    # the same for the float32 runs (precision check): their NT18 folds are those of the baselines-fp32 plan
    plans["single-nucleotide-fp32"] = Plan("single-nucleotide-fp32", ["genome-fasta", "nt18", "error-sets", "control-sets", "pbsim-sets", "real-reads"], [
        [job for job in fp32_runs if job.name.split(".nt18.")[0] in SINGLE_NUCLEOTIDE_FP32],
        [baseline_score_job(name, task, fold, nt18, ["clean"] + NOISY_SETS, "single_nucleotide")
         for name in SINGLE_NUCLEOTIDE_FP32 for task in NT18_TASKS for fold in nt18_folds]
        + [baseline_score_job(name, task, fold, data / "benchmarks/real_reads", REAL_SETS, "real_reads")
           for name in SINGLE_NUCLEOTIDE_FP32 for task in REAL_TASKS for fold in nt18_folds]])
    heldout = []
    for name, random in (("locus-2m", False), ("locus-2m-random-init", True)):  # random init: saw neither test set in pretraining
        for task in NT18_TASKS:
            for fold in other_folds:
                job = finetune("locus-2m", data / "benchmarks/nt18_heldout", f"{task}_revised", fold,
                               results / name / "nt18_heldout" / f"{task}_revised" / f"fold-{fold:02d}.json", MICRO_BATCHES["locus-2m"],
                               random_init=random)
                job.command += ["--extra-test", f"official={nt18 / f'{task}_revised'}"]
                job.name = job.name.replace("locus-2m.", f"{name}.", 1)
                heldout.append(job)
    plans["heldout"] = Plan("heldout", ["nt18", "nt18-heldout"], [heldout])
    windows = data / "GRCh38.p14.acgt.w4096.s4096"
    locality, lcp_stats = results / "parsing/locality.json", results / "parsing/lcp_statistics.json"
    boundaries = results / "parsing/boundary_information.json"
    plans["parsing"] = Plan("parsing", ["genome-windows"], [[
        Job("parsing.locality", script("benchmarks/parsing/locality.py", "--windows", windows, "--output", locality), locality, gpu=False),
        Job("parsing.lcp_statistics", script("benchmarks/parsing/lcp_statistics.py", "--windows", windows, "--output", lcp_stats), lcp_stats, gpu=False),
        Job("parsing.boundary_information", script("benchmarks/parsing/boundary_information.py", "--windows", windows, "--output", boundaries),
            boundaries, gpu=False)]])
    stability = results / "embedding/stability.json"
    plans["stability"] = Plan("stability", ["genome-windows"], [[Job("stability", script(
        "benchmarks/embedding/stability.py", "--windows", windows, "--output", stability,
        *(["--models-from", arguments.models_from] if arguments.models_from else [])), stability)]], exclusive=True)
    speed = results / "efficiency/finetune_speed.json"
    segmentation = results / "efficiency/segmentation_time.json"
    # the segmentation time first, alone, on an otherwise idle machine; then the GPU throughput
    plans["efficiency"] = Plan("efficiency", ["nt18"], [
        [Job("efficiency.segmentation", script("benchmarks/efficiency/segmentation_time.py", "--data-root", nt18, "--output", segmentation),
             segmentation, gpu=False)],
        [Job("efficiency", script("benchmarks/efficiency/finetune_speed.py", "--data-root", nt18, "--output", speed,
                                  *(["--models-from", arguments.models_from] if arguments.models_from else [])),
             speed, done=lambda: speed_complete(speed))]])
    timing = [finetune(model, nt18, f"{task}_revised", 0, results / "timing" / model / "nt18" / f"{task}_revised" / "fold-00.json", MICRO_BATCHES[model])
              for model in LOCUS for task in NT18_TASKS]
    timing += [baseline(name, f"{task}_revised", 0, results / "timing" / name / "nt18" / f"{task}_revised" / "fold-00.json")
               for name in FP32_BASELINES + BASELINES for task in NT18_TASKS]
    for job in timing:
        job.name = "timing." + job.name
    plans["timing"] = Plan("timing", ["nt18"], [timing], exclusive=True)
    # the pretraining-split ablation: one pretraining per GPU; then NT18 for the four models, fold by fold, with the recipe and
    # micro-batches of the released counterpart and the selected weights kept; then the 2M models' scoring, as for the twins
    # the split models: the released ones (downloaded by name), or local checkpoints: --models-from DIR, or the --runs of a
    # command that also runs split-pretrain
    local = split_source(arguments)
    split_pretrained = lambda name: local / name / "checkpoints/final.pt" if local else name
    plans["split-pretrain"] = Plan("split-pretrain", ["genome-windows", "bpe-no-chr20-21"], [[
        Job(f"pretrain.{name}", script("scripts/pretrain.py", "--config", ROOT / "configs/pretrain" / f"{name}.yaml", "--output-dir", arguments.runs / name,
                                       "--resume", "--set", f"data.genome_windows={data / 'GRCh38.p14.acgt.w4096.s4096'}"),
            arguments.runs / name / "checkpoints/final.pt") for name in SPLIT]], exclusive=True)

    def split_job(name, task, fold) -> Job:
        output = results / name / "nt18" / f"{task}_revised" / f"fold-{fold:02d}.json"
        return Job(f"{name}.nt18.{task}_revised.fold{fold}",
                   script("scripts/finetune.py", "--pretrained", split_pretrained(name), "--recipe",
                          ROOT / "configs/finetune" / f"{RECIPE[SPLIT[name]]}.yaml", "--data-root", nt18, "--task", f"{task}_revised", "--fold", fold,
                          "--output", output, "--micro-batches", MICRO_BATCHES[SPLIT[name]], "--save-best"), output)

    def split_score_job(name, task, fold, data_root, sets, study=None) -> Job:
        """Scores a saved split-ablation fold, next to its result or (with `study`) into results/<study>/<name>/nt18/<task>_revised."""
        result = results / name / "nt18" / f"{task}_revised" / f"fold-{fold:02d}.json"
        output_dir = results / study / name / "nt18" / f"{task}_revised" if study else None
        scored = (output_dir or result.parent) / f"fold-{fold:02d}.scored.json"
        complete = lambda: scored.is_file() and set(sets) <= set(json.loads(scored.read_text())["sets"])
        command = script("benchmarks/noisy_reads/score.py", result, "--sets", *sets, "--data-root", data_root)
        return Job(f"{name}.score{'_' + study if study else ''}.{task}.fold{fold}", command + (["--output-dir", str(output_dir)] if study else []),
                   scored, done=complete)

    plans["split-nt18"] = Plan("split-nt18", ["nt18"], [[split_job(name, task, fold) for fold in nt18_folds for name in SPLIT for task in NT18_TASKS]])
    plans["split-score"] = Plan("split-score", ["genome-fasta", "nt18", "error-sets", "control-sets", "pbsim-sets", "real-reads"], [
        [split_job(name, task, fold) for fold in nt18_folds for name in SPLIT_SCORED for task in NT18_TASKS],
        [split_score_job(name, task, fold, nt18, NOISY_SETS) for name in SPLIT_SCORED for task in NT18_TASKS for fold in nt18_folds]
        + [split_score_job(name, task, fold, data / "benchmarks/real_reads", REAL_SETS, "real_reads")
           for name in SPLIT_SCORED for task in REAL_TASKS for fold in nt18_folds]])
    return plans


def split_source(arguments):
    """Directory of the split models' checkpoints (DIR/<model>/checkpoints/final.pt), or None for the released models."""
    if arguments.models_from:
        return arguments.models_from
    return arguments.runs if "split-pretrain" in (getattr(arguments, "plans", None) or []) else None


def split_models_needed(plans: dict, names: list, only=None) -> list:
    """The pretrained split-ablation models that the selected jobs of the split plans start from (fine-tunes) or write (split-pretrain)."""
    jobs = [job.name for name in names if name in SPLIT_PLANS for stage in plans[name].stages for job in stage
            if not only or any(fnmatch.fnmatchcase(job.name, pattern) for pattern in only)]
    return [name for name in SPLIT if any(job.startswith(f"{name}.nt18.") or job == f"pretrain.{name}" for job in jobs)]


def check_split_checkpoints(runs: Path, models: list) -> None:
    """Each pretrained split-ablation model ran its full schedule with its own configuration (pretraining writes final.pt also
    when it is stopped early, and a final.pt of another configuration would be fine-tuned without notice)."""
    import torch

    from locus.config import load_config

    for name in models:
        path = runs / name / "checkpoints/final.pt"
        if not path.is_file():
            sys.exit(f"{path} is missing: run the split-pretrain plan first")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        wanted = load_config(ROOT / "configs/pretrain" / f"{name}.yaml").to_dict()
        found = checkpoint["config"]
        problems = []
        if checkpoint["step"] != wanted["training"]["max_steps"]:
            problems.append(f"it stopped at step {checkpoint['step']} of {wanted['training']['max_steps']}")
        if found["model"] != wanted["model"] or found["training"] != wanted["training"]:
            problems.append("its model or training settings differ from the configuration")
        if list(found["data"]["train_sources"]) != list(wanted["data"]["train_sources"]):
            problems.append("its training chromosomes differ from the configuration")
        if problems:
            sys.exit(f"{path}: " + "; ".join(problems) + ". Remove this final.pt and run the split-pretrain plan again (it continues "
                     "from the newest step checkpoint).")


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


def run_stage(jobs: list, gpu_slots: list, cpu_slots: int, dry_run: bool, logs: Path, verbose: bool = False,
              gpu_env: str = "CUDA_VISIBLE_DEVICES") -> list:
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
            env = dict(os.environ, **{gpu_env: slot})
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


def collect(results: Path, dry_run: bool, nt18_folds: int = NT18_FOLDS, other_folds: int = OTHER_FOLDS, runs: Path = None) -> None:
    command = [PYTHON, str(ROOT / "scripts/collect_results.py"), "--results", str(results), "--output", str(results / "tables"),
               "--nt18-folds", str(nt18_folds), "--other-folds", str(other_folds)] + (["--runs", str(runs)] if runs else [])
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plans", nargs="*", metavar="PLAN", help=f"{', '.join(PLANS + SPLIT_PLANS)}, all, or tables")
    parser.add_argument("--list", action="store_true", help="list the plans with their number of jobs and how many are done")
    parser.add_argument("--gpus", help="comma-separated GPU indices (default: every visible GPU)")
    parser.add_argument("--jobs-per-gpu", type=int, default=1, help="fine-tunes sharing one GPU (the paper: up to 3)")
    parser.add_argument("--nt18-folds", type=int, default=NT18_FOLDS, choices=range(1, 11), metavar="{1..10}",
                        help=f"folds of the NT18 plans and gue-splice (default {NT18_FOLDS}, the paper's; the split has 10)")
    parser.add_argument("--other-folds", type=int, default=OTHER_FOLDS, choices=range(1, 11), metavar="{1..10}",
                        help=f"folds of genomic-benchmarks and heldout (default {OTHER_FOLDS}, the paper's)")
    parser.add_argument("--gpu-env", default="CUDA_VISIBLE_DEVICES",
                        help="environment variable that gives each job its GPU (AMD GPUs with ROCm: HIP_VISIBLE_DEVICES)")
    parser.add_argument("--cpu-jobs", type=int, default=2, help="parallel CPU jobs (frozen-embedding fits, parsing tests)")
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
        for name in PLANS + SPLIT_PLANS:
            jobs = [job for stage in plans[name].stages for job in stage]
            print(f"{name:20s} {sum(job.is_done() for job in jobs):4d} / {len(jobs):4d} jobs done")
        return
    names = PLANS[1:] if "all" in arguments.plans else [n for n in arguments.plans if n != "tables"]
    unknown = [n for n in names if n not in plans]
    if unknown:
        parser.error(f"unknown plan(s) {', '.join(unknown)}; choose from {', '.join(PLANS + SPLIT_PLANS)}, all, tables")
    local = split_source(arguments)
    if local and not arguments.dry_run:  # split-pretrain writes the missing checkpoints of its models and skips the existing ones: those are checked
        written = split_models_needed(plans, ["split-pretrain"], arguments.only) if "split-pretrain" in names else []
        check_split_checkpoints(local, [name for name in split_models_needed(plans, names, arguments.only)
                                        if name not in written or (local / name / "checkpoints/final.pt").is_file()])
    if arguments.gpus is not None:
        gpus = [g for g in arguments.gpus.split(",") if g]
    else:
        import torch

        gpus = [str(i) for i in range(torch.cuda.device_count())]
    if not gpus and not arguments.dry_run and any(job.gpu for n in names for stage in plans[n].stages for job in stage):
        sys.exit("no GPU found; pass --gpus")

    logs = arguments.results / "logs"
    run_data(sorted({d for n in names for d in plans[n].data}, key=DATA_ORDER.index), arguments.data, arguments.dry_run, logs)
    seen, failed = set(), []
    for name in names:
        if name == "baselines-authors-bf16":  # its folds are read from the fp16 results, which an earlier plan may just have written
            plans[name] = build_plans(arguments)[name]
        per_gpu = 1 if plans[name].exclusive else arguments.jobs_per_gpu
        for stage in plans[name].stages:
            jobs = [job for job in stage if job.name not in seen]  # jobs shared by two plans run once
            seen.update(job.name for job in jobs)
            if arguments.only:
                jobs = [job for job in jobs if any(fnmatch.fnmatchcase(job.name, pattern) for pattern in arguments.only)]
            stage_failed = run_stage(jobs, [g for g in gpus for _ in range(per_gpu)], arguments.cpu_jobs, arguments.dry_run, logs, arguments.verbose,
                                     arguments.gpu_env)
            failed += stage_failed
            if stage_failed:
                print(f"{name}: {len(stage_failed)} job(s) failed; the plan's later stages are skipped", flush=True)
                break
    if not names or not set(names) <= {"pretrain", "split-pretrain"}:  # no names: the tables only
        collect(arguments.results, arguments.dry_run, arguments.nt18_folds, arguments.other_folds, arguments.runs)
    if failed:
        sys.exit(f"{len(failed)} job(s) failed: " + ", ".join(job.name for job in failed))


if __name__ == "__main__":
    main()
