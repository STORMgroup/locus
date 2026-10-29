"""NT18 test sets sequenced by a published nanopore read simulator: PBSIM3 (Ono et al., 2022).

PBSIM3 runs in template mode (--strategy templ): every clean test sequence is read exactly once, full length, on the
forward strand. Errors follow PBSIM3's error model trained on real ONT reads (ERRHMM-ONT-HQ, recommended for read
accuracy of at least 90%); the accuracy of each read is drawn around the requested mean ACCURACY (in percent). Labels
are unchanged, and only the test set is simulated.

    make pbsim3
    python benchmarks/noisy_reads/simulate_pbsim.py --accuracies 97 99

--pbsim-dir is a built checkout of github.com/yukiteruono/pbsim3 (default build/pbsim3, built by make pbsim3 at
commit 35ce0de, the one of the reported sets); the executable is src/pbsim and the error model
data/ERRHMM-ONT-HQ.model. The script reads
<data-root>/<task>_revised/test.jsonl.gz and writes <output-root>/<task>_pbsim<ACCURACY>_revised/test.jsonl.gz and
metadata.json (PBSIM3 commit, command, seed, realized read accuracy), and the further error draws
<task>_pbsim<ACCURACY>-d<k>_revised, k = 1 .. --draws - 1 (noisy_sets.py; for gue_splice the same names without the
_revised suffix). The seed of a set is
42 + 1000 * ACCURACY + the index of the task in the alphabetical order of the 18 NT18 tasks (18 for gue_splice), plus
1,000,000 * k for draw k.
"""
import argparse
import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from noisy_sets import DEFAULT_DATA_ROOT, DRAWS, OTHER_TASKS, ROOT, clean_task, draw_tag, variant_task

sys.path.insert(0, str(ROOT))

from locus.finetune.data import NT18_TASKS, SequenceRecord, load_records, sha256_file, write_records  # noqa: E402

REPORTED_COMMIT = "35ce0debb8b9df8a45ac9bb85420686776e9e721"
ERROR_MODEL = "data/ERRHMM-ONT-HQ.model"


def pbsim_seed(task: str, accuracy_percent: int, draw: int = 0) -> int:
    index = sorted(NT18_TASKS).index(task) if task in NT18_TASKS else len(NT18_TASKS) + OTHER_TASKS.index(task)
    return 42 + 1000 * accuracy_percent + index + 1_000_000 * draw


def pbsim_commit(pbsim_dir: Path) -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(pbsim_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def simulate(sequences: list, accuracy: float, seed: int, pbsim_dir: Path, error_model: Path, workdir: Path) -> tuple:
    """(one read per template in template order, the PBSIM3 command, PBSIM3's report of the realized read accuracy)."""
    templates = workdir / "templates.fa"
    templates.write_text("".join(f">t{index}\n{sequence}\n" for index, sequence in enumerate(sequences)))
    command = [str(pbsim_dir / "src/pbsim"), "--strategy", "templ", "--method", "errhmm", "--errhmm", str(error_model),
               "--template", str(templates), "--accuracy-mean", f"{accuracy:.2f}", "--seed", str(seed), "--prefix", str(workdir / "sim")]
    log = subprocess.run(command, capture_output=True, text=True, check=True).stderr
    reads = []
    with gzip.open(workdir / "sim.fq.gz", "rt") as handle:
        for line_number, line in enumerate(handle):
            if line_number % 4 == 1:
                reads.append(line.strip())
    sources = []
    with gzip.open(workdir / "sim.maf.gz", "rt") as handle:
        for line in handle:
            if line.startswith("s t"):
                sources.append(int(line.split()[1][1:]))
    if len(reads) != len(sequences) or sources != list(range(len(sequences))):
        raise RuntimeError(f"PBSIM3 output does not match the templates: {len(reads)} reads for {len(sequences)} templates")
    realized = [line for line in log.splitlines() if line.startswith("read accuracy mean")]
    return reads, command, realized[0] if realized else None


def make_set(data_root: Path, output_root: Path, task: str, accuracy_percent: int, pbsim_dir: Path, error_model: Path,
             commit: Optional[str], work_root: Optional[Path], draw: int = 0) -> dict:
    tag = draw_tag(f"pbsim{accuracy_percent}", draw)
    target = variant_task(task, tag)
    records = load_records(data_root / clean_task(task) / "test.jsonl.gz")
    seed = pbsim_seed(task, accuracy_percent, draw)
    with tempfile.TemporaryDirectory(dir=work_root) as workdir:
        reads, command, realized = simulate([record.sequence for record in records], accuracy_percent / 100, seed, pbsim_dir,
                                            error_model, Path(workdir).resolve())
    invalid = sum(1 for read in reads if not read or set(read) - set("ACGT"))
    if invalid:
        raise RuntimeError(f"{target}: {invalid} reads are empty or contain bases other than A, C, G, T")
    simulated = [SequenceRecord(name=f"{record.name}:{tag}", sequence=read, label=record.label, task=target,
                                chromosome=record.chromosome) for record, read in zip(records, reads)]
    path = output_root / target / "test.jsonl.gz"
    write_records(path, simulated)
    length_ratio = sum(map(len, reads)) / sum(len(record.sequence) for record in records)
    metadata = {"source_task": clean_task(task), "split": "test", "simulator": "PBSIM3", "pbsim_commit": commit,
                "error_model": error_model.name, "accuracy_mean_requested": accuracy_percent / 100, "realized": realized, "seed": seed,
                "command": " ".join(Path(part).name if part.startswith("/") else part for part in command), "count": len(simulated),
                "length_ratio_read_vs_template": round(length_ratio, 5), "sha256": sha256_file(path), **({"draw": draw} if draw else {})}
    (output_root / target / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{target}: {len(simulated)} reads, {realized}, read/template length {length_ratio:.4f}", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pbsim-dir", type=Path, default=ROOT / "build/pbsim3", help="built PBSIM3 checkout (default: make pbsim3)")
    parser.add_argument("--accuracies", type=int, nargs="+", default=[97, 99], help="mean read accuracies in percent")
    parser.add_argument("--draws", type=int, default=DRAWS, help=f"independent error draws per set (default {DRAWS})")
    parser.add_argument("--tasks", nargs="+", choices=NT18_TASKS + list(OTHER_TASKS), default=NT18_TASKS, metavar="TASK",
                        help="NT18 tasks without the _revised suffix, or gue_splice (default: the 18 NT18 tasks)")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="directory of the clean task directories")
    parser.add_argument("--output-root", type=Path, help="directory for the new test sets (default: --data-root)")
    parser.add_argument("--work-dir", type=Path, help="parent directory of the temporary PBSIM3 files (default: system temp)")
    arguments = parser.parse_args()
    pbsim_dir = arguments.pbsim_dir.resolve()
    error_model = pbsim_dir / ERROR_MODEL
    for path in (pbsim_dir / "src/pbsim", error_model):
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist; build PBSIM3 first (make pbsim3)")
    commit = pbsim_commit(pbsim_dir)
    if commit != REPORTED_COMMIT:
        print(f"warning: PBSIM3 commit {commit} differs from the commit of the reported sets ({REPORTED_COMMIT})", file=sys.stderr)
    for task in arguments.tasks:
        for accuracy_percent in arguments.accuracies:
            for draw in range(arguments.draws):
                make_set(arguments.data_root, arguments.output_root or arguments.data_root, task, accuracy_percent, pbsim_dir, error_model,
                         commit, arguments.work_dir, draw)


if __name__ == "__main__":
    main()
