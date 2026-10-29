"""The GUE splice test sets without the test rows that share their splice site with an NT18 splice record.

    python benchmarks/gue/independent_sets.py
    python benchmarks/gue/independent_sets.py --data-root data/benchmarks/gue --out data/benchmarks/gue_independent

benchmarks/gue/prepare.py lists, in gue_splice/metadata.json, the test rows whose 32 bases around the junction occur in
a record of an NT18 splice task (test_rows_sharing_nt18_splice_site). For the clean test set (gue_splice) and every
modified test set (gue_splice_<set>, written by benchmarks/noisy_reads; one read per test row, in row order) this script
writes the same file without those rows to <out>, under the same directory name, so that the noisy-set scores can also be
reported on test sites independent of NT18. Score them into their own directory, so that the scores on the full test sets
are kept: benchmarks/noisy_reads/score.py <fold result> --data-root <out> --sets clean ont1 ...
--output-dir results/gue_independent/<twin>/gue/gue_splice (scripts/reproduce.py plan gue-splice)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from locus.finetune.data import load_records, write_records  # noqa: E402

TASK = "gue_splice"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/benchmarks/gue")
    parser.add_argument("--out", type=Path, default=ROOT / "data/benchmarks/gue_independent")
    arguments = parser.parse_args()
    metadata = json.loads((arguments.data_root / TASK / "metadata.json").read_text())
    shared, count = set(metadata["test_rows_sharing_nt18_splice_site"]), metadata["splits"]["test"]["count"]
    directories = sorted(d for d in arguments.data_root.iterdir() if d.is_dir() and (d.name == TASK or d.name.startswith(f"{TASK}_")))
    for directory in directories:
        records = load_records(directory / "test.jsonl.gz")
        if len(records) != count:
            raise ValueError(f"{directory} has {len(records)} test rows, the clean test set {count}")
        write_records(arguments.out / directory.name / "test.jsonl.gz", [r for row, r in enumerate(records) if row not in shared])
    (arguments.out / "metadata.json").write_text(json.dumps({"source": str(arguments.data_root), "rows_removed": sorted(shared),
                                                             "rows_kept": count - len(shared), "sets": [d.name for d in directories]},
                                                            indent=1) + "\n")
    print(f"{len(directories)} sets, {count - len(shared)} of {count} test rows each, in {arguments.out}")


if __name__ == "__main__":
    main()
