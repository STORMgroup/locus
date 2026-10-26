"""Noisy-read test sets: draw 0 is the reported set (same seed, name and reads as a single-draw run), further error draws are
independent, and the tables use the mean over a set's draws."""
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/noisy_reads"))
sys.path.insert(0, str(ROOT / "scripts"))

import collect_results  # noqa: E402
import make_controls  # noqa: E402
import make_errors  # noqa: E402
from locus.finetune.data import SequenceRecord, load_records, write_records  # noqa: E402

TASK = "promoter_tata"


def clean_set(root: Path) -> list:
    rng = random.Random(0)
    records = [SequenceRecord(name=f"r{i}", sequence="".join(rng.choice("ACGT") for _ in range(300)), label=i % 2, task=f"{TASK}_revised",
                              chromosome="chr1") for i in range(20)]
    write_records(root / f"{TASK}_revised" / "test.jsonl.gz", records)
    return records


def test_error_draws(tmp_path):
    records = clean_set(tmp_path)
    for draw in range(3):
        make_errors.make_set(tmp_path, tmp_path, TASK, 3.0, draw)
    first = json.loads((tmp_path / f"{TASK}_ont3_revised" / "metadata.json").read_text())
    assert first["seed"] == f"42-{TASK}-test-ont3" and "draw" not in first
    rng = random.Random(f"42-{TASK}-test-ont3")  # the single-draw procedure
    expected = [make_errors.add_errors(record.sequence, 0.03, rng)[0] for record in records]
    reads = {draw: [r.sequence for r in load_records(tmp_path / f"{TASK}_{tag}_revised" / "test.jsonl.gz")]
             for draw, tag in enumerate(["ont3", "ont3-d1", "ont3-d2"])}
    assert reads[0] == expected
    assert reads[0] != reads[1] != reads[2] != reads[0]
    assert json.loads((tmp_path / f"{TASK}_ont3-d2_revised" / "metadata.json").read_text())["seed"] == f"42-{TASK}-test-ont3-draw2"


def test_control_draws(tmp_path):
    clean_set(tmp_path)
    for draw in range(2):
        make_controls.make_controls(tmp_path, tmp_path, TASK, list(make_controls.CONTROLS), [1.0, 3.0], draw)
    for tag in ("sub1", "sub3", "ontw1", "ontw3"):
        zero = json.loads((tmp_path / f"{TASK}_{tag}_revised" / "metadata.json").read_text())
        one = json.loads((tmp_path / f"{TASK}_{tag}-d1_revised" / "metadata.json").read_text())
        assert zero["seed"] == f"42-{TASK}-test-controls" and "draw" not in zero and one["draw"] == 1
        assert ([r.sequence for r in load_records(tmp_path / f"{TASK}_{tag}_revised" / "test.jsonl.gz")]
                != [r.sequence for r in load_records(tmp_path / f"{TASK}_{tag}-d1_revised" / "test.jsonl.gz")])


def test_tables_use_the_mean_over_draws(tmp_path):
    entry = {"validation_loss": 0.5, "validation_metrics": {"mcc": 0.6}, "test_metrics": {"mcc": 0.61}}
    (tmp_path / "fold-00.json").write_text(json.dumps({"history": [entry], "best_step": 10, "test_metrics": {"mcc": 0.61}}))
    scores = {"ont1": 0.3, "ont3": 0.11, "ont3-d1": 0.2, "ont3-d2": 0.35, "pbsim97": 0.4}
    (tmp_path / "fold-00.scored.json").write_text(json.dumps({"best_step": 10, "sets": {k: {"mcc": v} for k, v in scores.items()}}))
    row = collect_results.study_folds(tmp_path, 1)[0]
    assert row["clean"] == 0.61 and row["ont1"] == 0.3 and row["pbsim97"] == 0.4  # one draw: the value itself
    assert row["ont3"] == statistics.mean([0.11, 0.2, 0.35]) and row["ont3 draws"] == 3 and row["ont1 draws"] == 1
    assert not any("-d" in name for name in row)
