"""Definitions shared by the noisy-read test sets: the nanopore-like error model and the names of the test-set
variants.

Error model: every base of the clean sequence starts an error event with probability equal to the error rate. An event
is a substitution by one of the three other bases (40%), a deletion (35%) or an insertion before the base (25%).
Deletions and insertions are 1, 2 or 3 bases long with probability 0.75, 0.1875 and 0.0625. An inserted stretch
repeats the current base with probability 0.5 (homopolymer-like errors) and consists of uniformly drawn bases
otherwise.

A variant of the NT18 task <task>_revised is stored as <task>_<tag>_revised, for example H3K27ac_ont3_revised
(errors at 3% per base) or H3K27ac_pbsim97_revised (PBSIM3 reads with mean accuracy 97%). Only the test split is
modified; models are fine-tuned on the clean training data.

Every noisy set is drawn DRAWS times with independent random seeds, because one draw of errors on a small test set
(promoter_tata has 212 sequences) can by chance hit or spare the bases that carry the label. Draw 0 is the set itself
(H3K27ac_ont3_revised), draw k > 0 the set H3K27ac_ont3-d<k>_revised; the score of a fold on a set is the mean over its
draws (scripts/collect_results.py).
"""
from pathlib import Path

EVENT_MIX = (("substitution", 0.40), ("deletion", 0.35), ("insertion", 0.25))
INDEL_LENGTHS = ((1, 0.75), (2, 0.1875), (3, 0.0625))
HOMOPOLYMER_PROBABILITY = 0.5
DRAWS = 10

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = ROOT / "data/benchmarks/nt18"


def pick(x: float, table: tuple):
    """The value of `table` ((value, probability), ...) whose cumulative-probability interval contains x in [0, 1)."""
    cumulative = 0.0
    for value, probability in table:
        cumulative += probability
        if x < cumulative:
            return value
    return table[-1][0]


def rate_tag(rate_percent: float) -> str:
    """Short form of an error rate in percent: 1.0 -> "1", 0.5 -> "0.5"."""
    return f"{rate_percent:g}"


def draw_tag(tag: str, draw: int) -> str:
    """Name of error draw `draw` of the set `tag`: draw 0 is the set itself ("ont3"), draw k > 0 is "ont3-d<k>"."""
    return tag if draw == 0 else f"{tag}-d{draw}"


def clean_task(task: str) -> str:
    return f"{task}_revised"


def variant_task(task: str, tag: str) -> str:
    return f"{task}_{tag}_revised"
