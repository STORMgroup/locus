"""benchmarks/noisy_reads/real_reads.py: the read window cut from an alignment (no network, no pysam)."""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/noisy_reads"))

import real_reads  # noqa: E402

M, I, D, S = real_reads.M, real_reads.I, real_reads.D, real_reads.S


def alignment():
    """5 soft-clipped bases, then reference 100-110 matched, 2 inserted bases, 110-120 matched, 120-123 deleted, 123-133
    matched (read bases 'a' for clipped, 'x' for inserted, the matched bases copy the reference except one substitution)."""
    reference = "".join("ACGT"[i % 4] for i in range(200))
    matched = lambda start, end: reference[start:end]
    query = "aaaaa" + matched(100, 110) + "xx" + matched(110, 120) + matched(123, 133)
    query = query[:20] + ("A" if query[20] != "A" else "C") + query[21:]  # a substitution at reference position 113
    read = SimpleNamespace(cigartuples=[(S, 5), (M, 10), (I, 2), (M, 10), (D, 3), (M, 10)], reference_start=100, query_sequence=query)
    return read, reference


def cut(start, end):
    read, reference = alignment()
    return real_reads.cut_window(read, real_reads.cigar_index(read), start, end, reference[start:end]), read.query_sequence


def test_window_keeps_inner_insertion_and_counts_edits():
    (sequence, substitutions, inserted, deleted, longest), query = cut(105, 125)
    assert sequence == query[10:29]  # from the base aligned to 105 to the base aligned to 124
    assert (substitutions, inserted, deleted, longest) == (1, 2, 3, 3)


def test_insertion_at_the_window_edge_is_not_kept():
    (sequence, _, inserted, _, _), query = cut(110, 115)
    assert sequence == query[17:22] and inserted == 0


def test_window_inside_a_deletion_is_empty():
    (sequence, substitutions, inserted, deleted, longest), _ = cut(121, 123)
    assert sequence == "" and deleted == 2 and longest == 3
