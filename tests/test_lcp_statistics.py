"""benchmarks/parsing/lcp_statistics.py and benchmarks/embedding/stability.py: the parse they report on and the edit
bookkeeping they rely on."""
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/parsing"))
sys.path.insert(0, str(ROOT / "benchmarks/embedding"))

LIBRARY = ROOT / "build/lcptools/lib/liblcptools.so"
pytestmark = pytest.mark.skipif(not LIBRARY.is_file(), reason="lcptools not built (make lcptools)")

EXAMPLE = "ACCGAAATCTGTGCAGAGGACAAC"  # 24 bp of chromosome 1 with all four level-1 core types (the paper's LCP figure)


def test_example_parse():
    import lcp_statistics
    from locus.segmentation import LCPSegmenter

    library = LCPSegmenter(level=2).library
    level1 = lcp_statistics.cores(library, EXAMPLE, 1)
    assert {lcp_statistics.TYPES[kind] for _, _, kind in level1} == set(lcp_statistics.TYPES.values())
    assert [(s, e) for s, e, _ in lcp_statistics.cores(library, EXAMPLE, 2)] == LCPSegmenter(level=2).bounds(EXAMPLE)
    assert LCPSegmenter(level=2).bounds(EXAMPLE) == [(0, 10), (3, 14), (9, 18), (15, 24)]
    # each level-2 segment starts where the level-1 core left of its first core starts (coin tossing extends a core over its
    # left neighbour), so every segment start is a level-1 core start
    assert {s for s, _ in LCPSegmenter(level=2).bounds(EXAMPLE)} <= {s for s, _, _ in level1}


def test_segmentation_shares():
    import lcp_statistics
    from locus.segmentation import LCPSegmenter

    rng = random.Random(0)
    windows = ["".join(rng.choice("ACGT") for _ in range(4096)) for _ in range(3)]
    stats = lcp_statistics.segmentation(windows, LCPSegmenter(level=2).library)
    shares = stats["level 2"]["share_of_bases_covered_by"]
    assert sum(shares.values()) == pytest.approx(1.0)
    assert 0.7 < stats["level 2"]["share_of_bases_in_2_or_more"] < 0.95
    assert stats["level 1"]["share_of_non_run_cores_with_3_bases"] == 1.0


def test_stability_edits():
    """The reverse-strand position and shift of each edit match the reverse complement of the edited sequence, and every
    surviving clean base maps to the same base in the edited sequence."""
    import stability

    rng = random.Random(1)
    sequence = "".join(rng.choice("ACGT") for _ in range(stability.LENGTH))
    for kind in stability.EDITS:
        edited = stability.edit(sequence, kind, rng)
        (forward, shift), (reverse, reverse_shift) = stability.strand_edits(kind)
        assert shift == reverse_shift == len(edited) - len(sequence)
        a, b = stability.reverse_complement(sequence), stability.reverse_complement(edited)
        n = abs(shift)
        if kind == "substitution":
            assert a[:reverse] == b[:reverse] and a[reverse + 1:] == b[reverse + 1:] and a[reverse] != b[reverse]
        elif kind.startswith("insertion"):
            assert b[:reverse] == a[:reverse] and b[reverse + n:] == a[reverse:]
        else:
            assert b[:reverse] == a[:reverse] and b[reverse:] == a[reverse + n:]
        for x, mapped in enumerate(stability.follow(kind)):
            if mapped is not None and not (kind == "substitution" and x == stability.SITE):
                assert sequence[x] == edited[mapped[0]]
