"""Segmenters: fixed windows, LCP cores (reference boundaries of the library build used in the paper) and N handling."""
from locus.segmentation import LCPSegmenter, fixed_bounds

SEQUENCE = ("TTAGTTGTGCCGCAGCGAAGTAGTGCTTGAAATATGCGACCCCTAAGTAGGAGCGTATGCGCCCAGTAACCAATGCCTGTTGAGATGCCAGACGCGTAACCAAAACATAGAA"
            "ACCATCAA")
LEVEL2 = [(3, 13), (8, 17), (12, 23), (16, 27), (22, 33), (28, 40), (37, 47), (43, 53), (48, 58), (52, 65), (58, 68), (65, 74), (68, 78),
          (74, 84), (81, 91), (88, 98), (95, 106), (100, 114), (107, 117)]


def test_fixed_bounds():
    assert fixed_bounds("A" * 23, 10, 5) == [(0, 10), (5, 15), (10, 20), (15, 23)]
    assert fixed_bounds("A" * 20, 10, 5) == [(0, 10), (5, 15), (10, 20)]
    assert fixed_bounds("A" * 4, 10, 5) == [(0, 4)]


def test_lcp_level2_reference():
    assert LCPSegmenter(level=2).bounds(SEQUENCE) == LEVEL2


def test_lcp_n_runs_are_single_segments():
    bounds = LCPSegmenter(level=2).bounds(SEQUENCE[:40] + "NN" + SEQUENCE[40:80])
    assert (40, 41) in bounds and (41, 42) in bounds
    assert bounds[:6] == LEVEL2[:6]


def test_lcp_is_content_defined():
    # an edit changes only the cores near it: cores far from the edit keep their boundaries
    segmenter = LCPSegmenter(level=2)
    edited = SEQUENCE[:60] + ("A" if SEQUENCE[60] != "A" else "C") + SEQUENCE[61:]
    before, after = segmenter.bounds(SEQUENCE), segmenter.bounds(edited)
    assert [b for b in before if b[1] < 30] == [b for b in after if b[1] < 30]
    assert [b for b in before if b[0] > 90] == [b for b in after if b[0] > 90]
