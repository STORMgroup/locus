"""Nucleotide vocabulary shared by pretraining and fine-tuning."""

BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
PAD_ID = 5
# Input symbol of a base hidden by masked-segment prediction.
MASK_ID = 6
# Input symbol of a base covered by two masked segments that drew different corruptions (overlapping LCP cores);
# it shows neither corruption, so the true base cannot leak through the other segment.
OVERLAP_MASK_ID = 7
ALPHABET_SIZE = 8
# Prediction classes of the pretraining head: A, C, G, T.
OUTPUT_CLASSES = 4

_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement(sequence: str) -> str:
    """Reverse complement in upper case; symbols other than A/C/G/T (such as N) are kept."""
    return sequence.translate(_COMPLEMENT)[::-1].upper()
