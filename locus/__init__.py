"""LoCUS: DNA language models on locally consistent parsing (LCP) segments.

    import locus

    model = locus.load("locus-2m")                    # downloaded from the Hugging Face Hub on first use
    vectors = model.embed(["ACGT" * 100])             # [sequences, width]: mean over the bases, both strands averaged
    per_base = model.embed(["ACGT" * 100], per_base=True)   # list of [length, width] arrays
"""
from locus.inference import Pretrained, load

__all__ = ["Pretrained", "load"]
