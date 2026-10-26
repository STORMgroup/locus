"""Segmenters: LCP (locally consistent parsing, through lcptools), fixed-length windows, and BPE.

A segmentation is a list of half-open intervals [start, end) over the sequence. LCP and fixed segments may overlap;
BPE segments tile the sequence.
"""
import ctypes
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LCP_LIBRARY = PROJECT_ROOT / "build/lcptools/lib/liblcptools.so"


@dataclass(frozen=True)
class Segment:
    dna: str
    start: int
    end: int  # exclusive

    def is_valid(self, sequence: str) -> bool:
        return self.dna == sequence[self.start:self.end] and 0 <= self.start < self.end <= len(sequence)


class _Core(ctypes.Structure):
    # Must match struct core in third_party/lcptools/core.h (including the type field of local-changes.patch).
    _fields_ = [
        ("start", ctypes.c_uint32),
        ("end", ctypes.c_uint32),
        ("bit_rep", ctypes.POINTER(ctypes.c_uint)),
        ("label", ctypes.c_uint32),
        ("bit_size", ctypes.c_uint32),
        ("type", ctypes.c_int),
    ]


class _LPS(ctypes.Structure):
    _fields_ = [
        ("level", ctypes.c_int),
        ("size", ctypes.c_int),
        ("cores", ctypes.POINTER(_Core)),
    ]


class LCPSegmenter:
    """Level-`level` LCP cores of a sequence, computed by liblcptools.

    Runs of N are not parsed: every N becomes its own one-base segment and each A/C/G/T run between them is parsed on
    its own. Runs shorter than 3 bases are one segment. A run without any level-`level` core falls back to one
    segment per base."""

    def __init__(self, library_path: Path = DEFAULT_LCP_LIBRARY, level: int = 2):
        library_path = Path(library_path)
        if not library_path.is_file():
            raise FileNotFoundError(f"lcptools library does not exist: {library_path} (run `make lcptools`)")
        if level < 1:
            raise ValueError("level must be >= 1")
        self.level = level
        self.library = ctypes.CDLL(str(library_path.resolve()))
        self.library.LCP_INIT.argtypes = []
        self.library.LCP_INIT.restype = None
        self.library.init_lps.argtypes = [ctypes.POINTER(_LPS), ctypes.c_char_p, ctypes.c_int]
        self.library.init_lps.restype = None
        self.library.lps_deepen.argtypes = [ctypes.POINTER(_LPS), ctypes.c_int]
        self.library.lps_deepen.restype = ctypes.c_int
        self.library.free_lps.argtypes = [ctypes.POINTER(_LPS)]
        self.library.free_lps.restype = None
        self.library.LCP_INIT()

    def _acgt_bounds(self, sequence: str) -> list[tuple[int, int]]:
        encoded = sequence.encode("ascii")
        value = _LPS()
        self.library.init_lps(ctypes.byref(value), encoded, len(encoded))
        try:
            self.library.lps_deepen(ctypes.byref(value), self.level)
            if value.level != self.level:
                raise RuntimeError(f"lcptools could not construct level-{self.level} cores")
            bounds = [(int(value.cores[i].start), int(value.cores[i].end)) for i in range(value.size)]
        finally:
            self.library.free_lps(ctypes.byref(value))
        if not bounds:
            return [(position, position + 1) for position in range(len(sequence))]
        for start, end in bounds:
            if not 0 <= start < end <= len(sequence):
                raise ValueError(f"Invalid segment [{start}, {end}) for length {len(sequence)}")
        return bounds

    def bounds(self, sequence: str) -> list[tuple[int, int]]:
        if not sequence or set(sequence).difference("ACGTN"):
            raise ValueError("LCP requires a non-empty A/C/G/T/N sequence")
        if len(sequence) < 3:
            return [(0, len(sequence))]
        if "N" not in sequence:
            return self._acgt_bounds(sequence)
        bounds = []
        position = 0
        while position < len(sequence):
            if sequence[position] == "N":
                bounds.append((position, position + 1))
                position += 1
                continue
            run_end = sequence.find("N", position)
            if run_end < 0:
                run_end = len(sequence)
            run = sequence[position:run_end]
            run_bounds = [(0, len(run))] if len(run) < 3 else self._acgt_bounds(run)
            bounds.extend((position + start, position + end) for start, end in run_bounds)
            position = run_end
        return bounds


def fixed_bounds(sequence: str, length: int, stride: int) -> list[tuple[int, int]]:
    """Windows of `length` bases starting every `stride` bases; the last window ends at the sequence end."""
    if length <= 0 or stride <= 0:
        raise ValueError("length and stride must be positive")
    bounds = []
    start = 0
    while start < len(sequence):
        bounds.append((start, min(start + length, len(sequence))))
        if start + length >= len(sequence):
            break
        start += stride
    return bounds


class BPESegmenter:
    """Token boundaries of a Hugging Face `tokenizers` BPE model; tokens tile the sequence."""

    def __init__(self, tokenizer_path: Path):
        from tokenizers import Tokenizer

        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"BPE tokenizer does not exist: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def bounds(self, sequence: str) -> list[tuple[int, int]]:
        bounds = [(start, end) for start, end in self.tokenizer.encode(sequence).offsets if end > start]
        if not bounds or bounds[0][0] != 0 or bounds[-1][1] != len(sequence):
            raise ValueError("BPE offsets do not cover the complete sequence")
        for (_, left_end), (right_start, _) in zip(bounds, bounds[1:]):
            if left_end != right_start:
                raise ValueError("BPE offsets contain a gap or overlap")
        return bounds


class Segmenter:
    """The segmenter named by a data config ("lcp", "fixed" or "bpe")."""

    def __init__(self, data_config):
        self.name = data_config.segmenter
        if self.name == "lcp":
            self._segmenter = LCPSegmenter(data_config.lcp_library, level=data_config.lcp_level)
        elif self.name == "bpe":
            self._segmenter = BPESegmenter(data_config.bpe_tokenizer)
        elif self.name == "fixed":
            self._length, self._stride = data_config.fixed_length, data_config.fixed_stride
        else:
            raise ValueError(f"Unknown segmenter: {self.name}")

    def bounds(self, sequence: str) -> list[tuple[int, int]]:
        if self.name == "fixed":
            return fixed_bounds(sequence, self._length, self._stride)
        return self._segmenter.bounds(sequence)

    def segment(self, sequence: str) -> list[Segment]:
        segments = [Segment(sequence[start:end], start, end) for start, end in self.bounds(sequence)]
        if not segments:
            raise ValueError("The segmenter produced no segments")
        if not all(item.is_valid(sequence) for item in segments):
            raise ValueError("The segmenter produced an invalid segment")
        return segments
