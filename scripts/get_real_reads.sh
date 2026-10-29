#!/usr/bin/env bash
# Real HG002 nanopore reads of the NT18 test windows (the real-read test sets of the tokenizer study), and a check that
# they are identical to the paper's.
#
#     bash scripts/get_real_reads.sh                 # writes data/benchmarks/real_reads
#     bash scripts/get_real_reads.sh OUT_DIR
#
# The reads are streamed from the public ONT open-data CRAM files (giab_2023.05, flowcell PAO89685, sup and hac basecalls),
# so the script needs network access; only the cut-out read windows are kept (about 170 MB), which takes about 25 min.
# Missing inputs are prepared first: the GRCh38.p14 FASTA (scripts/prepare_genome.sh --download-only) and the NT18 tasks
# (benchmarks/nt18/prepare.py). The environment is the one of install.sh (.venv; pysam is pinned in requirements.txt).
#
# Output (benchmarks/noisy_reads/real_reads.py): the sets <task>_realref_revised, <task>_realsup[-dK]_revised and
# <task>_realhac[-dK]_revised, summary.json (read divergence and the windows kept) and checksums.tsv. The sets are written
# deterministically and compared with benchmarks/noisy_reads/real_reads_sha256.tsv, the checksums of the sets the paper's
# real-read results were scored on.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${1:-data/benchmarks/real_reads}"
PYTHON="${PYTHON:-.venv/bin/python}"

if ! "$PYTHON" -c "import pysam" 2>/dev/null; then
    echo "pysam is missing in $PYTHON: run bash install.sh first (pysam is pinned in requirements.txt)" >&2
    exit 1
fi
if [[ ! -f data/GRCh38.p14.fasta.gz ]]; then
    bash scripts/prepare_genome.sh --download-only
fi
if [[ ! -f data/benchmarks/nt18/splice_sites_all_revised/test.jsonl.gz ]]; then
    "$PYTHON" benchmarks/nt18/prepare.py
fi

"$PYTHON" benchmarks/noisy_reads/real_reads.py --nt18 data/benchmarks/nt18 --genome data/GRCh38.p14.fasta.gz --out "$OUT"

if (cd "$OUT" && sha256sum -c --quiet "$ROOT/benchmarks/noisy_reads/real_reads_sha256.tsv"); then
    echo "real reads identical to the paper's ($(wc -l < benchmarks/noisy_reads/real_reads_sha256.tsv) files): $OUT"
else
    echo "the extracted reads differ from the paper's (benchmarks/noisy_reads/real_reads_sha256.tsv); see above" >&2
    exit 1
fi
