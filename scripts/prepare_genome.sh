#!/usr/bin/env bash
# Download the GRCh38.p14 reference genome and split it into the genome windows used for pretraining.
#
#     make tools
#     bash scripts/prepare_genome.sh             # writes into data/
#     bash scripts/prepare_genome.sh DATA_DIR
#     bash scripts/prepare_genome.sh --download-only [DATA_DIR]   # only the genome FASTA (enough for the benchmarks)
#
# The genome is the NCBI RefSeq assembly GCF_000001405.40 (GRCh38.p14). It is checked against its MD5 checksum and kept
# as DATA_DIR/GRCh38.p14.fasta.gz. build/split_fasta then cuts every sequence into consecutive, non-overlapping windows
# that contain only A, C, G and T (an N or any other symbol ends the current window; incomplete windows are dropped):
#
#     DATA_DIR/GRCh38.p14.acgt.w4096.s4096      4,096 bp windows (LoCUS-0.5M, LoCUS-2M, twin-fixed-2m, twin-bpe-2m)
#
# The directory holds gzipped FASTA shards, manifests/reads.tsv and manifests/shards.tsv, dataset.json and reports of
# the rejected bases and discarded window tails. A directory that is already complete (it has a _SUCCESS marker) is left
# unchanged.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DOWNLOAD_ONLY=0
if [[ "${1:-}" == "--download-only" ]]; then
    DOWNLOAD_ONLY=1
    shift
fi
DATA_DIR="${1:-data}"
GENOME="$DATA_DIR/GRCh38.p14.fasta.gz"
GENOME_URL="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz"
GENOME_MD5="c30471567037b2b2389d43c908c653e1"
SPLIT_FASTA="$ROOT/build/split_fasta"

if [[ $DOWNLOAD_ONLY == 0 && ! -x "$SPLIT_FASTA" ]]; then
    echo "$SPLIT_FASTA is missing; build it with: make tools" >&2
    exit 1
fi
mkdir -p "$DATA_DIR"

check_md5() {
    if ! echo "$GENOME_MD5  $1" | md5sum --check --status; then
        echo "MD5 checksum of $1 is not $GENOME_MD5" >&2
        return 1
    fi
}

if [[ -f "$GENOME" ]]; then
    check_md5 "$GENOME"
else
    echo "Downloading $GENOME_URL"
    curl --fail --location --show-error --retry 5 --retry-delay 10 --output "$GENOME.part" "$GENOME_URL"
    check_md5 "$GENOME.part"
    mv "$GENOME.part" "$GENOME"
fi
echo "Genome: $GENOME (MD5 $GENOME_MD5)"
if [[ $DOWNLOAD_ONLY == 1 ]]; then
    exit 0
fi

for WINDOW in 4096; do
    OUTPUT="$DATA_DIR/GRCh38.p14.acgt.w${WINDOW}.s${WINDOW}"
    if [[ -f "$OUTPUT/_SUCCESS" ]]; then
        echo "$OUTPUT is complete"
        continue
    fi
    if [[ -e "$OUTPUT" ]]; then
        echo "$OUTPUT exists but is incomplete; remove it and run again" >&2
        exit 1
    fi
    STAGING="$OUTPUT.partial"
    rm -rf "$STAGING"
    echo "Splitting into ${WINDOW} bp windows: $OUTPUT"
    "$SPLIT_FASTA" "$GENOME" "$STAGING" GRCh38p14 "$WINDOW"
    for MANIFEST in reads shards; do
        gzip -cd "$STAGING/manifests/$MANIFEST.tsv.gz" > "$STAGING/manifests/$MANIFEST.tsv"
        rm "$STAGING/manifests/$MANIFEST.tsv.gz"
    done
    touch "$STAGING/_SUCCESS"
    mv "$STAGING" "$OUTPUT"
done
