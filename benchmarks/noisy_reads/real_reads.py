"""Real nanopore reads over the test windows of the 18 NT18 tasks (all human; every test record names its window, and the
test windows lie on chromosomes 20 and 21 of GRCh38).

The test window chrom:[start, end) named in each record is cut out of real HG002 Oxford Nanopore R10.4.1 reads (ONT open
data, giab_2023.05, flowcell PAO89685: minimap2 alignments to the GRCh38 no-alt analysis set, the "pass" reads with mean
quality >= 10), streamed from the public CRAM files, in two basecalls of the same molecules: "sup" (super accuracy) and
"hac" (high accuracy). Network access is needed; the reads are not stored.

A read is used for a window when it is a primary alignment with MAPQ >= 20 that spans the whole window and has no
insertion of 50 bp or more inside it and no deletion of 50 bp or more overlapping it (structural variants by the usual
definition, or alignment artifacts).
The read window is the read sequence from the base aligned to the window's first reference base to the base aligned to
its last one (insertions inside the window kept, insertions at its two edges not, except an insertion next to a deletion
of the window's first or last base, which is kept), oriented like the record (records whose sequence is the reverse
complement of the reference are reverse-complemented). Its divergence from the reference (substitutions, inserted and
deleted bases inside the window) is recorded in summary.json; it includes HG002's own variants. Records whose sequence differs from the alignment reference (the part of chromosome 21 that reference
hard-masks) are skipped.

Sets, per task, written like the noisy-read sets (<out>/<task>_<set>_revised/test.jsonl.gz, the same record fields):
  realref                      the records (reference sequence, unchanged) of every window with >= 10 reads usable in
                               both basecalls
  realsup, realsup-d1 .. -d9   draw k = the k-th of those reads (reads ordered by the SHA-256 of the read name), sup basecall
  realhac, realhac-d1 .. -d9   the same reads (same molecules), hac basecall
so every set of a task has the same windows in the same order, and realref is the clean comparison on those windows.

Also written:
  checksums.tsv     SHA-256 of every set file (sha256sum -c format); the sets are written deterministically, so
                    --expected benchmarks/noisy_reads/real_reads_sha256.tsv (passed by scripts/get_real_reads.sh and
                    scripts/reproduce.py) compares a new extraction with the paper's
Score them into their own directory: benchmarks/noisy_reads/score.py <tokenizer-study fold result> --data-root <out>
--sets realref realsup ... --output-dir results/real_reads/<twin>/nt18/<task>_revised (scripts/reproduce.py plan real-reads).

    python benchmarks/noisy_reads/real_reads.py
    python benchmarks/noisy_reads/real_reads.py --nt18 data/benchmarks/nt18 --genome data/GRCh38.p14.fasta.gz --out data/benchmarks/real_reads
"""
import argparse
import bisect
import collections
import gzip
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from locus.finetune.data import SequenceRecord, write_records  # noqa: E402

BASE = "https://ont-open-data.s3.amazonaws.com/giab_2023.05/analysis/hg002"
CRAMS = {"sup": f"{BASE}/sup/PAO89685.pass.cram", "hac": f"{BASE}/hac/PAO89685.pass.cram"}
TASKS = ["splice_sites_all", "splice_sites_acceptors", "splice_sites_donors", "promoter_all", "promoter_tata",
         "promoter_no_tata", "enhancers", "enhancers_types", "H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2",
         "H3K4me3", "H3K9ac", "H3K9me3", "H4K20me1"]
CHROMS = {"chr20": "NC_000020.11", "chr21": "NC_000021.9"}
REFGET = "https://www.ebi.ac.uk/ena/cram/md5"
CRAM_MD5 = {"chr20": "b18e6c531b0bd70e949a7fc20859cb01", "chr21": "974dc7aec0b755b19f031418fdedf293"}
DRAWS = 10
MIN_MAPQ = 20
# a read whose window holds an insertion or deletion of this length or more (a structural variant by the usual definition, or an
# alignment artifact; a deletion longer than the window leaves no bases at all) does not carry the labelled sequence and is not used
MAX_INDEL = 50
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")
M, I, D, N, S, H, P, EQ, X = range(9)
REF_OPS, QUERY_OPS = {M, D, N, EQ, X}, {M, I, S, EQ, X}


def revcomp(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def build_reference(genome: Path, out: Path) -> Path:
    """chr20 and chr21 exactly as in the CRAM header (MD5 checked), as an upper-case FASTA. chr20 of GRCh38.p14 matches;
    chr21 of the alignment reference (GRCh38 no-alt analysis set) has chr21:10,864,560-12,915,808 hard-masked (false
    duplications), so it is downloaded by its MD5 from the ENA CRAM reference registry. NT18 records inside the masked
    region no longer match the reference and are skipped by load_windows."""
    path = out / "reference" / "chr20_chr21.fa"
    if not path.exists():
        wanted = {v: k for k, v in CHROMS.items()}
        sequences, name, chunks = {}, None, []
        with gzip.open(genome, "rt") as handle:
            for line in handle:
                if line.startswith(">"):
                    if name in wanted:
                        sequences[wanted[name]] = "".join(chunks).upper()
                    name, chunks = line[1:].split()[0], []
                    if len(sequences) == len(wanted):
                        break
                elif name in wanted:
                    chunks.append(line.strip())
            if name in wanted and wanted[name] not in sequences:
                sequences[wanted[name]] = "".join(chunks).upper()
        for chrom, sequence in list(sequences.items()):
            digest = hashlib.md5(sequence.encode()).hexdigest()
            if digest != CRAM_MD5[chrom]:
                log(f"{chrom}: GRCh38.p14 MD5 {digest} differs from the CRAM header's; downloading {CRAM_MD5[chrom]}")
                with urllib.request.urlopen(f"{REFGET}/{CRAM_MD5[chrom]}", timeout=600) as response:
                    sequence = sequences[chrom] = response.read().decode().upper()
                digest = hashlib.md5(sequence.encode()).hexdigest()
                if digest != CRAM_MD5[chrom]:
                    sys.exit(f"{chrom}: downloaded MD5 {digest} differs from the CRAM header's {CRAM_MD5[chrom]}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as handle:
            for chrom in CHROMS:
                handle.write(f">{chrom}\n")
                for i in range(0, len(sequences[chrom]), 80):
                    handle.write(sequences[chrom][i:i + 80] + "\n")
        tmp.rename(path)
        import pysam

        pysam.faidx(str(path))
    return path


def load_windows(nt18: Path, reference):
    """records per task, and the distinct windows (chrom, start, end) -> list of (task, index, reverse)."""
    records, windows, problems = {}, collections.defaultdict(list), collections.Counter()
    for task in TASKS:
        with gzip.open(nt18 / f"{task}_revised" / "test.jsonl.gz", "rt") as handle:
            records[task] = [json.loads(line) for line in handle]
        for index, record in enumerate(records[task]):
            coordinates = record["name"].split("|")[0]
            chrom, span = coordinates.split(":")
            start, end = (int(x) for x in span.split("-"))
            ref = reference.fetch(chrom, start, end)
            sequence = record["sequence"].upper()
            if sequence == ref:
                reverse = False
            elif sequence == revcomp(ref):
                reverse = True
            else:
                problems[task] += 1
                continue
            windows[(chrom, start, end)].append((task, index, reverse))
    log(f"windows {len(windows)}; records not matching the reference (skipped): {dict(problems)}")
    return records, windows


def cigar_index(read):
    """Per CIGAR operation: op, length, reference start, query start (query_sequence coordinates, soft clips included)."""
    ops = np.array(read.cigartuples, dtype=np.int64)
    op, length = ops[:, 0], ops[:, 1]
    ref_len = np.where(np.isin(op, list(REF_OPS)), length, 0)
    query_len = np.where(np.isin(op, list(QUERY_OPS)), length, 0)
    ref_start = read.reference_start + np.concatenate(([0], np.cumsum(ref_len)[:-1]))
    query_start = np.concatenate(([0], np.cumsum(query_len)[:-1]))
    return op, length, ref_len, ref_start, query_start


def cut_window(read, index, start: int, end: int, ref_window: str):
    """(read sequence of the window, substitutions, inserted bases, deleted bases, longest insertion or deletion)."""
    op, length, ref_len, ref_start, query_start = index
    consuming = np.nonzero(ref_len)[0]
    starts = ref_start[consuming]
    first = consuming[np.searchsorted(starts, start, side="right") - 1]
    last = consuming[np.searchsorted(starts, end - 1, side="right") - 1]
    qs = query_start[first] + (start - ref_start[first] if op[first] in (M, EQ, X) else 0)
    qe = query_start[last] + (end - 1 - ref_start[last] + 1 if op[last] in (M, EQ, X) else 0)
    query = read.query_sequence
    substitutions = inserted = deleted = longest = 0
    for k in range(first, last + 1):
        o = op[k]
        if o == I:
            if start < ref_start[k] < end:
                inserted += length[k]
                longest = max(longest, int(length[k]))
        elif o in (D, N):
            deleted += min(end, ref_start[k] + length[k]) - max(start, ref_start[k])
            longest = max(longest, int(length[k]))
        elif o in (M, EQ, X):
            a, b = max(start, ref_start[k]), min(end, ref_start[k] + length[k])
            if b > a:
                qa = query_start[k] + (a - ref_start[k])
                read_part = np.frombuffer(query[qa:qa + b - a].encode(), dtype=np.uint8)
                ref_part = np.frombuffer(ref_window[a - start:b - start].encode(), dtype=np.uint8)
                substitutions += int((read_part != ref_part).sum())
    return query[qs:qe], substitutions, inserted, deleted, longest


def local_index(url: str, basecall: str, out: Path) -> Path:
    """The CRAM index, downloaded under a per-basecall name: both CRAMs are called PAO89685.pass.cram, and htslib's own
    download would reuse the first basecall's index for the second."""
    path = out / "reference" / f"{basecall}.crai"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url + ".crai", timeout=600) as response:
            data = response.read()
        path.with_suffix(".tmp").write_bytes(data)
        path.with_suffix(".tmp").rename(path)
    return path


def scan(url: str, index_path: Path, reference_path: Path, windows: dict, reference):
    """window -> {read name: (sequence, substitutions, inserted, deleted)} for every usable spanning read."""
    import pysam

    found = collections.defaultdict(dict)
    by_chrom = collections.defaultdict(list)
    for key in windows:
        by_chrom[key[0]].append(key)
    stats = collections.Counter()
    with pysam.AlignmentFile(url, "rc", reference_filename=str(reference_path), index_filename=str(index_path)) as cram:
        for chrom in CHROMS:
            keys = sorted(by_chrom[chrom], key=lambda k: (k[1], k[2]))
            if not keys:
                continue
            key_starts = [k[1] for k in keys]
            ref_windows = {k: reference.fetch(*k) for k in keys}
            for read in cram.fetch(chrom, keys[0][1], keys[-1][2]):
                stats["reads"] += 1
                if read.is_unmapped or read.is_secondary or read.is_supplementary or read.is_qcfail \
                        or read.mapping_quality < MIN_MAPQ or read.query_sequence is None:
                    stats["skipped"] += 1
                    continue
                lo = bisect.bisect_left(key_starts, read.reference_start)
                hi = bisect.bisect_right(key_starts, read.reference_end - 1)
                index = None
                for key in keys[lo:hi]:
                    if key[2] > read.reference_end:
                        continue
                    if index is None:
                        index = cigar_index(read)
                    sequence, substitutions, inserted, deleted, longest = cut_window(read, index, key[1], key[2], ref_windows[key])
                    if longest >= MAX_INDEL or not sequence:
                        stats[f"window skipped, indel >= {MAX_INDEL} bp"] += 1
                        continue
                    found[key][read.query_name] = (sequence, substitutions, inserted, deleted)
                    stats["windows cut"] += 1
                if stats["reads"] % 20000 == 0:
                    log(f"{url.split('/')[-2]} {chrom} {stats}")
    log(f"{url}: {dict(stats)}")
    return found


def name_order(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nt18", type=Path, default=ROOT / "data/benchmarks/nt18", help="prepared NT18 tasks (benchmarks/nt18/prepare.py)")
    parser.add_argument("--genome", type=Path, default=ROOT / "data/GRCh38.p14.fasta.gz", help="GRCh38.p14 (scripts/prepare_genome.sh)")
    parser.add_argument("--out", type=Path, default=ROOT / "data/benchmarks/real_reads")
    parser.add_argument("--limit", type=int, help="quick check: only the first LIMIT windows of each chromosome")
    parser.add_argument("--expected", type=Path, help="checksums the sets must match (sha256sum -c format, such as "
                        "benchmarks/noisy_reads/real_reads_sha256.tsv); summary.json is written only when they do")
    arguments = parser.parse_args()
    import pysam  # requirements.txt; imported here so that the other scripts do not need it

    arguments.out.mkdir(parents=True, exist_ok=True)
    reference_path = build_reference(arguments.genome, arguments.out)
    reference = pysam.FastaFile(str(reference_path))
    records, windows = load_windows(arguments.nt18, reference)
    if arguments.limit:
        keep = {}
        for chrom in CHROMS:
            for key in sorted((k for k in windows if k[0] == chrom), key=lambda k: k[1])[:arguments.limit]:
                keep[key] = windows[key]
        windows = keep
    found = {}
    for basecall, url in CRAMS.items():
        found[basecall] = scan(url, local_index(url, basecall, arguments.out), reference_path, windows, reference)

    summary = {"crams": CRAMS, "min_mapq": MIN_MAPQ, "max_indel": MAX_INDEL, "draws": DRAWS, "windows": len(windows), "tasks": {}}
    coverage = [len(set(found["sup"].get(k, {})) & set(found["hac"].get(k, {}))) for k in windows]
    if not coverage:
        sys.exit("no NT18 test window matches the alignment reference")
    summary["shared_reads_per_window"] = {"median": float(np.median(coverage)), "min": int(min(coverage)),
                                          f"windows with >= {DRAWS}": int(sum(c >= DRAWS for c in coverage))}
    chosen = {}
    for key in windows:
        shared = sorted(set(found["sup"].get(key, {})) & set(found["hac"].get(key, {})), key=name_order)
        if len(shared) >= DRAWS:
            chosen[key] = shared[:DRAWS]
    divergence = {b: collections.Counter() for b in CRAMS}
    per_read = {b: [] for b in CRAMS}
    for key, names in chosen.items():
        for basecall in CRAMS:
            for name in names:
                _, substitutions, inserted, deleted = found[basecall][key][name]
                divergence[basecall].update(substitutions=substitutions, inserted=inserted, deleted=deleted, bases=key[2] - key[1])
                per_read[basecall].append((substitutions + inserted + deleted) / (key[2] - key[1]))
    for task in TASKS:
        rows = {"realref": []}
        for basecall in CRAMS:
            for draw in range(DRAWS):
                rows[f"real{basecall}" + (f"-d{draw}" if draw else "")] = []
        for key, members in windows.items():
            if key not in chosen:
                continue
            for member_task, index, reverse in members:
                if member_task != task:
                    continue
                record = records[task][index]
                rows["realref"].append((index, dict(record)))
                for basecall in CRAMS:
                    for draw, name in enumerate(chosen[key]):
                        sequence = found[basecall][key][name][0]
                        out = dict(record)
                        out["sequence"] = revcomp(sequence) if reverse else sequence
                        out["name"] = f"{record['name']}:real{basecall}-d{draw}"
                        rows[f"real{basecall}" + (f"-d{draw}" if draw else "")].append((index, out))
        for name, items in rows.items():
            items.sort(key=lambda item: item[0])
            write_records(arguments.out / f"{task}_{name}_revised" / "test.jsonl.gz", [SequenceRecord(**record) for _, record in items])
        summary["tasks"][task] = {"test records": len(records[task]), "kept": len(rows["realref"])}
        log(f"{task}: kept {len(rows['realref'])} of {len(records[task])}")
    for basecall, counts in divergence.items():
        bases = max(counts["bases"], 1)
        summary[f"divergence_{basecall}"] = {k: counts[k] / bases for k in ("substitutions", "inserted", "deleted")}
        summary[f"divergence_{basecall}"]["total"] = sum(summary[f"divergence_{basecall}"].values())
        summary[f"divergence_{basecall}"]["per_read_quantiles"] = {
            str(q): float(np.quantile(per_read[basecall], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)} if per_read[basecall] else {}
    files = sorted(arguments.out.glob("*_revised/test.jsonl.gz"))
    checksums = {str(path.relative_to(arguments.out)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    (arguments.out / "checksums.tsv").write_text("".join(f"{digest}  {name}\n" for name, digest in checksums.items()))
    log(json.dumps(summary, indent=1))
    log(f"checksums.tsv: {len(files)} files")
    if arguments.expected:
        problems = checksum_problems(checksums, arguments.expected)
        if problems:
            sys.exit(f"the extracted reads differ from {arguments.expected} in {len(problems)} file(s), so summary.json is not "
                     "written:\n  " + "\n  ".join(problems[:20]) + ("\n  ..." if len(problems) > 20 else ""))
        log(f"identical to {arguments.expected} ({len(checksums)} files)")
    (arguments.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


def checksum_problems(checksums: dict, expected_path: Path) -> list:
    """The set files (path relative to the output directory -> SHA-256) that differ from, are missing in, or are not listed
    in a checksum file in sha256sum -c format."""
    expected = {name.strip().lstrip("*"): digest for digest, name in
                (line.split(None, 1) for line in expected_path.read_text().splitlines() if line.strip())}
    problems = [f"{name}: missing" if name not in checksums else f"{name}: differs"
                for name in sorted(expected) if checksums.get(name) != expected[name]]
    return problems + [f"{name}: not in {expected_path.name}" for name in sorted(set(checksums) - set(expected))]


if __name__ == "__main__":
    main()
