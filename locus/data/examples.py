"""Pretraining examples: a random crop of a genome window, read on one strand, segmented, with whole segments masked.

Every random draw of an example is seeded by (window identifier, pass seed, crop index), so the example stream does
not depend on the number of worker processes and an interrupted pass can be replayed exactly."""
import hashlib
import itertools
import multiprocessing
import random
from dataclasses import dataclass
from typing import Iterator, Sequence

from locus.alphabet import MASK_ID, OVERLAP_MASK_ID, reverse_complement
from locus.data.genome import GenomeRead
from locus.segmentation import Segment, Segmenter


@dataclass(frozen=True)
class SegmentedExample:
    sequence: str
    segments: Sequence[Segment]
    target_positions: Sequence[int]
    corrupted_target_ids: Sequence[int]
    corruption_actions: Sequence[str]
    eligible_position_count: int


def target_rng(key: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], byteorder="big"))


def crop_window(read: GenomeRead, min_context_length: int, seed: int) -> tuple[int, int]:
    """(start, length) of the crop: length uniform in [min_context_length, window length], start uniform."""
    if min_context_length <= 0 or min_context_length >= len(read.sequence):
        return 0, len(read.sequence)
    rng = target_rng(f"{read.read_id}:crop", seed)
    length = rng.randint(min_context_length, len(read.sequence))
    start = rng.randint(0, len(read.sequence) - length)
    return start, length


def covered_positions(sequence: str, segments: Sequence[Segment]) -> set[int]:
    """A/C/G/T positions that lie inside at least one segment."""
    change = [0] * (len(sequence) + 1)
    for item in segments:
        change[item.start] += 1
        change[item.end] -= 1
    covered, depth = set(), 0
    for position, base in enumerate(sequence):
        depth += change[position]
        if depth > 0 and base in "ACGT":
            covered.add(position)
    return covered


def select_segment_mask_targets(segments, sequence: str, candidates: set[int], rng: random.Random, mask_probability: float,
                                mask_replacement_probability: float, random_replacement_probability: float):
    """Masks whole segments: every segment that lies inside the eligible region is selected with `mask_probability`
    (at least one segment per example) and gets one corruption for all its bases: MASK, random bases, or unchanged.
    A base covered by two selected segments with different corrupted values gets OVERLAP_MASK_ID.
    Returns (target positions, corrupted input ids, corruption actions), sorted by position."""
    eligible = [item for item in segments
                if item.start in candidates and all(position in candidates for position in range(item.start, item.end))]
    if not eligible:
        raise ValueError("No segment lies entirely within the eligible target region")
    selected = [item for item in eligible if rng.random() < mask_probability]
    if not selected:
        selected = [rng.choice(eligible)]

    position_action: dict[int, tuple[str, int]] = {}
    conflicting: set[int] = set()
    for item in selected:
        draw = rng.random()
        if draw < mask_replacement_probability:
            action = "mask"
        elif draw < mask_replacement_probability + random_replacement_probability:
            action = "random"
        else:
            action = "unchanged"
        for position in range(item.start, item.end):
            if action == "mask":
                corrupted_id = MASK_ID
            elif action == "random":
                corrupted_id = rng.randrange(4)
            else:
                corrupted_id = "ACGT".index(sequence[position])
            if position in conflicting:
                continue
            if position not in position_action:
                position_action[position] = (action, corrupted_id)
            elif position_action[position][1] != corrupted_id:
                conflicting.add(position)
                del position_action[position]

    positions = sorted(set(position_action) | conflicting)
    corrupted_ids, actions = [], []
    for position in positions:
        if position in conflicting:
            corrupted_ids.append(OVERLAP_MASK_ID)
            actions.append("overlap_conflict")
        else:
            action, corrupted_id = position_action[position]
            corrupted_ids.append(corrupted_id)
            actions.append(action)
    return positions, corrupted_ids, actions


def build_example(read: GenomeRead, segmenter: Segmenter, data_config, seed: int, crop_index: int) -> SegmentedExample:
    """Crop `crop_index` of `read` in a pass seeded with `seed`. Odd crop indices are read on the reverse strand, which is
    segmented on its own (LCP is not strand-symmetric). Raises ValueError when a target-eligible base is not covered by
    any segment (rare low-complexity crops under LCP); such crops are skipped."""
    seed = seed if crop_index == 0 else seed * 1_000_003 + crop_index
    start, length = crop_window(read, data_config.min_context_length, seed)
    sequence = read.sequence[start:start + length]
    if crop_index % 2 == 1:
        sequence = reverse_complement(sequence)
    segments = segmenter.segment(sequence)
    candidates = [position for position in range(data_config.target_margin, len(sequence) - data_config.target_margin)
                  if sequence[position] in "ACGT"]
    if not candidates:
        raise ValueError(f"{read.read_id} has no eligible A/C/G/T positions")
    covered = covered_positions(sequence, segments)
    uncovered = [position for position in candidates if position not in covered]
    if uncovered:
        raise ValueError(f"{data_config.segmenter} does not cover eligible position {uncovered[0]} in {read.read_id}")
    rng = target_rng(read.read_id, seed)
    positions, corrupted_ids, actions = select_segment_mask_targets(
        segments, sequence, set(candidates), rng, data_config.mask_probability,
        data_config.mask_replacement_probability, data_config.random_replacement_probability)
    return SegmentedExample(sequence=sequence, segments=segments, target_positions=positions, corrupted_target_ids=corrupted_ids,
                            corruption_actions=actions, eligible_position_count=len(candidates))


def shuffle_with_buffer(items, buffer_size: int, rng: random.Random):
    """Streaming shuffle: keeps `buffer_size` items and emits a random one for each new item."""
    buffer = []
    for item in items:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = item
    rng.shuffle(buffer)
    yield from buffer


def pass_items(dataset, data_config, training_config, pass_index: int):
    """(window, crop index) items of one pass in their training order. The order is a function of the seed and the pass
    only, so a resumed run can skip the items it has already consumed without building them."""
    items = ((read, crop) for read in dataset for crop in range(data_config.crops_per_read))
    return shuffle_with_buffer(items, training_config.shuffle_buffer_size,
                               random.Random(training_config.seed * 1_000_003 + pass_index))


_WORKER_SEGMENTER = None
_WORKER_DATA_CONFIG = None


def _init_worker(data_config) -> None:
    global _WORKER_SEGMENTER, _WORKER_DATA_CONFIG
    _WORKER_SEGMENTER = Segmenter(data_config)
    _WORKER_DATA_CONFIG = data_config


def _build_job(job):
    index, read, seed, crop_index = job
    try:
        return index, build_example(read, _WORKER_SEGMENTER, _WORKER_DATA_CONFIG, seed, crop_index)
    except (ValueError, RuntimeError):
        return index, None


def pass_examples(dataset, data_config, training_config, pass_index: int, seed: int, skip_items: int = 0) -> Iterator:
    """(item index, example) for the items of a pass, skipping the first `skip_items` and the items whose example cannot
    be built. With data_config.example_workers > 1 the examples are built in "spawn" worker processes, in chunks of 64
    with one chunk in flight; the order is unchanged."""
    items = enumerate(pass_items(dataset, data_config, training_config, pass_index))
    jobs = ((index, read, seed, crop) for index, (read, crop) in items if index >= skip_items)
    if data_config.example_workers <= 1:
        segmenter = Segmenter(data_config)
        for index, read, seed_, crop in jobs:
            try:
                yield index, build_example(read, segmenter, data_config, seed_, crop)
            except (ValueError, RuntimeError):
                continue
        return
    context = multiprocessing.get_context("spawn")
    with context.Pool(data_config.example_workers, initializer=_init_worker, initargs=(data_config,)) as pool:
        pending = None
        while True:
            chunk = list(itertools.islice(jobs, 64))
            submitted = pool.map_async(_build_job, chunk, chunksize=4) if chunk else None
            if pending is not None:
                for index, example in pending.get():
                    if example is not None:
                        yield index, example
            if submitted is None:
                return
            pending = submitted


def batch_examples(examples, batch_size: int) -> Iterator[list]:
    """Consecutive groups of `batch_size`; an incomplete last group is dropped."""
    batch = []
    for example in examples:
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
