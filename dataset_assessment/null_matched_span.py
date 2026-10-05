"""The random subsampler's Delta-over-Raw with both streams scored over the same input span.

MESR averages complete 30,000-event slices and drops the tail. When `random_null` keeps a
fraction r of a recording, the slices it scores therefore come from a leading stretch of the
input whose length depends on r: at r = 0.05 a capped 10^6-event recording keeps 50,000
events, of which the first 30,000 are scored, and those were drawn from roughly the first 60%
of the input, while the unfiltered stream's 33 slices cover 99% of it. Delta-over-Raw then
compares two input spans as well as two retentions. A comparison at an equal retained count
(the label oracle, the blank-sample filters, the matched null of the E-MLB table) keeps the
same count in every input block on both sides, so both sides cover the same span.

Here both sides are scored on one leading frame of the input, chosen so that neither has an
incomplete slice. For r = u/v in lowest terms the frame holds

    L = SLICE * v * floor(E / (SLICE * v))

input events. The unfiltered frame is L / SLICE whole slices, and the block quota rebuilt on
the frame keeps exactly r * L = SLICE * u * floor(E / (SLICE * v)) events, also whole slices,
because BLOCK * r is an integer on the 0.05 grid. The quota is recomputed on the frame rather
than cropped from the full-stream mask, since cropping would change the last block's quota.
The subsampler's priorities are the ones the benchmark drew for that recording (the first L
of them), so the frame changes which input span is scored and nothing else.

A recording shorter than one frame at a given r is left out at that r, and the original
Delta is summarised on the same recordings beside the matched one.

Every corpus is scored under each of the ten seeds of `results/null_seeds/`; recordings are
scored in parallel, which changes no value.

Writes `results/null_matched_span.json`. Run:

    python -m dataset_assessment.null_matched_span --workers 16
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from fractions import Fraction
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from . import denoisors
from .esr import BLOCK, SLICE, mesr, retain_mask
from .run_benchmark import CAPPED_LOADERS, LOADERS, RETENTIONS

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "null_matched_span.json"
MAX_EVENTS = 1_000_000

#: The seeds of `results/null_seeds/`; the first is the benchmark's own draw.
SEEDS = tuple(range(20260726, 20260736))

#: Each corpus is scored under every draw in SEEDS.
CORPORA = ("dvsd22", "pure_ba", "emlb", "dnd21", "dvsclean")

GRID = [float(r) for r in RETENTIONS if r < 1.0]


def denominator(retention: float) -> int:
    """v in r = u/v, lowest terms, for a retention on the 0.05 grid."""

    return Fraction(retention).limit_denominator(100).denominator


def frame_length(n_events: int, retention: float, slice_size: int = SLICE) -> int:
    """The longest leading frame on which both sides hold whole slices; 0 if none fits."""

    step = slice_size * denominator(retention)
    return step * (n_events // step)


def null_scores(recording, seed: int) -> np.ndarray:
    """The subsampler's priorities for one draw, exactly as the benchmark assigns them."""

    saved = denoisors.RANDOM_NULL_SEED
    denoisors.RANDOM_NULL_SEED = seed
    try:
        return denoisors.score_events(denoisors.RANDOM_NULL, recording)
    finally:
        denoisors.RANDOM_NULL_SEED = saved


def scored_span(keep: np.ndarray, slice_size: int = SLICE) -> float:
    """Share of the input up to the last event MESR scores from a kept mask."""

    kept = np.flatnonzero(keep)
    scored = slice_size * (len(kept) // slice_size)
    return float((kept[scored - 1] + 1) / len(keep)) if scored else float("nan")


def compare(x: np.ndarray, y: np.ndarray, width: int, height: int, scores: np.ndarray,
            retentions: Iterable[float] = GRID) -> List[Dict]:
    """Original and matched-span Delta for one recording and one draw.

    `delta_original` is the benchmark's quantity: the thinned stream's MESR minus the whole
    unfiltered stream's. `delta_matched` scores both on the frame of `frame_length`.
    """

    n_events = len(x)
    raw = mesr(x, y, width, height)
    frame_raw: Dict[int, float] = {}
    rows = []
    for r in retentions:
        keep = retain_mask(scores, r, BLOCK)
        original = mesr(x[keep], y[keep], width, height)
        frame = frame_length(n_events, r)
        row = {"r": r, "frame": frame, "frame_share": frame / n_events,
               "delta_original": original - raw if np.isfinite(original) else None,
               "span_thinned": scored_span(keep),
               "span_unfiltered": SLICE * (n_events // SLICE) / n_events,
               "delta_matched": None}
        if frame > 0:
            framed = retain_mask(scores[:frame], r, BLOCK)
            expected = int(round(r * frame))
            if int(framed.sum()) != expected:
                raise AssertionError(f"quota kept {int(framed.sum())} of an expected "
                                     f"{expected} at r = {r}, frame {frame}")
            if frame not in frame_raw:
                frame_raw[frame] = mesr(x[:frame], y[:frame], width, height)
            matched = mesr(x[:frame][framed], y[:frame][framed], width, height)
            row["delta_matched"] = matched - frame_raw[frame]
        rows.append(row)
    return rows


def _summary(values: Sequence[float]) -> Dict:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "median": float(np.median(array)),
            "positive": int((array > 0).sum())}


def summarise(per_recording: Dict[str, List[Dict]]) -> List[Dict]:
    """Per retention, on the recordings long enough to hold a frame there.

    The original Delta is reported twice: on that same cohort, which is what the matched
    value is compared with, and on every recording where it is evaluable, which is the
    benchmark's own corpus mean.
    """

    out = []
    for r in GRID:
        points = [next(p for p in rows if p["r"] == r) for rows in per_recording.values()]
        full = [p["delta_original"] for p in points if p["delta_original"] is not None]
        paired = [p for p in points if p["delta_matched"] is not None]
        if not paired:
            out.append({"r": r, "n": 0, "n_original": len(full)})
            continue
        original = _summary([p["delta_original"] for p in paired])
        matched = _summary([p["delta_matched"] for p in paired])
        out.append({
            "r": r, "n": len(paired), "n_original": len(full),
            "mean_original_all": float(np.mean(full)) if full else None,
            "mean_original": original["mean"], "median_original": original["median"],
            "positive_original": original["positive"],
            "mean_matched": matched["mean"], "median_matched": matched["median"],
            "positive_matched": matched["positive"],
            "frame_share_median": float(np.median([p["frame_share"] for p in paired])),
            "span_thinned_median": float(np.nanmedian([p["span_thinned"] for p in paired])),
            "span_unfiltered_median": float(np.median([p["span_unfiltered"]
                                                       for p in paired])),
        })
    return out


def _same_sign(row: Dict) -> bool:
    """Whether the matched mean has the original mean's sign on the same recordings."""

    return bool(np.sign(row["mean_matched"]) == np.sign(row["mean_original"]))


def across_draws(by_seed: Sequence[Dict]) -> List[Dict]:
    """Per retention, the matched mean's range over the draws and how many keep each sign."""

    out = []
    for index, r in enumerate(GRID):
        rows = [entry["by_retention"][index] for entry in by_seed
                if entry["by_retention"][index]["n"]]
        if any(row["r"] != r for row in rows):
            raise AssertionError(f"retention rows out of grid order at r = {r}")
        if not rows:
            continue
        means = [row["mean_matched"] for row in rows]
        out.append({"r": r, "matched_mean_min": float(min(means)),
                    "matched_mean_max": float(max(means)),
                    "seeds_positive": int(sum(m > 0 for m in means)),
                    "seeds_negative": int(sum(m < 0 for m in means)),
                    "seeds_same_sign": int(sum(_same_sign(row) for row in rows)),
                    "seeds": len(rows)})
    return out


def _recordings(dataset: str):
    loader = LOADERS[dataset]
    kwargs = {"max_events": MAX_EVENTS} if dataset in CAPPED_LOADERS else {}
    for rec in loader(**kwargs):
        events = rec.events[:MAX_EVENTS]
        yield replace(rec, events=events,
                      labels=None if rec.labels is None else rec.labels[:len(events)])


def _one(recording, seeds: Sequence[int]):
    """One recording under every draw; module level so the pool can pickle it."""

    return recording.name, {seed: compare(recording.x, recording.y, recording.width,
                                          recording.height, null_scores(recording, seed))
                            for seed in seeds}


def analyse(dataset: str, seeds: Sequence[int] = SEEDS, workers: int = 8) -> Dict:
    seeds = tuple(seeds)
    per_seed: Dict[int, Dict[str, List[Dict]]] = {seed: {} for seed in seeds}
    started = time.perf_counter()
    recordings = (rec for rec in _recordings(dataset) if len(rec.events) >= SLICE)
    # Spawned, not forked: the parent holds pandas' and MKL's thread pools, and a child
    # forked from a multi-threaded process can deadlock.
    with get_context("spawn").Pool(workers) as pool:
        worker = partial(_one, seeds=seeds)
        for index, (name, draws) in enumerate(pool.imap(worker, recordings, chunksize=1)):
            for seed, rows in draws.items():
                per_seed[seed][name] = rows
            if index % 48 == 0:
                print(f"[{dataset} {index + 1}] {name} ({time.perf_counter() - started:.0f}s)",
                      flush=True)

    by_seed = []
    for seed in seeds:
        rows = summarise(per_seed[seed])
        paired = [row for row in rows if row["n"]]
        by_seed.append({
            "seed": seed, "by_retention": rows, "n_retentions": len(paired),
            "n_positive_matched": sum(row["mean_matched"] > 0 for row in paired),
            "n_positive_original": sum(row["mean_original"] > 0 for row in paired),
            "sign_changes": [row["r"] for row in paired if not _same_sign(row)]})

    across = across_draws(by_seed)
    default = per_seed[seeds[0]]
    return {
        "seeds": list(seeds),
        "n_recordings": len(default),
        "by_seed": by_seed,
        "across_seeds": across,
        "retentions_positive_under_every_seed": sum(
            a["seeds_positive"] == a["seeds"] for a in across),
        "retentions_negative_under_every_seed": sum(
            a["seeds_negative"] == a["seeds"] for a in across),
        "retentions_same_sign_under_every_seed": sum(
            a["seeds_same_sign"] == a["seeds"] for a in across),
        "recordings": [{"recording": name, "points": points}
                       for name, points in default.items()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(CORPORA),
                        choices=sorted(CORPORA))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if SEEDS[0] != denoisors.RANDOM_NULL_SEED:
        raise AssertionError("the first draw must be the benchmark's own")
    started = time.perf_counter()
    result = {"slice": SLICE, "block": BLOCK, "max_events": MAX_EVENTS,
              "frame_rule": "L = slice * v * floor(E / (slice * v)) for r = u/v",
              "corpora": {}}
    for dataset in args.datasets:
        block = analyse(dataset, SEEDS, args.workers)
        result["corpora"][dataset] = block
        first = block["by_seed"][0]
        print(f"\n{dataset}: {block['n_recordings']} recordings, {len(block['seeds'])} draw(s); "
              f"matched mean positive at {first['n_positive_matched']}/{first['n_retentions']} "
              f"(original, same cohort: {first['n_positive_original']}); under every draw "
              f"positive at {block['retentions_positive_under_every_seed']}, negative at "
              f"{block['retentions_negative_under_every_seed']}, of the original's sign at "
              f"{block['retentions_same_sign_under_every_seed']}")
        for row in first["by_retention"]:
            if row["n"]:
                print(f"  r={row['r']:.2f} n={row['n']:3d}/{row['n_original']:3d}  "
                      f"original {row['mean_original']:+.4f} ({row['positive_original']})  "
                      f"matched {row['mean_matched']:+.4f} ({row['positive_matched']})  "
                      f"span {row['span_thinned_median']:.2f} vs "
                      f"{row['span_unfiltered_median']:.2f}")
        result["wall_seconds"] = time.perf_counter() - started
        args.out.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"\nWrote {args.out} in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
