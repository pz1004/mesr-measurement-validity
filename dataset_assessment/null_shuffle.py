"""Does the random subsampler's gain need the stream's temporal order, and at what scale?

On DVSD22 and Pure_BA the random subsampler raises MESR at every retention, and on DND21 it
lowers it (Sec. V-B). Removing the busiest 0.1% of occupied pixels removes most of the gain,
which places it in those pixels' activity without saying whether the *order* of that
activity matters or only how much of it each pixel emits. This removes the order and keeps
the counts.

Within consecutive windows of `k` input events, the marks `(x, y, p)` are permuted at random
over the window's timestamps. Every pixel keeps its count, over the stream and within each
window, so the busiest-pixel share is unchanged; any temporal organisation shorter than `k`
events is destroyed. Both label-blind controls are then scored on the shuffled stream as the
benchmark scores them, against that stream's own unfiltered MESR.

Predictions, fixed before the run:

* With one window spanning the whole capped stream, the stream is exchangeable: every slice
  MESR scores, thinned or not, is a simple random sample of the same events, so random
  removal is neutral in expectation whatever the pixel counts. A gain that the busiest-pixel
  share does not remove but this shuffle does needs temporal order.
* If the gain comes from a count-delimited slice covering a longer stretch of input once
  events are discarded, it survives windows much shorter than a slice (N = 30,000 events)
  and vanishes once a window covers what a thinned slice spans, about N / r input events.

Either outcome is reported as it comes. The windows locate a time scale; they do not show
which process sets it. `k = 0` scores the stream unshuffled and must reproduce the
benchmark's own curves, which the run asserts.

Writes `results/null_shuffle.json` once every corpus is done; progress after each corpus
goes to `null_shuffle.partial.json`, so a failed run never replaces a complete artifact. Run:

    python -m dataset_assessment.null_shuffle --workers 12
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import denoisors
from .esr import mesr, retain_mask
from .null_matched_span import GRID, _recordings

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "null_shuffle.json"

CORPORA = ("dvsd22", "pure_ba", "dnd21")
#: Window lengths in input events; 0 leaves the stream as recorded, None shuffles it whole.
WINDOWS: Sequence[Optional[int]] = (0, 1_000, 10_000, 30_000, 100_000, 300_000, None)
SHUFFLE_SEEDS = (20261016, 20261017, 20261018)
#: Table II's retention per corpus, where the subsampler's corpus-mean Delta peaks.
REFERENCE_R = {"dvsd22": 0.05, "pure_ba": 0.20, "dnd21": 0.95}
CONTROLS = ("raw", denoisors.RANDOM_NULL)


def shuffle_marks(events: np.ndarray, window: Optional[int], seed: int,
                  name: str = "") -> np.ndarray:
    """`events` with `(x, y, p)` permuted within consecutive windows of `window` rows.

    Timestamps stay where they are. `window` 0 returns the events unchanged; None permutes
    the whole stream. The generator is keyed on the seed, the window and the recording, so
    each recording's shuffle is reproducible on its own.
    """

    if window == 0 or len(events) == 0:
        return events
    size = len(events) if window is None else window
    rng = np.random.default_rng([seed, size, zlib.crc32(name.encode())])
    out = events.copy()
    for start in range(0, len(events), size):
        stop = min(start + size, len(events))
        out[start:stop, 1:] = events[start + rng.permutation(stop - start), 1:]
    return out


def deltas(recording) -> Dict[str, Any]:
    """Each control's Delta-over-Raw at every grid retention; None where unevaluable."""

    x, y = recording.x, recording.y
    unfiltered = mesr(x, y, recording.width, recording.height)
    out: Dict[str, Any] = {}
    for control in CONTROLS:
        scores = denoisors.score_events(control, recording)
        row = []
        for r in GRID:
            keep = retain_mask(scores, r)
            value = mesr(x[keep], y[keep], recording.width, recording.height)
            row.append(float(value - unfiltered) if np.isfinite(value) else None)
        out[control] = row
    out["unfiltered_mesr"] = float(unfiltered)
    return out


def _one(recording) -> Dict:
    """Every window and seed on one recording; module level so the pool can pickle it."""

    runs = []
    for window in WINDOWS:
        for seed in (SHUFFLE_SEEDS[:1] if window == 0 else SHUFFLE_SEEDS):
            # Each control scores the shuffled stream itself, whatever it reads of it.
            shuffled = replace(recording, events=shuffle_marks(recording.events, window, seed,
                                                               recording.name))
            runs.append({"window": window, "seed": seed, **deltas(shuffled)})
    return {"recording": recording.name, "events": int(len(recording.events)), "runs": runs}


def benchmark_records(dataset: str) -> Dict[str, Dict]:
    """The benchmark's records by name; read once, before any shuffle is scored."""

    path = RESULTS / f"benchmark_{dataset}.json"
    return {r["recording"]: r for r in json.loads(path.read_text())["records"]}


def check_unshuffled(records: Dict[str, Dict], recordings: Sequence[Dict]) -> int:
    """Window 0 must be the benchmark's own curves, value for value. Returns cells checked."""

    checked = 0
    for rec in recordings:
        if rec["recording"] not in records:
            raise AssertionError(f"{rec['recording']} is not in the benchmark artifact; it is "
                                 "incomplete or from another run")
        bench = records[rec["recording"]]
        run = next(run for run in rec["runs"] if run["window"] == 0)
        for control in CONTROLS:
            curve = {point["r"]: point for point in bench["methods"][control]["curve"]}
            for r, delta in zip(GRID, run[control]):
                point = curve[r]
                expected = point["mesr"] - bench["raw_mesr"] if point["evaluable"] else None
                if (delta is None) != (expected is None) or (
                        delta is not None and abs(delta - expected) > 1e-12):
                    raise AssertionError(f"{rec['recording']} {control} r={r}: unshuffled "
                                         f"Delta {delta} against the benchmark's {expected}")
                checked += 1
    return checked


def summarise(recordings: Sequence[Dict], reference_r: float) -> List[Dict]:
    """Per (window, control): retentions with a positive mean, and Delta at the reference."""

    index = GRID.index(reference_r)
    out = []
    for window in WINDOWS:
        seeds = SHUFFLE_SEEDS[:1] if window == 0 else SHUFFLE_SEEDS
        for control in CONTROLS:
            per_seed: List[Dict[str, Any]] = []
            for seed in seeds:
                runs = [next(run for run in rec["runs"]
                             if run["window"] == window and run["seed"] == seed)
                        for rec in recordings]
                means = []
                for j in range(len(GRID)):
                    values = [run[control][j] for run in runs if run[control][j] is not None]
                    means.append(float(np.mean(values)) if values else None)
                evaluable = [m for m in means if m is not None]
                per_seed.append({
                    "seed": seed, "means": means, "evaluable": len(evaluable),
                    "positive": int(sum(m > 0 for m in evaluable)),
                    "largest_positive": float(max([m for m in evaluable if m > 0],
                                                  default=float("nan"))),
                    "at_reference": means[index]})
            references = [s["at_reference"] for s in per_seed if s["at_reference"] is not None]
            out.append({"window": window, "control": control, "per_seed": per_seed,
                        "positive": [s["positive"] for s in per_seed],
                        "evaluable": per_seed[0]["evaluable"],
                        "at_reference_min": float(min(references)) if references else None,
                        "at_reference_max": float(max(references)) if references else None})
    return out


def analyse(dataset: str, workers: int) -> Dict:
    started = time.perf_counter()
    records = benchmark_records(dataset)
    recordings, checked = [], 0
    with get_context("spawn").Pool(workers) as pool:
        for result in pool.imap(_one, _recordings(dataset), chunksize=1):
            checked += check_unshuffled(records, [result])      # fails at the first mismatch
            recordings.append(result)
    block: Dict[str, Any] = {
        "recordings": len(recordings), "reference_r": REFERENCE_R[dataset],
        "unshuffled_cells_matching_benchmark": checked,
        "summary": summarise(recordings, REFERENCE_R[dataset]),
        "per_recording": recordings, "wall_seconds": time.perf_counter() - started}
    for row in block["summary"]:
        if row["control"] == denoisors.RANDOM_NULL and row["at_reference_min"] is not None:
            print(f"{dataset} k={row['window']}: positive at {row['positive']} of "
                  f"{row['evaluable']}; Delta at r={REFERENCE_R[dataset]} "
                  f"[{row['at_reference_min']:+.4f}, {row['at_reference_max']:+.4f}]",
                  flush=True)
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    started = time.perf_counter()
    payload: Dict[str, Any] = {"windows": list(WINDOWS), "seeds": list(SHUFFLE_SEEDS),
                               "null_seed": denoisors.RANDOM_NULL_SEED, "grid": GRID,
                               "corpora": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.stem + ".partial.json")
    for dataset in CORPORA:
        payload["corpora"][dataset] = analyse(dataset, args.workers)
        payload["wall_seconds"] = time.perf_counter() - started
        partial.write_text(json.dumps(payload, indent=2) + "\n")
    partial.replace(args.out)
    print(f"\nWrote {args.out} ({payload['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
