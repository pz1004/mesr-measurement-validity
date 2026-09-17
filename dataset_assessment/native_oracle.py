"""Does MESR invert label-count ordering on the filters' *own* outputs, not a quota near them?

`label_quality` matches a filter against the label oracle at a swept retention, so both sides
retain the same count per block and the comparison is exact. `native_mask` then showed what
that costs: at the exact native rate the swept mask still differs from the filter's own mask
in a median 25-27% of the retained set, and recovers it exactly in none of 168 cells. So the
objects compared there are filter-plus-quota procedures, not the released filters.

This module removes the quota. It takes the filter's own mask, keeps the part MESR actually
reads, and builds the oracle to match *that*:

1. `F` is the native mask -- every event the filter accepts, no retention argument anywhere.
2. `esr.mesr` scores complete slices and drops the tail, so of `|F|` retained events it reads
   only the first ``m = SLICE * (|F| // SLICE)``. That prefix `F_s` is the comparison object.
   Cells with ``m == 0`` are unevaluable and are reported, not refilled at a nearby retention.
3. Per input block `b`, ``a_b = |F_s and I_b|`` -- how many of the scored events the filter
   drew from that block.
4. The oracle keeps exactly `a_b` events in block `b`, signal first, ties by arrival order.
   It never reads MESR.

Two properties make the comparison exact, and both are asserted rather than assumed:

* **Equal scored counts.** The oracle holds exactly `m` events, and `m` is a multiple of
  `SLICE`, so the oracle's own scored prefix is the whole of it. Both sides are scored on
  exactly `m` events. Quotas are built from the scored subset rather than the full native
  mask precisely so this holds -- taking them from `F` would leave the oracle a tail of its
  own to truncate, reintroducing the ambiguity this module exists to remove.
* **Oracle dominance.** ``TP_oracle = sum_b min(a_b, S_b) >= TP_{F_s}``, since `F_s` draws
  `a_b` events from block `b` and at most `min(a_b, S_b)` of them can be signal. A cell
  classified ``impossible`` therefore means the mask or the oracle is wrong, not that a
  filter beat the oracle. It must never occur.

A ``strict`` cell is one where the filter retains strictly less labelled signal -- hence
strictly more labelled noise, at equal count -- and MESR still scores it above the oracle.
That is a counterexample on the released filter's own output.

**Hot-pixel removal.** `--hot-pixel-removal` applies `hotpixel.remove_hot_pixels` to the
*input*, after the event cap and before any filter runs, so the filters see the trimmed stream
and the oracle is rebuilt from their new scored prefixes. Deleting pixels from an existing
comparison would evaluate a different pipeline. The rule reads event counts only, never MESR.

Writes `results/native_oracle_<dataset>.json` (`_hotpixel` appended with the flag). Run:

    python -m dataset_assessment.native_oracle --dataset dnd21
    python -m dataset_assessment.native_oracle --dataset dvsclean
    python -m dataset_assessment.native_oracle --dataset dnd21 --hot-pixel-removal
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .denoisors import CLASSICAL, score_events
from .esr import BLOCK, SLICE, mesr
from .hotpixel import remove_hot_pixels
from .label_quality import _classify, _quality
from .readers import iter_dnd21, iter_dvsclean

#: The two corpora with per-event labels. Everything here needs them.
LOADERS = {"dnd21": iter_dnd21, "dvsclean": iter_dvsclean}

OUT_DIR = Path(__file__).resolve().parents[1] / "results"


def scored_prefix(accepted: np.ndarray, slice_size: int = SLICE) -> np.ndarray:
    """The part of a native mask MESR reads: its first complete-slice multiple.

    `accepted` is a boolean mask over the input. The returned mask is a subset of it holding
    exactly ``slice_size * (accepted.sum() // slice_size)`` events, all-False when the filter
    retained less than one slice.
    """

    kept = np.flatnonzero(accepted)
    m = len(kept) // slice_size * slice_size
    prefix = np.zeros_like(accepted)
    prefix[kept[:m]] = True
    return prefix


def block_counts(mask: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """`a_b`: how many of the mask's events fall in each input block."""

    return np.array([int(mask[s:s + block].sum())
                     for s in range(0, len(mask), block)], dtype=np.int64)


def oracle_mask(labels: np.ndarray, counts: np.ndarray,
                block: int = BLOCK) -> np.ndarray:
    """Keep exactly `counts[b]` events in block `b`, labelled signal first, ties by arrival.

    `labels` is 1 for noise, 0 for signal, so a stable argsort on the label puts every signal
    event in the block ahead of every noise event and preserves arrival order within each
    group. The tie-break never consults MESR.
    """

    keep = np.zeros(len(labels), dtype=bool)
    for index, start in enumerate(range(0, len(labels), block)):
        quota = int(counts[index])
        if quota <= 0:
            continue
        stop = min(start + block, len(labels))
        order = np.argsort(labels[start:stop], kind="stable")
        keep[start + order[:quota]] = True
    return keep


def available_signal(labels: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """`S_b`: labelled signal events available in each input block."""

    return np.array([int((labels[s:s + block] == 0).sum())
                     for s in range(0, len(labels), block)], dtype=np.int64)


def prepare(rec, max_events: int, hot_pixel_removal: bool = False):
    """The input every filter sees: capped, then (optionally) with hot pixels removed.

    Returns the recording and the removal summary, which is None when removal is off.
    """

    rec = replace(rec, events=rec.events[:max_events],
                  labels=None if rec.labels is None else rec.labels[:max_events])
    if not hot_pixel_removal:
        return rec, None
    return remove_hot_pixels(rec)


def measure(dataset: str, max_events: int = 1_000_000,
            max_recordings: Optional[int] = None,
            methods: Sequence[str] = CLASSICAL,
            hot_pixel_removal: bool = False) -> Dict:
    """Every (recording, method) cell, each filter scored on its own native output."""

    cells: List[Dict] = []
    unevaluable: List[Dict] = []
    removals: List[Dict] = []
    signal_total = 0
    noise_total = 0
    started = time.perf_counter()

    for index, rec in enumerate(LOADERS[dataset]()):
        if max_recordings is not None and index >= max_recordings:
            break
        rec, hot_summary = prepare(rec, max_events, hot_pixel_removal)
        if hot_summary is not None:
            removals.append({"recording": rec.name, **hot_summary})
        if rec.labels is None:
            raise ValueError(f"{rec.name} has no labels; this analysis needs them")

        labels = rec.labels
        n_signal = int((labels == 0).sum())
        n_noise = int((labels == 1).sum())
        signal_total += n_signal
        noise_total += n_noise
        per_block_signal = available_signal(labels)

        for method in methods:
            scores = score_events(method, rec)
            accepted = scores == 0.0
            native = float(accepted.mean())
            prefix = scored_prefix(accepted)
            m = int(prefix.sum())
            if m == 0:
                unevaluable.append({
                    "recording": rec.name, "method": method,
                    "native_retention": native, "native_kept": int(accepted.sum()),
                    "reason": "native output is shorter than one complete slice"})
                continue

            counts = block_counts(prefix)
            oracle = oracle_mask(labels, counts)

            mq = _quality(labels, prefix)
            oq = _quality(labels, oracle)
            # Both sides hold exactly `m` events and `m` is a multiple of SLICE, so both are
            # scored whole. If that fails the comparison is meaningless, so stop.
            if mq["kept"] != oq["kept"] or mq["scored"] != oq["scored"] or mq["scored"] != m:
                raise AssertionError(
                    f"{rec.name} {method}: filter scored {mq['scored']} against the oracle's "
                    f"{oq['scored']} (m = {m}); the matched comparison is invalid")
            # The oracle takes min(a_b, S_b) signal per block, which bounds the filter's.
            predicted_tp = int(np.minimum(counts, per_block_signal).sum())
            if oq["tp"] != predicted_tp:
                raise AssertionError(
                    f"{rec.name} {method}: oracle retained {oq['tp']} signal events against "
                    f"the sum_b min(a_b, S_b) bound of {predicted_tp}")

            method_mesr = mesr(rec.x[prefix], rec.y[prefix], rec.width, rec.height)
            oracle_mesr = mesr(rec.x[oracle], rec.y[oracle], rec.width, rec.height)
            if not (np.isfinite(method_mesr) and np.isfinite(oracle_mesr)):
                unevaluable.append({
                    "recording": rec.name, "method": method,
                    "native_retention": native, "native_kept": int(accepted.sum()),
                    "reason": "MESR is not finite on the native output"})
                continue

            cells.append({
                "recording": rec.name, "method": method,
                "native_retention": native,
                "native_kept": int(accepted.sum()),
                "scored": m, "slices": m // SLICE,
                "scored_fraction": mq["scored_fraction"],
                "n_signal": n_signal, "n_noise": n_noise,
                "mesr": float(method_mesr), "oracle_mesr": float(oracle_mesr),
                "mesr_win": bool(method_mesr > oracle_mesr),
                "tp": mq["tp"], "fp": mq["fp"],
                "oracle_tp": oq["tp"], "oracle_fp": oq["fp"],
                "tp_scored": mq["tp_scored"], "fp_scored": mq["fp_scored"],
                "oracle_tp_scored": oq["tp_scored"],
                "oracle_fp_scored": oq["fp_scored"],
                "signal_retention": mq["tp"] / n_signal if n_signal else float("nan"),
                "noise_retention": mq["fp"] / n_noise if n_noise else float("nan"),
                "label_class": _classify(mq, oq),
                "label_class_full_mask": _classify(mq, oq, key="tp"),
            })
        print(f"[{index}] {rec.name}: {len(cells)} cells so far", flush=True)

    return {"dataset": dataset, "max_events": max_events,
            "hot_pixel_removal": hot_pixel_removal, "removals": removals,
            "signal_events": signal_total, "noise_events": noise_total,
            "cells": cells, "unevaluable": unevaluable,
            "wall_seconds": time.perf_counter() - started,
            "summary": summarise(cells, unevaluable)}


def widest_filter_gap(cells: Sequence[Dict]) -> Dict:
    """The largest mean MESR difference between two filters' own outputs, paired by recording.

    Each pair is averaged over the recordings on which both filters are scored, so a filter
    evaluable on fewer recordings is never compared across two cohorts. This is the
    same-corpus scale S V-E sets the slice-size spread against.
    """

    by_method: Dict[str, Dict[str, float]] = {}
    for c in cells:
        by_method.setdefault(c["method"], {})[c["recording"]] = float(c["mesr"])
    best: Dict = {"gap": float("nan")}
    methods = sorted(by_method)
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            shared = sorted(set(by_method[a]) & set(by_method[b]))
            if not shared:
                continue
            diff = float(np.mean([by_method[a][r] - by_method[b][r] for r in shared]))
            if not abs(diff) <= best["gap"]:              # also replaces the initial nan
                hi, lo = (a, b) if diff >= 0 else (b, a)
                best = {"gap": abs(diff), "a": hi, "b": lo, "n_recordings": len(shared)}
    return best


def summarise(cells: Sequence[Dict], unevaluable: Sequence[Dict]) -> Dict:
    """The counts the paper quotes: reversals on the filters' own outputs."""

    strict = [c for c in cells if c["label_class"] == "strict"]
    reversals = [c for c in strict if c["mesr_win"]]
    return {
        "cells": len(cells),
        "unevaluable": len(unevaluable),
        "recordings": len({c["recording"] for c in cells}),
        "eligible_strict": len(strict),
        "reversals": len(reversals),
        # Must be zero: the oracle bounds the filter's signal count by construction.
        "impossible": sum(1 for c in cells if c["label_class"] == "impossible"),
        "tie": sum(1 for c in cells if c["label_class"] == "tie"),
        "by_method": {
            m: {"cells": sum(1 for c in cells if c["method"] == m),
                "strict": sum(1 for c in strict if c["method"] == m),
                "reversals": sum(1 for c in reversals if c["method"] == m)}
            for m in sorted({c["method"] for c in cells})},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dnd21", choices=sorted(LOADERS))
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--hot-pixel-removal", action="store_true",
                        help="remove the busiest 0.1%% of occupied pixels from the input first")
    args = parser.parse_args()

    result = measure(args.dataset, max_recordings=args.max_recordings,
                     hot_pixel_removal=args.hot_pixel_removal)
    suffix = "_hotpixel" if args.hot_pixel_removal else ""
    out = args.out or OUT_DIR / f"native_oracle_{args.dataset}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float) + "\n")

    s = result["summary"]
    print(f"\n{args.dataset}: {s['recordings']} recordings, {s['cells']} evaluable cells, "
          f"{s['unevaluable']} unevaluable")
    print(f"strict (filter retains less signal at equal scored count): {s['eligible_strict']}")
    print(f"  of those, MESR prefers the filter: {s['reversals']}")
    print(f"ties {s['tie']}, impossible {s['impossible']} (must be 0)")
    print(f"\nWrote {out} ({result['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
