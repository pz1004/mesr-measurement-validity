"""E-MLB on the filters' own outputs: protocol item 1 applied, not its diagnostic form.

Table IV read each filter at the grid point nearest its native retention, through the block
quota every swept point uses. `native_mask` showed what that costs: the quota recovers the
filter's retained count but not its retained set, in none of 168 cells. This module scores
what the filter actually keeps.

Per recording, capped at 10^6 events exactly as `run_benchmark` caps it:

1. `raw` is the unfiltered input's MESR, the reference every Delta-over-Raw is taken against.
2. For each classical filter, `F` is its native mask and `F_s` the part MESR reads, the first
   ``m = SLICE * (|F| // SLICE)`` accepted events (`native_oracle.scored_prefix`). A cell with
   ``m == 0`` is unevaluable and reported, never refilled at a nearby retention.
3. The **matched control** draws, from each input block, exactly as many events as `F_s`
   does, uniformly at random within the block. It holds the same `m` events on the same
   blocks, so filter and control differ only in *which* events they keep. It reads no label
   and no score, and is seeded per (recording, method).

The contrast `filter_minus_null` is the one a nonselective control exists for: what choosing
the events is worth, at the filter's own count and over the filter's own time span.

Writes `results/native_emlb.json`. Run:

    python -m dataset_assessment.native_emlb --workers 16
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .analyze import cluster_bootstrap_delta_ci, scene_of
from .denoisors import CLASSICAL, score_events
from .esr import BLOCK, mesr
from .native_oracle import block_counts, scored_prefix
from .readers import iter_emlb

RESULTS = Path(__file__).resolve().parents[1] / "results"
MAX_EVENTS = 1_000_000
CONTROL_SEED = 20260917


def matched_random_mask(counts: np.ndarray, n_events: int, block: int = BLOCK,
                        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """`counts[b]` events drawn uniformly without replacement from each block `b`."""

    rng = rng or np.random.default_rng(CONTROL_SEED)
    keep = np.zeros(n_events, dtype=bool)
    for b, k in enumerate(counts):
        if k == 0:
            continue
        start = b * block
        size = min(block, n_events - start)
        keep[start + rng.choice(size, size=int(k), replace=False)] = True
    return keep


def _seed(recording: str, method: str) -> List[int]:
    return [CONTROL_SEED, zlib.crc32(recording.encode()), zlib.crc32(method.encode())]


def measure_recording(rec, methods: Sequence[str] = CLASSICAL,
                      scorer: Callable = score_events) -> Dict:
    """Every filter's native output and its matched control on one capped recording."""

    raw = float(mesr(rec.x, rec.y, rec.width, rec.height))
    n = len(rec.events)
    cells: List[Dict] = []
    unevaluable: List[Dict] = []
    for method in methods:
        accepted = scorer(method, rec) == 0.0
        native = float(accepted.mean())
        prefix = scored_prefix(accepted)
        m = int(prefix.sum())
        if m == 0:
            unevaluable.append({"recording": rec.name, "method": method,
                                "native_retention": native,
                                "native_kept": int(accepted.sum()),
                                "reason": "native output is shorter than one complete slice"})
            continue
        control = matched_random_mask(block_counts(prefix), n,
                                      rng=np.random.default_rng(_seed(rec.name, method)))
        if int(control.sum()) != m:
            raise AssertionError(f"{rec.name} {method}: control holds {int(control.sum())} "
                                 f"events against the prefix's {m}")
        score = float(mesr(rec.x[prefix], rec.y[prefix], rec.width, rec.height))
        null = float(mesr(rec.x[control], rec.y[control], rec.width, rec.height))
        cells.append({"recording": rec.name, "method": method,
                      "native_retention": native, "native_kept": int(accepted.sum()),
                      "scored": m, "mesr": score, "null_mesr": null,
                      "delta_over_raw": score - raw, "null_delta_over_raw": null - raw,
                      "filter_minus_null": score - null})
    return {"recording": rec.name, "events": n, "raw_mesr": raw,
            "cells": cells, "unevaluable": unevaluable}


def _one(rec, methods: Sequence[str] = CLASSICAL) -> Dict:
    return measure_recording(rec, methods)


def merge(existing: Sequence[Dict], added: Sequence[Dict]) -> List[Dict]:
    """Append `added`'s cells to `existing`'s records, leaving every existing cell untouched.

    Used to add a method (EDformer) without re-running the six filters: the classical cells
    are carried over verbatim, so they cannot move. Recordings must match one to one.
    """

    by_name = {r["recording"]: r for r in added}
    if set(by_name) != {r["recording"] for r in existing}:
        raise ValueError("merge needs the same recordings on both sides")
    merged: List[Dict] = []
    for record in existing:
        extra = by_name[record["recording"]]
        if extra["raw_mesr"] != record["raw_mesr"] or extra["events"] != record["events"]:
            raise ValueError(f"{record['recording']}: inputs differ between runs")
        clash = ({c["method"] for c in record["cells"] + record["unevaluable"]}
                 & {c["method"] for c in extra["cells"] + extra["unevaluable"]})
        if clash:
            raise ValueError(f"{record['recording']}: {sorted(clash)} already present")
        merged.append({**record, "cells": record["cells"] + extra["cells"],
                       "unevaluable": record["unevaluable"] + extra["unevaluable"]})
    return merged


def _marginal(cells: Sequence[Dict], key: str) -> Dict:
    ci = cluster_bootstrap_delta_ci([c[key] for c in cells],
                                    [scene_of(c["recording"]) for c in cells])
    return {"mean": ci["mean"], "lo": ci["lo"], "hi": ci["hi"],
            "n_scenes": ci.get("n_clusters"),
            "excludes_zero": bool(ci.get("excludes_zero", False))}


def summarise(records: Sequence[Dict]) -> Dict:
    """Per filter: how often it is evaluable, where it operates, and both contrasts."""

    cells = [c for r in records for c in r["cells"]]
    unevaluable = [u for r in records for u in r["unevaluable"]]
    out: Dict[str, Dict] = {}
    for method in sorted({c["method"] for c in cells} | {u["method"] for u in unevaluable}):
        mine = [c for c in cells if c["method"] == method]
        lost = [u for u in unevaluable if u["method"] == method]
        row = {"evaluable": len(mine), "unevaluable": len(lost)}
        if mine:
            row.update({
                "native_retention_mean": float(np.mean([c["native_retention"] for c in mine])),
                "delta_over_raw": _marginal(mine, "delta_over_raw"),
                "null_delta_over_raw": _marginal(mine, "null_delta_over_raw"),
                "filter_minus_null": _marginal(mine, "filter_minus_null"),
                "recordings_filter_above_null":
                    int(sum(c["filter_minus_null"] > 0 for c in mine)),
            })
        if lost:
            row["unevaluable_native_retention_mean"] = float(
                np.mean([u["native_retention"] for u in lost]))
        out[method] = row
    return {"recordings": len(records), "cells": len(cells),
            "unevaluable": len(unevaluable), "by_method": out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=list(CLASSICAL))
    parser.add_argument("--merge", action="store_true",
                        help="add --methods to the existing native_emlb.json instead of "
                             "replacing it; existing cells are carried over verbatim")
    args = parser.parse_args()

    started = time.perf_counter()
    recordings = iter_emlb(max_events=MAX_EVENTS)
    records: List[Dict] = []
    with Pool(args.workers) as pool:
        stream = (r for i, r in enumerate(recordings)
                  if args.max_recordings is None or i < args.max_recordings)
        worker = partial(_one, methods=tuple(args.methods))
        for index, record in enumerate(pool.imap(worker, stream, chunksize=1)):
            records.append(record)
            if index % 16 == 0:
                print(f"[{index + 1}] {record['recording']} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
    out = args.out or RESULTS / "native_emlb.json"
    if args.merge:
        records = merge(json.loads(out.read_text())["records"], records)

    result = {"dataset": "emlb", "max_events": MAX_EVENTS, "control_seed": CONTROL_SEED,
              "records": records, "wall_seconds": time.perf_counter() - started,
              "summary": summarise(records)}
    out.write_text(json.dumps(result, indent=2, default=float) + "\n")

    print(f"\n{result['summary']['recordings']} recordings, {result['summary']['cells']} "
          f"evaluable cells, {result['summary']['unevaluable']} unevaluable")
    for method, row in result["summary"]["by_method"].items():
        if "delta_over_raw" not in row:
            print(f"  {method:7s} evaluable 0")
            continue
        d, s = row["delta_over_raw"], row["filter_minus_null"]
        print(f"  {method:7s} n={row['evaluable']:3d}  r={row['native_retention_mean']:.2f}  "
              f"delta {d['mean']:+.4f} [{d['lo']:+.3f}, {d['hi']:+.3f}]  "
              f"vs null {s['mean']:+.4f} [{s['lo']:+.3f}, {s['hi']:+.3f}]")
    print(f"\nWrote {out} ({result['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
