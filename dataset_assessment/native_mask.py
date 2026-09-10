"""Is the swept curve at r = native the released filter, or only something near it?

Appendix D said "at native it is exactly the filter". It is not, and the gap has two parts.

**The block quota.** `esr.retain_mask` keeps `k_b = round(n_b * r)` events in every
10,240-event block, the same count whatever the scores. A binary filter accepts `a_b` in
block `b`, and `a_b` varies across blocks -- that variation is the filter responding to scene
content, which is the whole point of it. Under the accepted-before-rejected ordering the two
retained sets differ by

    |F_native  symmetric-difference  F_adapted(r)| = sum_b |a_b - k_b(r)|,

so exact recovery needs equality in *every* block, not just in aggregate. Note the direction
this cuts: at a retention below the global native rate some blocks still have `k_b > a_b`, so
the sweep *reintroduces* rejected events there while discarding accepted ones elsewhere.

**The grid.** A native retention is a real number and the sweep offers multiples of 0.05, so
"at its native operating point" means "at the nearest grid point". `protocol` bounds how far
that may reach; this module reports how far it actually reaches.

Both are measured here rather than argued, per recording and per method.

Writes `results/native_mask.json`. Run:

    python -m dataset_assessment.native_mask --dataset emlb --recordings 24
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .denoisors import CLASSICAL, score_events
from .esr import BLOCK, retain_mask
from .protocol import native_point_is_eligible
from .run_benchmark import CAPPED_LOADERS, LOADERS

RESULTS = Path(__file__).resolve().parents[1] / "results"
RETENTIONS = [round(0.05 * i, 2) for i in range(1, 21)]


def block_accept_counts(scores: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """`a_b`: how many events the filter accepts in each block."""

    return np.array([int((scores[s:s + block] == 0.0).sum())
                     for s in range(0, len(scores), block)], dtype=np.int64)


def block_quota(n_events: int, retention: float, block: int = BLOCK) -> np.ndarray:
    """`k_b(r)`: what `retain_mask` keeps in each block, reproduced exactly."""

    return np.array([max(1, int(round(min(block, n_events - s) * retention)))
                     for s in range(0, n_events, block)], dtype=np.int64)


def compare_masks(scores: np.ndarray, retention: float) -> Dict:
    """Symmetric difference between the filter's own mask and the swept mask at `retention`."""

    accepted = scores == 0.0
    swept = retain_mask(scores, retention)
    a_b = block_accept_counts(scores)
    k_b = block_quota(len(scores), retention)
    symmetric = int(np.logical_xor(accepted, swept).sum())
    return {
        "n_events": int(len(scores)),
        "n_blocks": int(len(a_b)),
        "native_kept": int(accepted.sum()),
        "swept_kept": int(swept.sum()),
        "symmetric_difference": symmetric,
        "symmetric_difference_share": symmetric / max(1, int(accepted.sum())),
        "predicted_by_block_identity": int(np.abs(a_b - k_b).sum()),
        # Blocks where the quota exceeds what the filter accepted, so the sweep puts
        # rejected events back even though the global retention is at or below native.
        "blocks_reintroducing_rejects": int((k_b > a_b).sum()),
        "block_accept_share": {"mean": float(a_b.mean() / BLOCK),
                               "sd": float(a_b.std() / BLOCK),
                               "min": float(a_b.min() / BLOCK),
                               "max": float(a_b.max() / BLOCK)},
    }


def measure(dataset: str, recordings: int = 24, max_events: int = 1_000_000,
            methods: Sequence[str] = CLASSICAL) -> Dict:
    """Per (recording, method): the grid gap, the block gap, and whether it is eligible."""

    loader = LOADERS[dataset]
    kwargs = {"max_events": max_events} if dataset in CAPPED_LOADERS else {}
    cells: List[Dict] = []
    started = time.perf_counter()
    for index, rec in enumerate(loader(**kwargs)):
        if index >= recordings:
            break
        events = rec.events[:max_events]
        for method in methods:
            scores = score_events(method, rec.__class__(
                events, None if rec.labels is None else rec.labels[:len(events)],
                rec.width, rec.height, rec.name, rec.synthetic))
            native = float((scores == 0.0).mean())
            nearest = min(RETENTIONS, key=lambda r: abs(r - native))
            row = {"recording": rec.name, "method": method,
                   "native_retention": native,
                   "nearest_grid_retention": nearest,
                   "grid_gap": abs(nearest - native),
                   "eligible": native_point_is_eligible(native, nearest)}
            # At the *exact* native rate, so the grid is held out and only the block quota
            # can move the mask. This is the part no choice of grid could repair.
            row["at_exact_native"] = compare_masks(scores, native)
            row["at_nearest_grid"] = compare_masks(scores, nearest)
            cells.append(row)
    exact = np.array([c["at_exact_native"]["symmetric_difference_share"] for c in cells])
    grid = np.array([c["at_nearest_grid"]["symmetric_difference_share"] for c in cells])
    return {
        "dataset": dataset,
        "recordings": len({c["recording"] for c in cells}),
        "cells": cells,
        "wall_seconds": time.perf_counter() - started,
        "summary": {
            "cells": len(cells),
            "eligible": int(sum(c["eligible"] for c in cells)),
            # The headline: even with the grid held out, the swept mask is not the filter.
            "at_exact_native": {"median_share": float(np.median(exact)),
                                "max_share": float(exact.max()),
                                "exact_recoveries": int((exact == 0).sum())},
            "at_nearest_grid": {"median_share": float(np.median(grid)),
                                "max_share": float(grid.max())},
            "identity_holds": bool(all(
                c["at_exact_native"]["symmetric_difference"]
                == c["at_exact_native"]["predicted_by_block_identity"] for c in cells)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="emlb", choices=sorted(LOADERS))
    parser.add_argument("--recordings", type=int, default=24)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = measure(args.dataset, recordings=args.recordings)
    out = args.out or RESULTS / f"native_mask_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float) + "\n")

    s = result["summary"]
    print(f"{args.dataset}: {result['recordings']} recordings, {s['cells']} cells, "
          f"{s['eligible']} eligible at the grid")
    print(f"sum_b |a_b - k_b| predicts the symmetric difference exactly: "
          f"{s['identity_holds']}")
    e, g = s["at_exact_native"], s["at_nearest_grid"]
    print(f"at the EXACT native rate: median {e['median_share']:.1%} of the retained set "
          f"differs, worst {e['max_share']:.1%}; exact recoveries {e['exact_recoveries']}"
          f"/{s['cells']}")
    print(f"at the nearest grid point: median {g['median_share']:.1%}, "
          f"worst {g['max_share']:.1%}")
    print(f"\nWrote {out} ({result['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
