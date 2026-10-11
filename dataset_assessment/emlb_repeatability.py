"""MESR repeatability: the three repetitions E-MLB ships for every (scene, ND) cell.

Every other experiment here varies the *selection* at fixed input, or one declared
convention. This one varies neither. E-MLB records each (scene, ND) cell three times under
nominally identical conditions and ships all three; the rest of this study reads the first.
Scoring the other two gives what no published MESR table reports: the spread of the statistic
under repeated acquisition of the same scene at the same noise level, with no filter, no
retention argument and no convention change anywhere in the path.

That is a Type A repeatability component in the sense of the GUM. It is *not* an uncertainty
for the quality MESR stands in for - without an operational measurand there is nothing for
that uncertainty to be of - but it is an uncertainty for the statistic itself, and it is the
one number that says whether a published gap between two methods is larger than the noise of
acquiring the same scene twice.

Per (part, scene, ND) cell, capped at 10^6 events exactly as `run_benchmark` caps it:

1. MESR of the unfiltered stream, once per repetition.
2. The cell's spread, `max - min` over the repetitions present, and its sample standard
   deviation. A cell with fewer than two readable repetitions is reported and not scored.
3. The reference the spreads are read against is the EDformer-EDmamba difference as
   published on E-MLB, 0.0092, which is what a table asks the reader to resolve.

Writes `results/emlb_repeatability.json`. Run:

    python -m dataset_assessment.emlb_repeatability --workers 16
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

from .esr import mesr
from .readers import iter_emlb

RESULTS = Path(__file__).resolve().parents[1] / "results"
MAX_EVENTS = 1_000_000
REPS = (1, 2, 3)

#: EDformer vs. EDmamba on E-MLB, as published. The gap a reader is asked to resolve.
PUBLISHED_GAP = 0.0092

#: Draws of which repetition each complete cell contributes to the corpus mean, and their
#: seed. The paper quotes the resulting sd and central 95% width; both are generated here.
AGGREGATE_DRAWS = 20_000
AGGREGATE_SEED = 20260726


def cell_of(recording: str) -> str:
    """The (part, scene, ND) cell a repetition belongs to: its name without the `-<rep>`."""

    return recording.rsplit("-", 1)[0]


def nd_level_of(cell: str) -> str:
    """The neutral-density level of a cell, e.g. `ND00`, from its name's last field."""

    return cell.rsplit("-", 1)[-1]


def lighting_of(cell: str) -> str:
    """The lighting set of a cell, `D-END` (daytime) or `N-END` (night), from its path."""

    return cell.split("/")[1]


def repetition_draws(cells: Sequence[Dict], draws: int = AGGREGATE_DRAWS,
                     seed: int = AGGREGATE_SEED) -> Dict:
    """How much the corpus mean moves with which repetition each cell contributes.

    Each draw picks one repetition per cell independently and averages over cells. Only
    cells with every repetition present enter, so all draws average the same cells.
    """

    full = max((len(c["mesr"]) for c in cells), default=0)
    readings = np.array([c["mesr"] for c in cells if len(c["mesr"]) == full], dtype=float)
    if readings.size == 0:
        return {"draws": 0, "cells": 0}
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, full, size=(draws, readings.shape[0]))
    means = readings[np.arange(readings.shape[0]), picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"draws": draws, "seed": seed, "cells": int(readings.shape[0]),
            "first_repetition_mean": float(readings[:, 0].mean()),
            "sd": float(means.std(ddof=1)),
            "central_95_width": float(high - low)}


def _one(rec) -> Dict:
    """MESR of one unfiltered repetition."""

    return {"recording": rec.name,
            "cell": cell_of(rec.name),
            "events": int(len(rec.events)),
            "mesr": float(mesr(rec.x, rec.y, rec.width, rec.height))}


def summarise(records: Sequence[Dict], gap: float = PUBLISHED_GAP) -> Dict:
    """Per-cell spread over repetitions, and how it compares with the published gap."""

    by_cell: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        if np.isfinite(record["mesr"]):
            by_cell[record["cell"]].append(record)

    cells, incomplete = [], []
    for cell, rows in sorted(by_cell.items()):
        values = np.array([r["mesr"] for r in rows], dtype=float)
        if len(values) < 2:
            incomplete.append({"cell": cell, "repetitions": len(values)})
            continue
        cells.append({"cell": cell,
                      "repetitions": len(values),
                      "mesr": [float(v) for v in values],
                      "mean": float(values.mean()),
                      "spread": float(values.max() - values.min()),
                      "sd": float(values.std(ddof=1))})

    spreads = np.array([c["spread"] for c in cells], dtype=float)
    sds = np.array([c["sd"] for c in cells], dtype=float)
    # Pooled within-cell sd: the repeatability standard deviation, sqrt of the mean variance.
    pooled = float(np.sqrt(np.mean(sds ** 2))) if len(sds) else float("nan")
    worst = max(cells, key=lambda c: c["spread"]) if cells else None

    def median_spread_by(key: Callable[[str], str]) -> Dict[str, float]:
        groups: Dict[str, List[float]] = defaultdict(list)
        for c, spread in zip(cells, spreads):
            groups[key(str(c["cell"]))].append(float(spread))
        return {name: float(np.median(values)) for name, values in sorted(groups.items())}

    return {"cells": len(cells),
            "incomplete_cells": incomplete,
            "repetition_readings": int(len(records)),
            "published_gap": gap,
            "spread": {"mean": float(spreads.mean()), "median": float(np.median(spreads)),
                       "min": float(spreads.min()), "max": float(spreads.max())},
            "repeatability_sd": pooled,
            "cells_exceeding_gap": int((spreads > gap).sum()),
            "fraction_exceeding_gap": float((spreads > gap).mean()),
            "median_spread_in_gaps": float(np.median(spreads) / gap),
            "median_spread_by_nd_level": median_spread_by(nd_level_of),
            "median_spread_by_lighting": median_spread_by(lighting_of),
            "repetition_draws": repetition_draws(cells),
            "worst_cell": worst}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--scene-stride", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    started = time.perf_counter()
    recordings = iter_emlb(max_events=MAX_EVENTS, reps=REPS, scene_stride=args.scene_stride)
    records: List[Dict] = []
    with Pool(args.workers) as pool:
        for index, record in enumerate(pool.imap(_one, recordings, chunksize=1)):
            records.append(record)
            if index % 48 == 0:
                print(f"[{index + 1}] {record['recording']} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)

    summary = summarise(records)
    result = {"dataset": "emlb", "max_events": MAX_EVENTS, "repetitions": list(REPS),
              "records": records, "wall_seconds": time.perf_counter() - started,
              "summary": summary}
    out = args.out or RESULTS / "emlb_repeatability.json"
    out.write_text(json.dumps(result, indent=2, default=float) + "\n")

    s = summary["spread"]
    print(f"\n{summary['cells']} cells x {len(REPS)} repetitions "
          f"({summary['repetition_readings']} readings)")
    print(f"  spread over repetitions: mean {s['mean']:.4f}  median {s['median']:.4f}  "
          f"max {s['max']:.4f}")
    print(f"  repeatability sd (pooled within cell): {summary['repeatability_sd']:.4f}")
    print(f"  cells whose spread exceeds the published {PUBLISHED_GAP} gap: "
          f"{summary['cells_exceeding_gap']}/{summary['cells']} "
          f"({100 * summary['fraction_exceeding_gap']:.1f}%)")
    print(f"  median spread is {summary['median_spread_in_gaps']:.1f}x that gap")
    draws = summary["repetition_draws"]
    if draws.get("draws"):
        print(f"  corpus mean over {draws['draws']} draws of the repetition per cell: "
              f"sd {draws['sd']:.4f}, central 95% width {draws['central_95_width']:.4f}")
    if summary["worst_cell"]:
        w = summary["worst_cell"]
        print(f"  worst cell: {w['cell']} spread {w['spread']:.4f}")
    print(f"\nWrote {out} ({result['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
