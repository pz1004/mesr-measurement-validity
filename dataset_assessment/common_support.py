"""Is AUC_r comparable across the rows it ranks?

`protocol.summarise_curve` integrates each curve over **its own** evaluable range and divides
by that range's span. Two things follow that the paper did not state:

1. It is a span-normalised trapezoid, not "the mean MESR over the evaluable range". A mean
   over grid points weights every point alike; a trapezoid weights the interior twice as
   heavily as the endpoints, and the normalisation is by the span rather than by the count.
   With a ragged grid the two differ.
2. Recordings do not share a range. The measurable floor is set by how many complete
   30,000-event slices a recording yields, so a short recording's curve starts higher up.
   Averaging per-recording AUCs across a corpus therefore averages integrals taken over
   *different domains*, and the corpus mean the tables rank on is not one quantity.

Within a recording every method shares the floor -- `retain_mask` keeps the same count for
every ranking, so evaluability is a property of the recording, not of the method (asserted in
`tests/test_common_support.py`). The raggedness is entirely between recordings.

This module recomputes the ranking on common support two ways, because they trade off
differently and neither is obviously right:

* **common grid** -- keep every recording, integrate all of them over the widest range every
  recording supports. Loses the low-retention tail on corpora with a short recording in them.
* **common cohort** -- keep the full grid, restrict to the recordings that support it. Loses
  recordings.

If the ranking is a property of the methods it should survive both. `n(r)` is reported at
every grid point so a reader can see what each summary is averaging over.

Writes `results/common_support.json`. Run:

    python -m dataset_assessment.common_support
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .analyze import RESULTS, _mesr_at
from .denoisors import has_native_operating_point
from .protocol import rank_methods

DATASETS = ("dnd21", "dvsclean", "emlb", "dvsd22", "pure_ba")
OUT = RESULTS / "common_support.json"


def _points(row: Dict) -> List[Tuple[float, float]]:
    """Evaluable (r, MESR) pairs of one curve, sorted by r."""

    return sorted((float(c["r"]), float(c["mesr"])) for c in row.get("curve", [])
                  if c.get("evaluable") and np.isfinite(c["mesr"]))


def curves(dataset: str) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    """recording -> method -> evaluable curve points, oracles excluded."""

    payload = json.loads((RESULTS / f"benchmark_{dataset}.json").read_text())
    out: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    for record in payload["records"]:
        cell = {}
        for method, row in record["methods"].items():
            if "error" in row or row.get("is_oracle"):
                continue
            pts = _points(row)
            if pts:
                cell[method] = pts
        if cell:
            out[record["recording"]] = cell
    return out


def coverage(dataset: str) -> Dict:
    """`n(r)`: how many recordings and cells are evaluable at each grid point.

    A summary that silently averages over a shrinking sample as `r` falls looks like a
    property of the metric when it is a property of which recordings survived.
    """

    by_recording = curves(dataset)
    grid = sorted({r for cell in by_recording.values()
                   for pts in cell.values() for r, _ in pts})
    n_rec, n_cell = {}, {}
    for r in grid:
        n_rec[f"{r:.2f}"] = sum(any(r == p for p, _ in next(iter(cell.values())))
                                for cell in by_recording.values())
        n_cell[f"{r:.2f}"] = sum(1 for cell in by_recording.values()
                                 for pts in cell.values() if any(r == p for p, _ in pts))
    return {"grid": grid, "recordings_at_r": n_rec, "cells_at_r": n_cell,
            "n_recordings": len(by_recording)}


def _range_of(cell: Dict[str, List[Tuple[float, float]]]) -> Tuple[float, float]:
    """The evaluable range of a recording. Every method in it shares this."""

    pts = next(iter(cell.values()))
    return pts[0][0], pts[-1][0]


def auc(points: Sequence[Tuple[float, float]],
        lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    """Span-normalised trapezoid over [lo, hi], the quantity `AUC_r` actually is.

    Named for what it computes. `summarise_curve` calls the same thing `auc_over_r` and the
    paper called it a mean; it is neither a mean over grid points nor an unnormalised area.
    """

    kept = [(r, m) for r, m in points
            if (lo is None or r >= lo - 1e-9) and (hi is None or r <= hi + 1e-9)]
    if len(kept) < 2:
        return float(kept[0][1]) if kept else float("nan")
    rs = np.array([r for r, _ in kept])
    ms = np.array([m for _, m in kept])
    span = float(rs[-1] - rs[0])
    return float(np.trapezoid(ms, rs) / span) if span > 0 else float(ms[0])


def _native_of(dataset: str) -> Dict[str, Dict[str, float]]:
    """recording -> method -> MESR at the method's own operating point."""

    payload = json.loads((RESULTS / f"benchmark_{dataset}.json").read_text())
    out: Dict[str, Dict[str, float]] = {}
    for record in payload["records"]:
        cell = {}
        for method, row in record["methods"].items():
            if "error" in row or row.get("is_oracle") or "curve" not in row:
                continue
            if not has_native_operating_point(method):
                continue
            value = _mesr_at(row["curve"], row["native_retention"])
            if np.isfinite(value):
                cell[method] = float(value)
        if cell:
            out[record["recording"]] = cell
    return out


def _agreement(a: Dict[str, float], b: Dict[str, float]) -> Dict:
    """Rank agreement between two method -> score maps over the methods they share."""

    from scipy.stats import kendalltau, spearmanr

    shared = sorted(set(a) & set(b))
    if len(shared) < 3:
        return {"n_methods": len(shared), "spearman": float("nan"),
                "kendall": float("nan"), "differ": None}
    x = [a[m] for m in shared]
    y = [b[m] for m in shared]
    rank_a = rank_methods([{"method": m, "s": a[m]} for m in shared], "s")
    rank_b = rank_methods([{"method": m, "s": b[m]} for m in shared], "s")
    return {"n_methods": len(shared),
            "spearman": float(spearmanr(x, y).statistic),
            "kendall": float(kendalltau(x, y).statistic),
            "differ": rank_a != rank_b,
            "rank_a": rank_a, "rank_b": rank_b}


def recompute(dataset: str) -> Dict:
    """AUC_r three ways -- own range, common grid, common cohort -- against native."""

    by_recording = curves(dataset)
    native = _native_of(dataset)
    ranges = {rec: _range_of(cell) for rec, cell in by_recording.items()}
    common_lo = max(lo for lo, _ in ranges.values())
    common_hi = min(hi for _, hi in ranges.values())
    full_lo = min(lo for lo, _ in ranges.values())
    cohort = sorted(rec for rec, (lo, hi) in ranges.items()
                    if lo <= full_lo + 1e-9 and hi >= common_hi - 1e-9)

    def mean_auc(recordings: Sequence[str], lo, hi) -> Dict[str, float]:
        acc: Dict[str, List[float]] = {}
        for rec in recordings:
            for method, pts in by_recording[rec].items():
                value = auc(pts, lo, hi)
                if np.isfinite(value):
                    acc.setdefault(method, []).append(value)
        return {m: float(np.mean(v)) for m, v in acc.items()}

    all_recs = sorted(by_recording)
    variants = {
        "own_range": mean_auc(all_recs, None, None),
        "common_grid": mean_auc(all_recs, common_lo, common_hi),
        "common_cohort": mean_auc(cohort, full_lo, common_hi),
    }
    native_mean: Dict[str, List[float]] = {}
    for cell in native.values():
        for method, value in cell.items():
            native_mean.setdefault(method, []).append(value)
    native_mean_f = {m: float(np.mean(v)) for m, v in native_mean.items()}

    return {
        "dataset": dataset,
        "n_recordings": len(by_recording),
        "evaluable_range_per_recording": {
            "distinct": sorted({(lo, hi) for lo, hi in ranges.values()}),
            "common": [common_lo, common_hi],
            "widest": [full_lo, common_hi],
            "ragged": len({lo for lo, _ in ranges.values()}) > 1,
        },
        "common_cohort": {"n_recordings": len(cohort),
                          "share": len(cohort) / len(by_recording)},
        "coverage": coverage(dataset),
        "mean_auc": variants,
        "vs_native": {name: _agreement(v, native_mean_f) for name, v in variants.items()},
        # Does restricting the support change the ranking itself, independent of native?
        "own_vs_common_grid": _agreement(variants["own_range"], variants["common_grid"]),
        "own_vs_common_cohort": _agreement(variants["own_range"], variants["common_cohort"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    result = {d: recompute(d) for d in DATASETS
              if (RESULTS / f"benchmark_{d}.json").is_file()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float) + "\n")

    for dataset, r in result.items():
        er = r["evaluable_range_per_recording"]
        print(f"\n{dataset}: {r['n_recordings']} recordings, ranges "
              f"{'RAGGED' if er['ragged'] else 'uniform'} "
              f"{er['distinct']}")
        print(f"  common grid [{er['common'][0]:.2f}, {er['common'][1]:.2f}]; "
              f"common cohort {r['common_cohort']['n_recordings']}/{r['n_recordings']} "
              f"recordings over [{er['widest'][0]:.2f}, {er['widest'][1]:.2f}]")
        for name, a in r["vs_native"].items():
            print(f"  AUC_r ({name:13s}) vs native: rho={a['spearman']:+.3f} "
                  f"tau={a['kendall']:+.3f} differ={a['differ']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
