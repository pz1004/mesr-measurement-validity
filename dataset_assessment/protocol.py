"""The minimal reporting protocol this paper proposes.

Defects measured in Task 4, one fix each:

1. MESR depends on retention, and papers do not state it  -> report the CURVE, summarised by
   (r*, MESR*, area over r), never a bare scalar.
2. Absolute MESR moves with the event budget              -> always report Delta-over-Raw,
   which is stable across caps (measured 0.0004 vs 0.0151 for absolutes in the parent project).
3. Two MESRs can differ because their streams occupy different numbers of pixels rather than
   because one denoiser is better -> report the (ntss, ln, occupied_px) decomposition.

A fourth rule is procedural rather than computational: an optimum sitting at the smallest
*evaluable* retention is not evidence of a monotone metric, because short recordings cannot
be measured at aggressive retention at all. `summarise_curve` reports that flag explicitly
so the distinction cannot be lost.

WHY THERE IS NO sqrt(K) NORMALISATION HERE
------------------------------------------
An earlier draft of this protocol divided MESR by sqrt(W*H), on the premise that
`ln = K - sum(1-M/N)^n` scales with `K = W*H`. It does not. An empty pixel contributes
exactly `(1-M/N)^0 = 1`, so K cancels:

    ln = sum_occupied_px [1 - (1-M/N)^n]  <=  #occupied pixels.

ESR is therefore invariant to the declared sensor size - verified bit-identical at 346x260,
640x480 and 1280x720 for the same event stream, and to 1e-14 at 4096x4096 where float64
cancellation appears. Dividing by sqrt(K) would *manufacture* a resolution dependence that
the metric does not have, which is precisely the class of error this paper documents.
Cross-sensor differences come from occupancy, which is data, so the protocol reports the
decomposition and lets the reader see it. See `plan/dataset_assessment_notes.md`, F-LN.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def delta_over_raw(method_mesr: float, raw_mesr: float) -> float:
    """Improvement over the unfiltered stream.

    This was once described here as "the cap-invariant quantity". It is not.
    `cap_sensitivity.py` measures it: between a 10^6 and a 2x10^6 cap, with the evaluable
    grid identical for all 336 cells, 52% of them move by more than the 0.0092 the field
    ranks methods by, and the worst moves 35x it. Corpus *means* are steadier -- every
    method's mean moves less than 0.0092 across that pair -- but that is a measured property
    at those caps, not an invariance, and it degrades downward: at 5x10^5 the means move by
    up to 0.19 and one method's changes sign.
    """

    return float(method_mesr - raw_mesr)


#: A binary filter has one operating point, but the sweep only offers grid retentions, so
#: "at its native operating point" always means "at the nearest grid point". These bound how
#: far that substitution may reach. One grid step alone is the wrong yardstick at the low end:
#: a method natively keeping 1% is not measured at its operating point when scored at r=0.05,
#: which retains five times the events it actually keeps, even though the absolute gap is
#: under one step. The relative bound is what makes "native" mean native.
NATIVE_GRID_STEP = 0.05
NATIVE_REL_TOL = 0.5


def native_point_is_eligible(native_retention: float, grid_retention: float,
                             grid_step: float = NATIVE_GRID_STEP,
                             rel_tol: float = NATIVE_REL_TOL) -> bool:
    """May `grid_retention` stand in for a method's own operating point?

    One rule for every native-operating-point number in the paper. `label_quality` applied it
    from the start; Table IV did not, and 445 of its 2304 E-MLB cells (19.3%) fall outside it
    -- 261 of EvFlow's 384 alone, whose median native retention is 0.015 against a grid floor
    of 0.05. Applying it in one place is what stops the two tables meaning different things
    by the same phrase.
    """

    if not np.isfinite(native_retention) or not np.isfinite(grid_retention):
        return False
    gap = abs(grid_retention - native_retention)
    return bool(gap <= grid_step + 1e-9 and gap <= rel_tol * native_retention + 1e-12)


def decompose_esr(x: np.ndarray, y: np.ndarray, width: int, height: int) -> Dict:
    """Split one slice's ESR into its two factors plus the quantity that bounds `ln`.

    `ntss` is the probability that two distinct events of the slice share a pixel;
    `ln` is the expected number of pixels still occupied after removing M = floor(2N/3)
    events at random. `ESR = sqrt(ntss * ln)`, and `ln <= occupied_px` always.
    """

    n_events = len(x)
    if n_events < 2:
        return {"ntss": float("nan"), "ln": float("nan"), "occupied_px": 0,
                "n_events": int(n_events)}
    pixels = width * height
    counts = np.bincount(y.astype(np.int64) * width + x.astype(np.int64),
                         minlength=pixels).astype(np.float64)
    m = int(n_events * 2 / 3)
    eps = np.spacing(1)
    occupied = counts > 0
    ntss = (counts * (counts - 1)).sum() / (n_events + eps) / (n_events - 1 + eps)
    ln = float((1.0 - (1 - m / n_events) ** counts[occupied]).sum())
    return {"ntss": float(ntss), "ln": ln, "occupied_px": int(occupied.sum()),
            "n_events": int(n_events)}


def summarise_curve(curve: Sequence[Dict]) -> Dict:
    """Reduce a MESR@r curve to the numbers a table should carry."""

    usable = [c for c in curve if c.get("evaluable") and np.isfinite(c["mesr"])]
    if not usable:
        raise ValueError("no evaluable retention points; cannot summarise this curve")
    best = max(usable, key=lambda c: c["mesr"])
    rs = np.array([c["r"] for c in usable], dtype=float)
    ms = np.array([c["mesr"] for c in usable], dtype=float)
    order = np.argsort(rs)
    rs, ms = rs[order], ms[order]
    span = float(rs[-1] - rs[0])
    area = float(np.trapezoid(ms, rs) / span) if span > 0 else float(ms[0])
    return {
        "r_star": float(best["r"]),
        "mesr_star": float(best["mesr"]),
        "auc_over_r": area,
        "evaluable_range": (float(rs[0]), float(rs[-1])),
        "optimum_at_evaluable_floor": bool(best["r"] == rs[0]),
    }


def protocol_row(name: str, curve: Sequence[Dict], raw_mesr: float,
                 width: int, height: int,
                 decomposition: Optional[Dict] = None) -> Dict:
    """The canonical, self-describing row. Everything a reader needs, nothing implicit."""

    summary = summarise_curve(curve)
    row = {
        "method": name,
        "sensor": [width, height],
        **summary,
        "raw_mesr": float(raw_mesr),
        "delta_over_raw_at_r_star": delta_over_raw(summary["mesr_star"], raw_mesr),
    }
    if decomposition is not None:
        row.update({k: decomposition[k] for k in ("ntss", "ln", "occupied_px", "n_events")
                    if k in decomposition})
    return row


def rank_methods(rows: List[Dict], key: str) -> List[str]:
    """Method names ordered best-first by `key`."""

    return [r["method"] for r in sorted(rows, key=lambda r: -r[key])]


def rank_agreement(rows: List[Dict], key_a: str, key_b: str) -> Tuple[float, float]:
    """Spearman rho and Kendall tau between two ranking keys over the same methods."""

    from scipy.stats import kendalltau, spearmanr

    a = [r[key_a] for r in rows]
    b = [r[key_b] for r in rows]
    return float(spearmanr(a, b).statistic), float(kendalltau(a, b).statistic)
