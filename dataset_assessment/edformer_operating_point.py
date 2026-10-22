"""The operating-point sweep of section V-D, with EDformer's own scores in place of synthetic ones.

`measure_metric_deps` holds a synthetic score vector fixed (unit-variance Gaussians shifted for
noise so the ROC AUC is 0.93) and sweeps the block quota over retention on every whole DND21
recording; the spread of MESR across that sweep is the operating point's effect under fixed
event scores. A reviewer can fairly ask whether a real denoiser's scores behave the same way.

This module repeats that sweep exactly -- same whole recordings, same retention grid, same
`esr.mesr_curve` and block quota -- with one change: the score vector is EDformer's released
per-event sigmoid, read from `edformer.py`'s whole-recording cache (cap 0). The sigmoid is a
noise probability and `retain_mask` keeps the lowest scores, so it is used as it is; events
past the last full 4096-event block, which the release never scores, carry `TAIL_SCORE` and
are removed first. Each recording also reports the scores' ROC AUC over the events EDformer
scores, and the retention its released threshold keeps.

Torch-free: it reads the cache. Run `edformer.py --corpus dnd21 --cap 0` first, then

    python -m dataset_assessment.edformer_operating_point
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .edformer import TAIL_SCORE, cache_path, keep_mask
from .esr import mesr_curve
from .measure_metric_deps import RETENTIONS, _spread_summary
from .readers import iter_dnd21

OUT = Path(__file__).resolve().parents[1] / "results" / "edformer_operating_point.json"
WHOLE = 0                      # edformer.py's cap for whole recordings


def scores_for(recording) -> np.ndarray:
    """The cached whole-recording sigmoid, refused unless it has one score per event."""

    path = cache_path(recording.name, WHOLE)
    if not path.exists():
        raise ValueError(f"no cached EDformer scores at {path}; run "
                         "`python -m dataset_assessment.edformer --corpus dnd21 --cap 0`")
    scores = np.load(path)
    if len(scores) != len(recording.events):
        raise ValueError(f"{recording.name}: cached {len(scores)} scores for "
                         f"{len(recording.events)} events")
    return scores


def sweep_row(name: str, scores: np.ndarray, labels: np.ndarray, x: np.ndarray, y: np.ndarray,
              width: int, height: int,
              retentions: Optional[Sequence[float]] = None) -> Dict:
    """One recording: ROC AUC, the released retention, and the MESR spread over the sweep.

    `retentions` defaults to `measure_metric_deps.RETENTIONS`, the synthetic sweep's grid."""

    from sklearn.metrics import roc_auc_score

    grid = [float(r) for r in (RETENTIONS if retentions is None else retentions)]
    scored = scores != TAIL_SCORE
    curve = mesr_curve(scores, x, y, width, height, grid)
    usable = [c for c in curve if c["evaluable"]]
    if not usable:
        raise ValueError(f"{name}: no evaluable retention")
    lo = min(usable, key=lambda c: c["mesr"])
    hi = max(usable, key=lambda c: c["mesr"])
    return {
        "recording": name,
        "events": int(len(scores)),
        "scored": int(scored.sum()),
        # Noise is the positive class, as for the synthetic scores (`rank_by_label`).
        "roc_auc": float(roc_auc_score(labels[scored], scores[scored])),
        "released_retention": float(keep_mask(scores).mean()),
        "min_mesr": float(lo["mesr"]), "max_mesr": float(hi["mesr"]),
        "spread": float(hi["mesr"] - lo["mesr"]),
        "argmin_r": float(lo["r"]), "argmax_r": float(hi["r"]),
        "evaluable_retentions": len(usable),
        "curve": curve,
    }


def measure() -> Dict:
    rows: List[Dict] = []
    for rec in iter_dnd21():
        if rec.labels is None:
            raise ValueError(f"{rec.name} has no labels")
        rows.append(sweep_row(rec.name, scores_for(rec), rec.labels, rec.x, rec.y,
                              rec.width, rec.height))
        r = rows[-1]
        print(f"{rec.name}: AUC {r['roc_auc']:.4f}, released r {r['released_retention']:.3f}, "
              f"spread {r['spread']:.4f}", flush=True)

    spreads = [r["spread"] for r in rows]
    by_base = {}
    for base in ("hotel-bar", "driving"):
        sub = [r["spread"] for r in rows if r["recording"].endswith(base)]
        by_base[base] = {"n": len(sub), "mean": float(np.mean(sub)),
                         "min": float(np.min(sub)), "max": float(np.max(sub))}
    aucs = [r["roc_auc"] for r in rows]
    return {
        "what": "MESR spread over the retention sweep with EDformer's own per-event scores, "
                "on whole DND21 recordings (the sweep of measure_metric_deps, scores swapped)",
        "retentions": [float(r) for r in RETENTIONS],
        "rows": rows,
        "spread_over_recordings": _spread_summary(spreads),
        "spread_by_base": by_base,
        "roc_auc": {"mean": float(np.mean(aucs)), "min": float(np.min(aucs)),
                    "max": float(np.max(aucs))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    result = measure()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float) + "\n")
    s = result["spread_over_recordings"]
    print(f"\nspread over {len(result['rows'])} recordings: mean {s['mean']:.4f} "
          f"[{s['lo']:.4f}, {s['hi']:.4f}]; ROC AUC mean "
          f"{result['roc_auc']['mean']:.4f}\nWrote {args.out}")


if __name__ == "__main__":
    main()
