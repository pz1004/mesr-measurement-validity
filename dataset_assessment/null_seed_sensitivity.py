"""How much of the specificity verdict is the particular draw the null made.

`random_null` is seeded (`denoisors.RANDOM_NULL_SEED`), so every number this paper quotes for
it comes from one realisation per recording. The scene bootstrap behind those numbers resamples
recordings; it does not resample the selector. Those are different sources of variation, and a
claim that the null's gain exceeds "sampling variation" has to say which one it cleared.

This module scores the null under ten draws on the two corpora that carry the specificity and
blank-sample verdicts, and reports the across-seed spread of the quantities the main text
states: the count of retentions whose mean Delta-over-Raw is positive, the mean at each
retention, and -- on Pure_BA -- the best filter's lead over the control at r = 0.60. The
filters are deterministic, so their side of that lead is read from the published run and only
the control varies.

Inputs are the per-seed grids written by

    for S in ...; do python -m dataset_assessment.run_benchmark --dataset dvsd22 \
        --methods random_null --null-seed $S --out results/null_seeds/dvsd22_s$S.json; done

Writes `results/null_seed_sensitivity.json`. Run:

    python -m dataset_assessment.null_seed_sensitivity
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .analyze import scene_of

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEED_DIR = RESULTS / "null_seeds"
NULL = "random_null"

#: The two corpora whose unfiltered streams carry a positive null response as released.
CORPORA = ("dvsd22", "pure_ba")

#: The retention the blank-sample comparison is read at: the smallest all 26 Pure_BA
#: recordings support. Fixed in advance, as Sec. V-C fixes it.
BLANK_R = 0.60


def _seed_of(path: Path) -> int:
    match = re.search(r"_s(\d+)\.json$", path.name)
    if match is None:
        raise ValueError(f"cannot read a seed from {path.name}")
    return int(match.group(1))


def profile(path: Path, method: str = NULL) -> Dict:
    """Mean Delta-over-Raw at each grid point, by the definition `analyze` uses.

    Delta is `curve mesr - raw_mesr` per recording, averaged over the recordings for which
    that grid point is evaluable, so the cohort can differ between retentions. r = 1.0 is the
    identity and is excluded from every count: it contributes a structural zero.
    """

    payload = json.loads(path.read_text())
    grid = [float(r) for r in payload["retentions"]]
    per_r: Dict[float, List[float]] = {r: [] for r in grid}
    for record in payload["records"]:
        row = record["methods"].get(method)
        if row is None or "error" in row or "raw_mesr" not in row:
            continue
        base = row["raw_mesr"]
        for point in row["curve"]:
            r = float(point["r"])
            if point.get("evaluable") and np.isfinite(point["mesr"]) and r in per_r:
                per_r[r].append(float(point["mesr"] - base))

    points = [{"r": r, "mean": float(np.mean(per_r[r])), "n": len(per_r[r])}
              for r in grid if per_r[r]]
    below_one = [p for p in points if p["r"] < 1.0]
    return {"seed": _seed_of(path),
            "points": points,
            "n_retentions_below_one": len(below_one),
            "n_retentions_with_positive_mean": sum(p["mean"] > 0 for p in below_one),
            "best_fixed": max(below_one, key=lambda p: p["mean"]) if below_one else None}


def null_at(path: Path, retention: float, method: str = NULL) -> Dict[str, float]:
    """Each recording's null Delta-over-Raw at one retention, keyed by recording."""

    payload = json.loads(path.read_text())
    out: Dict[str, float] = {}
    for record in payload["records"]:
        row = record["methods"].get(method)
        if row is None or "error" in row or "raw_mesr" not in row:
            continue
        for point in row["curve"]:
            if (abs(float(point["r"]) - retention) < 1e-9 and point.get("evaluable")
                    and np.isfinite(point["mesr"])):
                out[record["recording"]] = float(point["mesr"] - row["raw_mesr"])
    return out


def filter_deltas_at(dataset: str, retention: float,
                     exclude: Sequence[str] = ("raw", NULL)) -> Dict[str, Dict[str, float]]:
    """Each real filter's Delta-over-Raw at one retention, from the published run.

    The filters do not depend on the null's seed, so this side of the blank-sample lead is
    read once rather than recomputed per draw.
    """

    payload = json.loads((RESULTS / f"benchmark_{dataset}.json").read_text())
    out: Dict[str, Dict[str, float]] = {}
    for record in payload["records"]:
        for method, row in record["methods"].items():
            if method in exclude or "error" in row or "raw_mesr" not in row:
                continue
            for point in row["curve"]:
                if (abs(float(point["r"]) - retention) < 1e-9 and point.get("evaluable")
                        and np.isfinite(point["mesr"])):
                    out.setdefault(method, {})[record["recording"]] = \
                        float(point["mesr"] - row["raw_mesr"])
    return out


def blank_lead(dataset: str, paths: Sequence[Path], retention: float = BLANK_R) -> Dict:
    """The best filter's paired lead over the control, one value per draw.

    Paired on the recordings where the filter and that draw's control are both evaluable, which
    is the cohort Sec. V-C reports the lead on.
    """

    filters = filter_deltas_at(dataset, retention)
    rows = []
    for path in sorted(paths, key=_seed_of):
        control = null_at(path, retention)
        leads = {}
        for method, deltas in filters.items():
            shared = sorted(set(deltas) & set(control))
            if shared:
                leads[method] = float(np.mean([deltas[r] - control[r] for r in shared]))
        if not leads:
            continue
        best = max(leads, key=lambda m: leads[m])
        rows.append({"seed": _seed_of(path), "best_filter": best,
                     "lead": leads[best], "n": len(set(filters[best]) & set(control)),
                     "all_leads": leads})
    return {"retention": retention, "per_seed": rows,
            "spread": _spread([r["lead"] for r in rows]),
            "best_filter_is_always": sorted({r["best_filter"] for r in rows})}


def _spread(values: Sequence[float]) -> Dict:
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    return {"n": int(array.size), "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "min": float(array.min()), "max": float(array.max()),
            "range": float(array.max() - array.min())}


def analyse(dataset: str) -> Optional[Dict]:
    paths = sorted(SEED_DIR.glob(f"{dataset}_s*.json"), key=_seed_of)
    if not paths:
        return None
    profiles = [profile(p) for p in paths]
    counts = [p["n_retentions_with_positive_mean"] for p in profiles]
    grid = sorted({p["r"] for prof in profiles for p in prof["points"] if p["r"] < 1.0})

    by_r = []
    for r in grid:
        means = [next((p["mean"] for p in prof["points"] if abs(p["r"] - r) < 1e-9), None)
                 for prof in profiles]
        means = [m for m in means if m is not None]
        by_r.append({"r": r, **_spread(means),
                     "seeds_positive": int(sum(m > 0 for m in means))})

    return {"seeds": [p["seed"] for p in profiles],
            "n_retentions_below_one": profiles[0]["n_retentions_below_one"],
            "positive_mean_count": {"per_seed": counts, **_spread(counts)},
            "by_retention": by_r,
            "best_fixed_per_seed": [{"seed": p["seed"], **p["best_fixed"]}
                                    for p in profiles if p["best_fixed"]],
            "blank_lead": blank_lead(dataset, paths) if dataset == "pure_ba" else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = {"null": NULL, "seed_dir": str(SEED_DIR.relative_to(RESULTS.parent)),
              "corpora": {}}
    for dataset in CORPORA:
        block = analyse(dataset)
        if block is None:
            print(f"{dataset}: no per-seed grids under {SEED_DIR}")
            continue
        result["corpora"][dataset] = block
        count = block["positive_mean_count"]
        print(f"\n{dataset}: {len(block['seeds'])} draws, "
              f"{block['n_retentions_below_one']} retentions below 1")
        print(f"  retentions with positive mean Delta: {count['min']:.0f}-{count['max']:.0f} "
              f"(mean {count['mean']:.1f}, sd {count['sd']:.2f})  per seed {count['per_seed']}")
        widest = max(block["by_retention"], key=lambda b: b["range"])
        print(f"  widest across-seed range of the mean: {widest['range']:.4f} at "
              f"r = {widest['r']:.2f} (sd {widest['sd']:.4f})")
        print("  mean Delta by r: " + "  ".join(
            f"{b['r']:.2f}:{b['mean']:+.3f}+-{b['sd']:.3f}" for b in block["by_retention"]))
        if block["blank_lead"]:
            lead = block["blank_lead"]
            print(f"  blank lead at r={lead['retention']}: "
                  f"{lead['spread']['mean']:+.4f} +- {lead['spread']['sd']:.4f} "
                  f"(range {lead['spread']['min']:+.4f} to {lead['spread']['max']:+.4f}), "
                  f"best filter always {lead['best_filter_is_always']}")

    out = args.out or RESULTS / "null_seed_sensitivity.json"
    out.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
