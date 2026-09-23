"""How much of the downstream verdict rests on one filter.

RED carries several of this study's largest numbers: the blank-sample maximum, the largest
E-MLB delta, the worst event-cap sensitivity, and the highest MESR at every partial retention
of the frozen downstream probe. A reader is entitled to ask which conclusions survive its
removal, and the answer differs by experiment, so it is measured here rather than asserted.

Recomputes the downstream correlations of Sec. V-F with and without RED, from the shipped
artifacts and not from the tables that print them:

* `results/downstream_gesture.json` - the trained 2D CNN over 40 conditions.
* `results/downstream_gesture_frozen.json` - the released classifier, frozen, 33 conditions.

Writes `results/red_sensitivity.json`. Run:

    python -m dataset_assessment.red_sensitivity
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import spearmanr

RESULTS = Path(__file__).resolve().parents[1] / "results"
EXCLUDED = "red"

#: The eight methods Sec. V-F reports. `downstream_gesture.json` also carries the two
#: native3d rows, which are not on that grid and would change n from 40 to 50.
METHODS = ("raw", "random_null", "dwf", "evflow", "knoise", "red", "ts", "ynoise")


def _rho(pairs: Sequence[Sequence[float]]) -> Dict:
    """Spearman rho and its nominal p over (mesr, accuracy) pairs."""

    if len(pairs) < 3:
        return {"n": len(pairs), "rho": float("nan"), "p": float("nan")}
    rho, p = spearmanr([x[0] for x in pairs], [x[1] for x in pairs])
    return {"n": len(pairs), "rho": float(rho), "p": float(p)}


def within_retention(conditions: Sequence[Dict], drop: Sequence[str] = ()) -> Dict:
    """Spearman at each fixed retention below 1, and the mean over them.

    Every comparison is within a retention, which the equal-count binning leaves unbiased;
    the mean is over levels, each of which has one point per method.
    """

    drop = {d.lower() for d in drop}
    by_r: Dict[float, List] = defaultdict(list)
    for c in conditions:
        if c["method"].lower() in drop or c["retention"] >= 1.0:
            continue
        by_r[round(float(c["retention"]), 3)].append((c["mesr"], c["accuracy"]))

    levels = [{"retention": r, **_rho(by_r[r])} for r in sorted(by_r)]
    rhos = [lv["rho"] for lv in levels if np.isfinite(lv["rho"])]
    return {"levels": levels, "mean_rho": float(np.mean(rhos)) if rhos else float("nan")}


def inversion(conditions: Sequence[Dict], drop: Sequence[str] = ()) -> Dict:
    """Where the highest-MESR method is also the least accurate, retention by retention."""

    drop = {d.lower() for d in drop}
    by_r: Dict[float, List[Dict]] = defaultdict(list)
    for c in conditions:
        if c["method"].lower() in drop or c["retention"] >= 1.0:
            continue
        by_r[round(float(c["retention"]), 3)].append(c)

    levels = []
    for r in sorted(by_r):
        rows = by_r[r]
        top = max(rows, key=lambda c: c["mesr"])
        order = sorted(rows, key=lambda c: c["accuracy"])
        levels.append({"retention": r, "top_mesr_method": top["method"],
                       "accuracy": float(top["accuracy"]),
                       "accuracy_rank": order.index(top) + 1, "methods": len(rows)})
    return {"levels": levels,
            "levels_least_accurate": sum(lv["accuracy_rank"] == 1 for lv in levels)}


def pooled(conditions: Sequence[Dict], drop: Sequence[str] = ()) -> Dict:
    """Spearman over every condition, retention not held."""

    drop = {d.lower() for d in drop}
    return _rho([(c["mesr"], c["accuracy"]) for c in conditions
                 if c["method"].lower() not in drop])


def analyse(conditions: Sequence[Dict]) -> Dict:
    return {"with_red": {"pooled": pooled(conditions),
                         "within_retention": within_retention(conditions),
                         "inversion": inversion(conditions)},
            "without_red": {"pooled": pooled(conditions, [EXCLUDED]),
                            "within_retention": within_retention(conditions, [EXCLUDED]),
                            "inversion": inversion(conditions, [EXCLUDED])}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    frozen = json.loads((RESULTS / "downstream_gesture_frozen.json").read_text())
    trained = json.loads((RESULTS / "downstream_gesture.json").read_text())
    trained_conditions = [c for c in (trained.get("conditions") or trained["records"])
                          if c["method"].lower() in METHODS]

    result = {"excluded": EXCLUDED,
              "frozen": analyse(frozen["conditions"]),
              "trained_cnn2d": analyse(trained_conditions)}
    out = args.out or RESULTS / "red_sensitivity.json"
    out.write_text(json.dumps(result, indent=2, default=float) + "\n")

    for probe in ("frozen", "trained_cnn2d"):
        print(f"\n{probe}")
        for key in ("with_red", "without_red"):
            block = result[probe][key]
            wr, inv, pl = block["within_retention"], block["inversion"], block["pooled"]
            print(f"  {key:12s} pooled rho {pl['rho']:+.3f} (p={pl['p']:.3f}, n={pl['n']})"
                  f"   within-retention mean rho {wr['mean_rho']:+.3f}"
                  f"   least-accurate-at-top {inv['levels_least_accurate']}"
                  f"/{len(inv['levels'])}")
            print("               " + "  ".join(
                f"r={lv['retention']:.1f}:{lv['rho']:+.2f}" for lv in wr["levels"]))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
