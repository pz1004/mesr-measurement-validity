"""Does a null raise MESR beyond sampling variation, retention by retention?

The specificity criterion forbids a *rise beyond sampling variation* under removal that
selects nothing. `analyze.null_gain_at_fixed_retention` counts the retentions below 1 at
which a null's mean Delta-over-Raw is positive, and puts a scene-clustered interval on the
one retention that maximises the mean. A sign count says nothing about sampling variation,
and an interval at a selected retention describes rather than tests. This module puts the
same scene-clustered interval (`analyze.cluster_bootstrap_delta_ci`) on *every* retention
below 1 and counts those whose interval lies wholly above zero, or wholly below it -- at the
nominal 95% level, and Bonferroni-corrected across the corpus's retentions, which holds
under the dependence between neighbouring retentions.

Writes `results/null_intervals.json`. Run:

    python -m dataset_assessment.null_intervals
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .analyze import RESULTS, cluster_bootstrap_delta_ci, scene_of

CORPORA = ("dnd21", "dvsclean", "emlb", "dvsd22", "pure_ba")
NULLS = ("raw", "random_null")
OUT = RESULTS / "null_intervals.json"


def deltas_by_retention(payload: Dict, null: str) -> Dict[float, Tuple[List[float], List[str]]]:
    """Per-retention Delta-over-Raw and scene of every evaluable recording, as `analyze` reads them."""

    grid = [float(r) for r in payload["retentions"]]
    out: Dict[float, Tuple[List[float], List[str]]] = {r: ([], []) for r in grid}
    for record in payload["records"]:
        row = record["methods"].get(null)
        if row is None or "error" in row or "raw_mesr" not in row:
            continue
        scene = scene_of(record["recording"])
        for point in row["curve"]:
            r = float(point["r"])
            if point.get("evaluable") and np.isfinite(point["mesr"]) and r in out:
                out[r][0].append(float(point["mesr"] - row["raw_mesr"]))
                out[r][1].append(scene)
    return out


def profile(by_r: Dict[float, Tuple[List[float], List[str]]]) -> Dict:
    """Scene interval at every retention below 1, and how many exclude zero on each side."""

    rows = [r for r in sorted(by_r) if r < 1.0 and by_r[r][0]]
    familywise = 0.05 / len(rows) if rows else 0.05
    points = []
    for r in rows:
        values, scenes = by_r[r]
        nominal = cluster_bootstrap_delta_ci(values, scenes)
        corrected = cluster_bootstrap_delta_ci(values, scenes, alpha=familywise)
        points.append({"r": r, "mean": nominal["mean"], "n": nominal["n"],
                       "n_clusters": nominal["n_clusters"],
                       "lo": nominal["lo"], "hi": nominal["hi"],
                       "lo_bonferroni": corrected["lo"], "hi_bonferroni": corrected["hi"]})
    return {
        "n_retentions_below_one": len(rows),
        "bonferroni_alpha": familywise,
        "n_positive_mean": sum(p["mean"] > 0 for p in points),
        "n_above_zero": sum(p["lo"] > 0 for p in points),
        "n_below_zero": sum(p["hi"] < 0 for p in points),
        "n_above_zero_bonferroni": sum(p["lo_bonferroni"] > 0 for p in points),
        "n_below_zero_bonferroni": sum(p["hi_bonferroni"] < 0 for p in points),
        "not_above_zero_at": [p["r"] for p in points if not p["lo"] > 0],
        "profile": points,
    }


def measure(corpora: Sequence[str] = CORPORA, nulls: Sequence[str] = NULLS) -> Dict:
    out: Dict[str, Dict] = {}
    for dataset in corpora:
        payload = json.loads((RESULTS / f"benchmark_{dataset}.json").read_text())
        out[dataset] = {null: profile(deltas_by_retention(payload, null)) for null in nulls}
    return out


def main() -> None:
    result = measure()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    for dataset, rows in result.items():
        for null, row in rows.items():
            print(f"{dataset} {null}: above zero {row['n_above_zero']}/"
                  f"{row['n_retentions_below_one']} "
                  f"(Bonferroni {row['n_above_zero_bonferroni']}), below zero "
                  f"{row['n_below_zero']}")


if __name__ == "__main__":
    main()
