"""Does the E-MLB ordering survive a paired comparison, rather than an interval overlap?

Table~IV ranked methods by a mean Delta-over-Raw and then reasoned about the ranking from
whether two marginal confidence intervals overlap. That reasoning is wrong in both
directions the table used it:

* **"its interval contains the other's value, so we claim only that it is not below it"** --
  overlapping marginal intervals do not imply the difference is indistinguishable from zero.
  Two intervals can overlap substantially while the paired difference is far from zero.
* **"these two overlap, so the lower half is not a ranking"** -- same error. Absence of a
  separation between marginals is not evidence of no difference.

The repair is available and strictly stronger: every classical row is scored on the *same*
384 E-MLB recordings, so the comparison is paired. This module forms the per-recording
difference `Delta_a - Delta_b` at the two methods' own native operating points and bootstraps
it over scenes, which is the unit E-MLB can actually carry (48 scenes x 4 ND levels x 2
lighting conditions, resampled as 96 (scene, lighting) clusters by `analyze.scene_of`).

A paired difference removes the recording-to-recording variance the marginal intervals
carry, so it resolves differences the marginals cannot -- which is the point: the previous
reasoning discarded real orderings as unresolvable.

EDformer is deliberately excluded. Its row is a published fixed threshold on an uncapped run
with eight (lighting, ND) cells, so it is not paired with anything here and stays descriptive
(\\appref{app:edformer}).

Writes `results/paired_contrasts.json`. Run:

    python -m dataset_assessment.paired_contrasts --dataset emlb
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .analyze import (RESULTS, _mesr_at, _nearest_evaluable, cluster_bootstrap_delta_ci,
                      scene_of)
from .protocol import native_point_is_eligible
from .denoisors import has_native_operating_point, is_null


def native_deltas_by_recording(dataset: str,
                               eligible_only: bool = True) -> Dict[str, Dict[str, float]]:
    """Delta-over-Raw at each method's native operating point, keyed recording -> method.

    `analyze.per_recording_native_deltas` drops the recording name, which is exactly what a
    paired comparison needs; this keeps it.

    `eligible_only` applies `protocol.native_point_is_eligible`. It matters more here than
    anywhere else in the paper. On E-MLB, 445 of 2304 cells have a native retention below the
    recording's measurable floor, so the sweep scores them at a retention the filter never
    chose -- and those cells are not a random sample. RED's 73 ineligible cells average
    +2.7180 against +0.4550 on its 311 eligible ones, so the loose cohort put RED at the top
    of the table on cells where it could not be measured at its own operating point at all.
    """

    payload = json.loads((RESULTS / f"benchmark_{dataset}.json").read_text())
    out: Dict[str, Dict[str, float]] = {}
    for record in payload["records"]:
        methods = record["methods"]
        raw = methods.get("raw", {})
        if "curve" not in raw or "error" in raw:
            continue
        raw_native = _mesr_at(raw["curve"], raw.get("native_retention", 1.0))
        if not np.isfinite(raw_native):
            continue
        cell: Dict[str, float] = {}
        for method, row in methods.items():
            if "error" in row or "curve" not in row or row.get("is_oracle"):
                continue
            if not has_native_operating_point(method) or is_null(method):
                continue
            point = _nearest_evaluable(row["curve"], row["native_retention"])
            if point is None or not np.isfinite(point["mesr"]):
                continue
            if eligible_only and not native_point_is_eligible(row["native_retention"],
                                                              point["r"]):
                continue
            cell[method] = float(point["mesr"] - raw_native)
        if cell:
            out[record["recording"]] = cell
    return out


def paired_difference(deltas: Dict[str, Dict[str, float]], a: str, b: str) -> Dict:
    """Mean of `Delta_a - Delta_b` over the recordings scoring both, bootstrapped by scene."""

    shared = sorted(r for r, cell in deltas.items() if a in cell and b in cell)
    diffs = [deltas[r][a] - deltas[r][b] for r in shared]
    scenes = [scene_of(r) for r in shared]
    ci = cluster_bootstrap_delta_ci(diffs, scenes)
    wins = int(sum(d > 0 for d in diffs))
    return {
        "a": a, "b": b,
        "n_recordings": len(shared),
        "n_scenes": ci.get("n_clusters"),
        "mean_difference": ci["mean"],
        "lo": ci["lo"], "hi": ci["hi"],
        "excludes_zero": bool(ci.get("excludes_zero", False)),
        "median_difference": float(np.median(diffs)) if diffs else float("nan"),
        # Sign consistency is what a marginal interval cannot show at all: how often the
        # ordering holds within a recording, not just on average over recordings.
        "recordings_with_a_above_b": wins,
        "share_a_above_b": wins / len(diffs) if diffs else float("nan"),
    }


def compare(dataset: str = "emlb", order: Optional[Sequence[str]] = None) -> Dict:
    """Every ordered pair of classical methods, ranked by mean Delta-over-Raw."""

    deltas = native_deltas_by_recording(dataset)
    methods = sorted({m for cell in deltas.values() for m in cell})
    means = {m: float(np.mean([cell[m] for cell in deltas.values() if m in cell]))
             for m in methods}
    ranked = list(order) if order else sorted(methods, key=lambda m: -means[m])

    # The marginals the table prints. Each row is now its own cohort, because eligibility is
    # a property of (method, recording), so these are NOT comparable across rows -- the
    # pairwise contrasts below are, and they are what every comparative statement rests on.
    marginals = {}
    for m in ranked:
        recs = sorted(r for r, cell in deltas.items() if m in cell)
        ci = cluster_bootstrap_delta_ci([deltas[r][m] for r in recs],
                                        [scene_of(r) for r in recs])
        marginals[m] = {"n_recordings": len(recs), "n_scenes": ci.get("n_clusters"),
                        "mean": ci["mean"], "lo": ci["lo"], "hi": ci["hi"],
                        "excludes_zero": bool(ci.get("excludes_zero", False))}

    pairs = [paired_difference(deltas, a, b)
             for a, b in itertools.combinations(ranked, 2)]
    adjacent = [p for p in pairs
                if ranked.index(p["b"]) == ranked.index(p["a"]) + 1]
    return {
        "dataset": dataset,
        "n_recordings": len(deltas),
        "n_scenes": len({scene_of(r) for r in deltas}),
        "ranked_by_mean": ranked,
        "mean_delta": {m: means[m] for m in ranked},
        "marginals": marginals,
        "pairs": pairs,
        "adjacent_pairs": adjacent,
        "adjacent_all_resolved": all(p["excludes_zero"] for p in adjacent),
        "pairs_resolved": sum(1 for p in pairs if p["excludes_zero"]),
        "pairs_total": len(pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="emlb")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = compare(args.dataset)
    out = args.out or RESULTS / "paired_contrasts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float) + "\n")

    print(f"{args.dataset}: {result['n_recordings']} recordings, "
          f"{result['n_scenes']} scenes; ranking "
          f"{' > '.join(result['ranked_by_mean'])}")
    print(f"{result['pairs_resolved']}/{result['pairs_total']} pairs resolved by a "
          f"scene-clustered paired interval")
    print("\nadjacent pairs in the ranking:")
    for p in result["adjacent_pairs"]:
        mark = "resolved" if p["excludes_zero"] else "NOT resolved"
        print(f"  {p['a']:8s} - {p['b']:8s} {p['mean_difference']:+.4f} "
              f"[{p['lo']:+.4f}, {p['hi']:+.4f}]  "
              f"{p['share_a_above_b']:.0%} of recordings  {mark}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
