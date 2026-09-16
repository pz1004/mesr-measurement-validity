"""Is the seed really larger than the filter? Pair the difference before asking.

`downstream_gesture` reports the across-seed standard deviation *within* one condition, and
the paper sets it beside the differences between methods: the seed spread is larger, so a
single-seed reading measures its own seed. That comparison is between two marginal
quantities, and it is not the uncertainty of the contrast anyone draws. For methods A and B
scored on the same seeds and the same test streams,

    Var(a_A - a_B) = Var(a_A) + Var(a_B) - 2 Cov(a_A, a_B),

so a large marginal spread can coexist with a stable difference whenever the seed moves both
conditions together. The question the paper asks needs the left-hand side, which the grid
already contains: every condition is trained under the same three seeds, so the difference
can be formed seed by seed rather than between means.

This module forms it. For each retention below 1 and each pair of the reported methods, it
takes the per-seed difference, and reports the spread of that difference beside the marginal
spreads it is usually compared with. Two readouts follow, neither of which needs a sampling
model the design does not supply:

* **resolved** -- the mean difference exceeds the across-seed standard deviation of the
  difference itself;
* **sign-consistent** -- all three seeds order the pair the same way.

`r = 1.0` is excluded throughout: every method is the unfiltered stream there, so the
difference is identically zero and would inflate both counts.

Nothing here reads MESR, and no existing statistic is recomputed: this adds the paired
readout beside the marginal one rather than replacing it. Either outcome is reportable. If
the paired spread is no smaller, the paper's claim stands as written; if it is much smaller
on some architecture, the claim does not hold there and the text has to say so.

Writes `results/downstream_paired_seeds.json`. Run:

    python -m dataset_assessment.downstream_paired
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .downstream_gesture import PAPER_METHODS

#: The three artifacts, keyed by the architecture label the paper uses.
ARCHITECTURES = {
    "cnn2d": "downstream_gesture.json",
    "cnn3d": "downstream_gesture_cnn3d.json",
    "mlp": "downstream_gesture_mlp.json",
}

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _seed_accuracies(condition: Dict) -> Optional[np.ndarray]:
    """Per-seed accuracies in a fixed seed order, or None when the cell carries none.

    The artifact stores them keyed by seed, so the order has to be imposed here; pairing is
    only meaningful if both sides are read in the same seed order.
    """

    per_seed = condition.get("accuracy_per_seed")
    if not per_seed:
        return None
    return np.array([float(per_seed[k]) for k in sorted(per_seed)], dtype=float)


def paired_differences(conditions: Iterable[Dict],
                       methods: Optional[Sequence[str]] = None) -> List[Dict]:
    """One row per (retention, method pair), each carrying the paired and marginal spreads.

    Restricted to `methods` -- the eight the paper reports -- because the extended grid adds
    two filters from a sibling project that no reported claim covers.
    """

    allowed = set(PAPER_METHODS if methods is None else methods)
    by_retention: Dict[float, Dict[str, np.ndarray]] = {}
    for row in conditions:
        if row["method"] not in allowed or float(row["retention"]) >= 1.0:
            continue
        seeds = _seed_accuracies(row)
        if seeds is None or len(seeds) < 2:
            continue
        by_retention.setdefault(float(row["retention"]), {})[row["method"]] = seeds

    out: List[Dict] = []
    for retention in sorted(by_retention):
        cells = by_retention[retention]
        for a, b in itertools.combinations(sorted(cells), 2):
            diff = cells[a] - cells[b]
            mean = float(np.mean(diff))
            sd = float(np.std(diff, ddof=1))
            out.append({
                "retention": retention,
                "a": a,
                "b": b,
                "n_seeds": int(len(diff)),
                "mean_difference": mean,
                "sd_of_difference": sd,
                "marginal_sd_a": float(np.std(cells[a], ddof=1)),
                "marginal_sd_b": float(np.std(cells[b], ddof=1)),
                # The comparison the paper makes, and the one the design supports.
                "resolved": bool(abs(mean) > sd),
                "sign_consistent": bool(np.all(diff > 0) or np.all(diff < 0)),
            })
    return out


def summarise(conditions: Iterable[Dict],
              methods: Optional[Sequence[str]] = None) -> Dict:
    """The counts the paper quotes, with the medians they are read against."""

    rows = paired_differences(conditions, methods)
    if not rows:
        return {"n_pairs": 0, "error": "no condition carried per-seed accuracies"}

    paired_sd = np.array([r["sd_of_difference"] for r in rows])
    marginal = np.array([r["marginal_sd_a"] for r in rows] + [r["marginal_sd_b"] for r in rows])
    abs_mean = np.array([abs(r["mean_difference"]) for r in rows])
    return {
        "n_pairs": len(rows),
        "n_resolved": int(sum(r["resolved"] for r in rows)),
        "n_sign_consistent": int(sum(r["sign_consistent"] for r in rows)),
        "median_paired_sd": float(np.median(paired_sd)),
        "median_marginal_sd": float(np.median(marginal)),
        "median_abs_mean_difference": float(np.median(abs_mean)),
        # > 1 would mean pairing buys precision; ~1 means the seed moves the conditions
        # independently and the marginal comparison was not misleading after all.
        "marginal_over_paired_sd": float(np.median(marginal) / np.median(paired_sd))
        if np.median(paired_sd) else float("nan"),
        "retentions": sorted({r["retention"] for r in rows}),
        "methods": sorted({m for r in rows for m in (r["a"], r["b"])}),
    }


def run(results: Path = RESULTS) -> Dict:
    """Every architecture's artifact, summarised; missing ones are reported, not skipped."""

    started = time.time()
    out: Dict[str, Dict] = {}
    for arch, filename in ARCHITECTURES.items():
        path = results / filename
        if not path.is_file():
            out[arch] = {"error": f"{filename} not generated"}
            continue
        payload = json.loads(path.read_text())
        out[arch] = summarise(payload["conditions"])
        out[arch]["pairs"] = paired_differences(payload["conditions"])
    return {
        "scope": "the eight reported methods, retentions below 1, three seeds",
        "architectures": out,
        "wall_seconds": round(time.time() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = run()
    out = args.out or RESULTS / "downstream_paired_seeds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float) + "\n")

    for arch, summary in result["architectures"].items():
        if "error" in summary:
            print(f"{arch:6s} {summary['error']}")
            continue
        print(f"{arch:6s} {summary['n_resolved']:3d}/{summary['n_pairs']} pairs resolved, "
              f"{summary['n_sign_consistent']:3d} sign-consistent; "
              f"paired sd {summary['median_paired_sd']:.3f} against marginal "
              f"{summary['median_marginal_sd']:.3f}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
