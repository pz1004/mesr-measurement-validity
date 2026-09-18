"""What removing the busiest 0.1% of pixels does to the two failures seen on real captures.

Specificity (a nonselective null raising MESR) and the blank-sample response (filters beating
a nonselective control on a signal-free capture) fail only on the two real unfiltered corpora,
DVSD22 and Pure_BA. `run_benchmark --hot-pixel-removal` repeats those runs with the busiest
0.1% of occupied pixels removed from the input first -- the rule `regime` uses to characterise
the corpora, fixed without reference to MESR. This module reads each pair of runs the same way:

* **Only recordings evaluable under both.** A before/after difference is then a difference
  in the stream and never in the cohort, so the before-means sit just under the published
  full-cohort columns.
* **Specificity**: the count of retentions below 1 at which `random_null`'s mean
  Delta-over-Raw is positive, before and after, as `analyze.null_gain_at_fixed_retention`
  counts it, and the mean at one stated retention with how many recordings drop.
* **Blank sample** (Pure_BA only; it is the capture with nominally no scene): each filter's
  Delta and its paired margin over `random_null` at stated retentions, before and after. No
  interval is attached, as in the paper: Pure_BA is one graded session, so resampling its
  recordings reweights levels rather than bounding acquisition.

Writes `results/hotpixel_contrast.json`. Run:

    python -m dataset_assessment.hotpixel_contrast
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

import numpy as np

from .analyze import RESULTS

OUT = RESULTS / "hotpixel_contrast.json"
FILTERS = ("dwf", "evflow", "knoise", "red", "ts", "ynoise")
NULL = "random_null"
PAIRS = {
    "dvsd22": ("benchmark_dvsd22.json", "benchmark_dvsd22_hotpixel.json"),
    "pure_ba": ("benchmark_pure_ba.json", "benchmark_pure_ba_hotpixel_all.json"),
}
SPECIFICITY_AT = {"dvsd22": 0.05, "pure_ba": 0.20}
BLANK_AT = (0.05, 0.20, 0.60)


def deltas(payload: Dict, method: str, r: float) -> Dict[str, float]:
    """Delta-over-Raw of one method at one retention, by recording, evaluable points only."""

    out: Dict[str, float] = {}
    for record in payload["records"]:
        row = record["methods"].get(method)
        if row is None or "error" in row or "raw_mesr" not in row:
            continue
        for point in row.get("curve", []):
            if (abs(float(point["r"]) - r) < 1e-9 and point.get("evaluable")
                    and np.isfinite(point["mesr"])):
                out[record["recording"]] = float(point["mesr"] - row["raw_mesr"])
    return out


def paired(before: Dict[str, float], after: Dict[str, float]) -> Optional[Dict]:
    """Means over the recordings present in both, and how many fall."""

    shared = sorted(set(before) & set(after))
    if not shared:
        return None
    b = np.array([before[k] for k in shared])
    a = np.array([after[k] for k in shared])
    return {"n": len(shared), "before": float(b.mean()), "after": float(a.mean()),
            "n_drop": int((a < b).sum())}


def retentions_below_one(payload: Dict) -> List[float]:
    return [float(r) for r in payload["retentions"] if float(r) < 1.0]


def specificity(before: Dict, after: Dict, at: float) -> Dict:
    """`random_null`'s positive-mean count before and after, on a shared cohort per r."""

    counts = {"before": 0, "after": 0}
    after_positive: List[Dict] = []
    for r in retentions_below_one(before):
        row = paired(deltas(before, NULL, r), deltas(after, NULL, r))
        if row is None:
            continue
        counts["before"] += row["before"] > 0
        counts["after"] += row["after"] > 0
        if row["after"] > 0:
            after_positive.append({"r": r, "mean": row["after"]})
    return {"n_positive_before": int(counts["before"]), "n_positive_after": int(counts["after"]),
            "n_retentions": len(retentions_below_one(before)),
            "after_positive": after_positive,
            "at": at, "at_r": paired(deltas(before, NULL, at), deltas(after, NULL, at))}


def margins(payload: Dict, r: float, recordings: Sequence[str]) -> Dict[str, Dict]:
    """Each method's mean Delta, and each filter's paired margin over the null, on a cohort."""

    null = deltas(payload, NULL, r)
    out: Dict[str, Dict] = {}
    for method in FILTERS + ("raw", NULL):
        d = deltas(payload, method, r)
        keys = [k for k in recordings if k in d and k in null]
        if not keys:
            continue
        diff = [d[k] - null[k] for k in keys]
        out[method] = {"n": len(keys), "delta": float(np.mean([d[k] for k in keys])),
                       "minus_null": float(np.mean(diff)),
                       "n_above_null": int(sum(x > 0 for x in diff))}
    return out


def blank(before: Dict, after: Dict, retentions: Sequence[float] = BLANK_AT) -> Dict:
    """Filters against the null on the blank capture, on recordings evaluable under both."""

    out: Dict[str, Dict] = {}
    for r in retentions:
        present = [set(deltas(p, m, r)) for p in (before, after) for m in FILTERS + ("raw", NULL)]
        cohort = sorted(set.intersection(*present)) if present else []
        rows = {"before": margins(before, r, cohort), "after": margins(after, r, cohort)}
        best = {when: max(FILTERS, key=lambda m: rows[when][m]["minus_null"])
                for when in rows if all(m in rows[when] for m in FILTERS)}
        out[f"{r:.2f}"] = {"n": len(cohort), "best": best, **rows}
    return out


def measure() -> Dict:
    out: Dict[str, Dict] = {}
    for dataset, (released, removed) in PAIRS.items():
        before = json.loads((RESULTS / released).read_text())
        after = json.loads((RESULTS / removed).read_text())
        if not after.get("hot_pixel_removal"):
            raise ValueError(f"{removed} was not run with --hot-pixel-removal")
        row = {"files": [released, removed],
               "specificity": specificity(before, after, SPECIFICITY_AT[dataset])}
        if dataset == "pure_ba":
            row["blank"] = blank(before, after)
        out[dataset] = row
    return out


def main() -> None:
    result = measure()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    for dataset, row in result.items():
        s = row["specificity"]
        print(f"{dataset}: null positive at {s['n_positive_before']} -> {s['n_positive_after']}"
              f"/{s['n_retentions']}; at r={s['at']}: {s['at_r']}")
        for r, b in row.get("blank", {}).items():
            print(f"  blank r={r} n={b['n']} best {b['best']}: "
                  + ", ".join(f"{w} {b[w][b['best'][w]]['minus_null']:+.4f}"
                              for w in ("before", "after")))


if __name__ == "__main__":
    main()
