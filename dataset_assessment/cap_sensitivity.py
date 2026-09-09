"""Is Delta-over-Raw really invariant to the event cap?

The paper asserted in six places that it is, and used that assertion to place capped
classical rows beside an uncapped EDformer row in the E-MLB table. Nothing measured it.

There is no mechanism that would make it hold. MESR is a mean of ESR over complete
30,000-event slices, so raising the cap adds slices to *both* terms of

    Delta_B(method) = MESR_B(filter(E)) - MESR_B(E),

and the two means shift by different amounts unless the filter's effect is homogeneous over
the recording -- which is exactly what a denoiser responding to scene content is not. On
E-MLB the cap also binds hard: 308 of the 384 recordings in the published run are truncated
by it, so this is not a question about a rare long recording.

This module compares runs of `run_benchmark` that differ only in `--max-events`, on the same
recordings and methods, and reports the per-cell spread of Delta-over-Raw across caps against
0.0092 -- the published gap the paper uses as its unit of scale.

    python -m dataset_assessment.cap_sensitivity \\
        results/cap_sensitivity/emlb_cap_*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .analyze import _mesr_at
from .denoisors import has_native_operating_point

#: The published EDformer-vs-EDmamba gap, the paper's unit of scale for every ratio.
UNIT_OF_SCALE = 0.0092


def native_deltas(payload: Dict) -> Dict[str, Dict[str, float]]:
    """Delta-over-Raw at each method's native operating point, keyed recording -> method.

    Same convention as `analyze.per_recording_native_deltas`, which is what the E-MLB table
    prints: the method at its own operating point, minus the unfiltered stream at r = 1.
    """

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
            if not has_native_operating_point(method):
                continue
            value = _mesr_at(row["curve"], row["native_retention"])
            if np.isfinite(value):
                cell[method] = float(value - raw_native)
        if cell:
            out[record["recording"]] = cell
    return out


#: Retentions at which the "pure" cap effect is measured -- both are evaluable in every run,
#: so the comparison is at a genuinely fixed r and cannot be contaminated by the floor moving.
FIXED_RETENTIONS = (0.50, 0.95)


def fixed_retention_effect(runs: List[Dict], payloads: List[Dict],
                           retentions: Sequence[float] = FIXED_RETENTIONS) -> Dict:
    """Cap sensitivity of Delta-over-Raw at a *fixed* retention.

    Separating this from the at-native number matters. At native operating points two things
    move when the cap changes: the value itself, and which retentions are evaluable at all --
    a smaller cap yields fewer complete slices, so the measurable floor rises and a native
    point below it is snapped somewhere else entirely. Holding r fixed isolates the first.
    """

    by_cap = [{r["recording"]: r for r in p["records"]} for p in payloads]
    common = set.intersection(*[set(d) for d in by_cap])
    out: Dict[str, Dict] = {}
    for r in retentions:
        diffs: List[float] = []
        for rec in sorted(common):
            deltas = []
            for d in by_cap:
                methods = d[rec]["methods"]
                raw = _mesr_at(methods["raw"]["curve"], r)
                if not np.isfinite(raw):
                    deltas = None
                    break
                deltas.append({m: _mesr_at(row["curve"], r) - raw
                               for m, row in methods.items()
                               if "curve" in row and has_native_operating_point(m)})
            if not deltas:
                continue
            for m in set.intersection(*[set(d) for d in deltas]):
                if m == "raw":
                    continue                      # zero by construction, see above
                values = [d[m] for d in deltas]
                if all(np.isfinite(v) for v in values):
                    diffs.append(float(max(values) - min(values)))
        arr = np.array(diffs)
        out[f"r={r:.2f}"] = {
            "cells": len(arr),
            "median": float(np.median(arr)) if arr.size else float("nan"),
            "max": float(arr.max()) if arr.size else float("nan"),
            "share_above_unit_of_scale": (float((arr > UNIT_OF_SCALE).mean())
                                          if arr.size else float("nan")),
        }
    return out


def evaluable_range_changes(payloads: List[Dict]) -> Dict:
    """How many cells keep the same evaluable retention range across the caps.

    If this is zero, then no at-native comparison across caps is holding r fixed, and the
    at-native spread below mixes the cap's effect on the value with its effect on the grid.
    """

    by_cap = [{r["recording"]: r for r in p["records"]} for p in payloads]
    common = set.intersection(*[set(d) for d in by_cap])
    same = total = 0
    for rec in sorted(common):
        methods = set.intersection(*[set(d[rec]["methods"]) for d in by_cap])
        for m in methods:
            if not has_native_operating_point(m):
                continue
            ranges = [tuple(d[rec]["methods"][m].get("evaluable_range", ())) for d in by_cap]
            total += 1
            same += len(set(ranges)) == 1
    return {"cells": total, "identical_range": same,
            "share_identical": same / total if total else float("nan")}


def means_over_all_recordings(runs: List[Dict]) -> Dict:
    """Per-method corpus mean of Delta-over-Raw at native, over *every* common recording.

    `compare`'s `per_method` block restricts to cells the smallest cap can bind, which is the
    right cohort for asking how big the cap effect is where the cap acts at all. It is the
    wrong cohort for asking whether the numbers the E-MLB table prints move, because that
    table averages over all 384 recordings including the short ones the cap never touches.
    Quoting the restricted mean as if it were the printed one overstates the movement.

    At 10^6 this reproduces the table exactly, which is what makes the other caps' columns
    comparable with it.
    """

    common = set.intersection(*[set(r["deltas"]) for r in runs])
    methods = sorted({m for r in runs for cell in r["deltas"].values() for m in cell})
    by_method: Dict[str, Dict] = {}
    for method in methods:
        mean_by_cap = {}
        for r in runs:
            vals = [r["deltas"][rec][method] for rec in common if method in r["deltas"][rec]]
            if len(vals) != len(common):      # not scored on every common recording
                mean_by_cap = {}
                break
            mean_by_cap[r["cap"]] = float(np.mean(vals))
        if not mean_by_cap:
            continue
        means = list(mean_by_cap.values())
        by_method[method] = {
            "mean_by_cap": mean_by_cap,
            "spread_across_caps": float(max(means) - min(means)),
            "spread_in_units_of_scale": float((max(means) - min(means)) / UNIT_OF_SCALE),
        }
    return {"recordings": len(common), "by_method": by_method}


def compare(paths: Sequence[Path]) -> Dict:
    """Spread of Delta-over-Raw across caps, over the cells common to every run."""

    runs = []
    payloads = []
    for p in sorted(paths, key=lambda q: json.loads(q.read_text())["max_events"]):
        payload = json.loads(p.read_text())
        payloads.append(payload)
        runs.append({"cap": payload["max_events"], "path": p.name,
                     "deltas": native_deltas(payload),
                     "events": {r["recording"]: r["events"] for r in payload["records"]}})
    if len(runs) < 2:
        raise ValueError("need at least two runs at different caps")

    common = set.intersection(*[{(rec, m) for rec, cell in r["deltas"].items() for m in cell}
                                for r in runs])
    cells = []
    for rec, method in sorted(common):
        values = [r["deltas"][rec][method] for r in runs]
        # A recording shorter than the smallest cap is untouched by any of them, so it
        # cannot show cap sensitivity and would dilute the summary. Count it separately.
        truncated = min(r["events"].get(rec, 0) for r in runs) >= min(r["cap"] for r in runs)
        cells.append({"recording": rec, "method": method,
                      "by_cap": {r["cap"]: v for r, v in zip(runs, values)},
                      "spread": float(max(values) - min(values)),
                      "truncated_at_smallest_cap": bool(truncated)})

    bound = [c for c in cells if c["truncated_at_smallest_cap"]]
    # `raw` is Delta-over-Raw against itself, so its spread is zero by construction. Keeping
    # it in `per_method` is a useful self-check, but letting it into the aggregate would
    # dilute every share by one method's worth of guaranteed zeros.
    scored = [c for c in bound if c["method"] != "raw"]
    spreads = np.array([c["spread"] for c in scored]) if scored else np.array([])

    per_method: Dict[str, Dict] = {}
    for method in sorted({c["method"] for c in bound}):
        rows = [c for c in bound if c["method"] == method]
        per_method[method] = {
            "cells": len(rows),
            "mean_by_cap": {cap: float(np.mean([r["by_cap"][cap] for r in rows]))
                            for cap in rows[0]["by_cap"]},
            "max_spread": float(max(r["spread"] for r in rows)),
            "median_spread": float(np.median([r["spread"] for r in rows])),
        }
    for m, v in per_method.items():
        means = list(v["mean_by_cap"].values())
        v["mean_spread_across_caps"] = float(max(means) - min(means))

    return {
        "caps": [r["cap"] for r in runs],
        "runs": [r["path"] for r in runs],
        "cells_compared": len(cells),
        "cells_bound_by_cap": len(bound),
        "cells_scored": len(scored),
        "unit_of_scale": UNIT_OF_SCALE,
        "spread": {
            "max": float(spreads.max()) if spreads.size else float("nan"),
            "median": float(np.median(spreads)) if spreads.size else float("nan"),
            "mean": float(spreads.mean()) if spreads.size else float("nan"),
            "share_above_unit_of_scale": (float((spreads > UNIT_OF_SCALE).mean())
                                          if spreads.size else float("nan")),
            "max_in_units_of_scale": (float(spreads.max() / UNIT_OF_SCALE)
                                      if spreads.size else float("nan")),
        },
        "per_method": per_method,
        "means_over_all_recordings": means_over_all_recordings(runs),
        "at_fixed_retention": fixed_retention_effect(runs, payloads),
        "evaluable_range": evaluable_range_changes(payloads),
        "worst_cells": sorted(scored, key=lambda c: -c["spread"])[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path,
                        help="benchmark JSONs differing only in --max-events")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "results" / "cap_sensitivity.json")
    args = parser.parse_args()

    result = compare(args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float) + "\n")

    s = result["spread"]
    print(f"caps {result['caps']}  cells {result['cells_compared']} "
          f"({result['cells_bound_by_cap']} bound by the smallest cap)")
    print(f"Delta-over-Raw spread across caps: max {s['max']:.4f} "
          f"({s['max_in_units_of_scale']:.1f}x the {UNIT_OF_SCALE} unit of scale), "
          f"median {s['median']:.4f}")
    print(f"cells whose spread exceeds the unit of scale: "
          f"{s['share_above_unit_of_scale']:.0%}")
    for m, v in result["per_method"].items():
        by = ", ".join(f"{c/1e6:g}M={d:+.4f}" for c, d in v["mean_by_cap"].items())
        print(f"  {m:12s} {by}   mean moves {v['mean_spread_across_caps']:+.4f}, "
              f"worst cell {v['max_spread']:.4f}")
    er = result["evaluable_range"]
    print(f"\nevaluable range identical across caps: {er['identical_range']}/{er['cells']} "
          f"cells ({er['share_identical']:.0%})")
    print("at a FIXED retention (the cap effect with the floor held out):")
    for label, v in result["at_fixed_retention"].items():
        print(f"  {label}: n={v['cells']} median {v['median']:.5f} max {v['max']:.4f} "
              f"({v['max']/UNIT_OF_SCALE:.0f}x), share above unit of scale "
              f"{v['share_above_unit_of_scale']:.0%}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
