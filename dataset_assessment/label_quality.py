"""Does a filter that out-scores the label oracle on MESR actually select events better?

`analyze.oracle_violations` answers "how often does a real denoiser beat `label_oracle`" by
comparing `mesr_star` on both sides -- each method's own maximum over the retention grid.
That comparison has two defects, and this module exists to replace it:

1. **It is not matched.** The method's argmax and the oracle's argmax are different
   retentions, so the two outputs need not even retain the same number of events. A filter
   can "beat" the oracle by being measured somewhere else on the curve.
2. **It selects on the test metric.** `mesr_star` is the metric-oracle quantity the paper's
   own reporting protocol tells everyone else not to rank on. Using it for the headline
   oracle result is the paper committing the error it names.

Matched retention removes both. At a retention `r`, `esr.retain_mask` keeps the same count
in every 8192-event block for every method, so the two outputs retain **exactly** the same
number of events. That makes the label comparison a clean one:

    TP + FP = k   for both sides, with the same k,

so signal retention and noise retention are perfectly anti-correlated and a cell falls into
exactly one of three classes:

* ``strict``      -- the filter retains *fewer* signal events than the oracle at the same
                     count, hence necessarily more noise, and still scores higher MESR.
                     This is a genuine label-quality reversal.
* ``tie``         -- both retain the same signal count. Both are block-wise label-optimal,
                     so MESR is separating two equally good selections; the metric is not a
                     function of label quality. Weaker than ``strict``, still informative.
* ``impossible``  -- the filter retains *more* signal than the oracle at the same count.
                     Cannot happen: the oracle sorts by label within each block, so it
                     maximises retained signal at any block-wise count. Counted as a
                     self-check; a nonzero value means the oracle or the mask is wrong.

`label_oracle` is a *label* oracle, not a metric oracle. It ranks by ground-truth label and
breaks the resulting ties by arrival order (`retain_mask` uses a stable argsort, so at a
count below the signal total it keeps the temporally-first signal events of each block).
That is one label-optimal subset out of many, and ESR is not indifferent between them --
which is exactly why ``tie`` cells exist and why the oracle bounds nothing about MESR.

Writes `results/label_quality.json`. Run:

    python -m dataset_assessment.label_quality --dataset dnd21
    python -m dataset_assessment.label_quality --dataset dvsclean
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .denoisors import is_null, is_oracle, native_retention, score_events
from .esr import mesr, retain_mask
from .readers import iter_dnd21, iter_dvsclean

#: The two corpora with per-event labels. Everything here needs them.
LOADERS = {"dnd21": iter_dnd21, "dvsclean": iter_dvsclean}

#: Same grid as `run_benchmark`, so cells line up with `benchmark_<dataset>.json`.
RETENTIONS = np.round(np.arange(0.05, 1.001, 0.05), 3)

ORACLE = "label_oracle"
OUT_DIR = Path(__file__).resolve().parents[1] / "results"


def _quality(labels: np.ndarray, keep: np.ndarray) -> Dict[str, int]:
    """Retained signal and noise counts. `labels` is 1 for noise, 0 for signal."""

    kept = labels[keep]
    return {"kept": int(keep.sum()),
            "tp": int((kept == 0).sum()),      # signal retained
            "fp": int((kept == 1).sum())}      # noise retained


def _classify(method: Dict[str, int], oracle: Dict[str, int]) -> str:
    """Which of the three classes this cell falls into, given equal retained counts."""

    if method["tp"] < oracle["tp"]:
        return "strict"
    if method["tp"] == oracle["tp"]:
        return "tie"
    return "impossible"


def measure(dataset: str, max_events: int = 1_000_000,
            max_recordings: Optional[int] = None) -> Dict:
    """Every (recording, method, retention) cell with both MESR and label quality."""

    signal_total = 0
    noise_total = 0
    cells: List[Dict] = []
    started = time.perf_counter()

    for index, rec in enumerate(LOADERS[dataset]()):
        if max_recordings is not None and index >= max_recordings:
            break
        rec = replace(rec, events=rec.events[:max_events],
                      labels=None if rec.labels is None else rec.labels[:max_events])
        if rec.labels is None:
            raise ValueError(f"{rec.name} has no labels; this analysis needs them")

        labels = rec.labels
        n_signal = int((labels == 0).sum())
        n_noise = int((labels == 1).sum())
        signal_total += n_signal
        noise_total += n_noise

        scores = {m: score_events(m, rec)
                  for m in ("dwf", "evflow", "knoise", "red", "ts", "ynoise", ORACLE)}
        native = {m: native_retention(s) for m, s in scores.items()}

        for r in RETENTIONS:
            masks = {m: retain_mask(s, float(r)) for m, s in scores.items()}
            values = {m: mesr(rec.x[k], rec.y[k], rec.width, rec.height)
                      for m, k in masks.items()}
            if not np.isfinite(values[ORACLE]):
                continue                      # oracle unevaluable here; nothing to match to
            oq = _quality(labels, masks[ORACLE])

            for method in scores:
                if method == ORACLE or is_null(method) or is_oracle(method):
                    continue
                if not np.isfinite(values[method]):
                    continue
                mq = _quality(labels, masks[method])
                # The counts must agree for the comparison to mean anything. `retain_mask`
                # keeps the same number per block for every method, so they do -- but assert
                # it rather than assume it, because the whole classification rests on it.
                if mq["kept"] != oq["kept"]:
                    raise AssertionError(
                        f"{rec.name} {method} r={r}: retained {mq['kept']} against the "
                        f"oracle's {oq['kept']}; the matched comparison is invalid")
                cells.append({
                    "recording": rec.name, "method": method, "r": float(r),
                    "kept": mq["kept"],
                    "n_signal": n_signal, "n_noise": n_noise,
                    "mesr": values[method], "oracle_mesr": values[ORACLE],
                    "mesr_win": bool(values[method] > values[ORACLE]),
                    "tp": mq["tp"], "fp": mq["fp"],
                    "oracle_tp": oq["tp"], "oracle_fp": oq["fp"],
                    "signal_retention": mq["tp"] / n_signal if n_signal else float("nan"),
                    "noise_retention": mq["fp"] / n_noise if n_noise else float("nan"),
                    "oracle_signal_retention": oq["tp"] / n_signal if n_signal else float("nan"),
                    "oracle_noise_retention": oq["fp"] / n_noise if n_noise else float("nan"),
                    "label_class": _classify(mq, oq),
                    "native_retention": native[method],
                })
        print(f"[{index}] {rec.name}: {len(cells)} cells so far", flush=True)

    return {"dataset": dataset, "max_events": max_events,
            "retentions": RETENTIONS.tolist(),
            "signal_events": signal_total, "noise_events": noise_total,
            "cells": cells, "wall_seconds": time.perf_counter() - started}


def summarise(payload: Dict, grid_step: float = 0.05, rel_tol: float = 0.5) -> Dict:
    """Counts that the paper quotes, at matched retention rather than at duelling argmaxes.

    Three cohorts, because they answer different questions:

    * ``all_matched``  -- every (recording, method, retention). Descriptive only: retentions
      within a recording are not independent, so this is a cell count, not a test.
    * ``at_native``    -- one cell per (recording, method), at the retention nearest the
      method's own operating point, and only where that grid point is within both one grid
      step and `rel_tol` of it. Oracle-free on the method's side, so this is the cohort the
      reporting protocol permits and the one the paper should quote.
    * ``per_cell_any`` -- (recording, method) pairs with at least one matched-retention win.
      Reported for comparison with the published number, and inflated by multiple looks.
    """

    cells = payload["cells"]
    out: Dict = {"dataset": payload["dataset"]}

    def tally(rows: List[Dict]) -> Dict:
        wins = [c for c in rows if c["mesr_win"]]
        return {
            "cells": len(rows),
            "mesr_wins": len(wins),
            "share_mesr_win": len(wins) / len(rows) if rows else float("nan"),
            "strict_reversals": sum(1 for c in wins if c["label_class"] == "strict"),
            "ties": sum(1 for c in wins if c["label_class"] == "tie"),
            "impossible": sum(1 for c in wins if c["label_class"] == "impossible"),
        }

    out["all_matched"] = tally(cells)

    at_native: List[Dict] = []
    per_pair: Dict[tuple, List[Dict]] = {}
    for c in cells:
        per_pair.setdefault((c["recording"], c["method"]), []).append(c)
    for (_rec, _m), rows in per_pair.items():
        nat = rows[0]["native_retention"]
        if nat is None or not np.isfinite(nat):
            continue
        nearest = min(rows, key=lambda c: abs(c["r"] - nat))
        # Two tolerances, because one grid step is the wrong yardstick at the low end. A
        # method that natively keeps 1% is not being measured "at its operating point" when
        # compared at r=0.05: that is five times the events it actually retains, even though
        # the absolute gap is under one step. The relative bound is what makes "native" mean
        # native. Below the grid floor a native operating point has no matched cell at all;
        # excluding it is an eligibility rule, not a result, and `total_pairs` records it.
        gap = abs(nearest["r"] - nat)
        if gap <= grid_step + 1e-9 and gap <= rel_tol * nat + 1e-12:
            at_native.append(nearest)
    out["at_native"] = tally(at_native)
    out["at_native"]["eligible_pairs"] = len(at_native)
    out["at_native"]["total_pairs"] = len(per_pair)

    any_win = [rows for rows in per_pair.values() if any(c["mesr_win"] for c in rows)]
    out["per_cell_any"] = {
        "pairs": len(per_pair),
        "pairs_with_a_win": len(any_win),
        "share": len(any_win) / len(per_pair) if per_pair else float("nan"),
    }

    # Scoped to the native cohort, because that is the cohort the paper quotes. A larger
    # deficit exists at swept retentions, but a swept binary filter is "filter plus
    # decimation" and not the released method, so it is the weaker example to lead with.
    strict = [c for c in at_native if c["mesr_win"] and c["label_class"] == "strict"]
    if strict:
        worst = max(strict, key=lambda c: c["oracle_signal_retention"] - c["signal_retention"])
        out["largest_signal_deficit"] = worst
        out["max_mesr_margin_strict"] = max(c["mesr"] - c["oracle_mesr"] for c in strict)
    out["by_method"] = {
        m: tally([c for c in at_native if c["method"] == m])
        for m in sorted({c["method"] for c in at_native})
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    parser.add_argument("--max-events", type=int, default=1_000_000)
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payload = measure(args.dataset, args.max_events, args.max_recordings)
    payload["summary"] = summarise(payload)
    out = args.out or OUT_DIR / f"label_quality_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float) + "\n")

    s = payload["summary"]
    print(f"\n{args.dataset}: {s['all_matched']['cells']} matched cells")
    for cohort in ("all_matched", "at_native"):
        t = s[cohort]
        print(f"  {cohort:12s} MESR wins {t['mesr_wins']}/{t['cells']} "
              f"-> strict {t['strict_reversals']}, tie {t['ties']}, "
              f"impossible {t['impossible']}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
