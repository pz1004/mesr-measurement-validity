"""Does the slice-size convention reorder filters, or only shift every score together?

`measure_metric_deps` sweeps the joint (N, M) convention on each DND21 recording's unfiltered
stream and finds MESR moving by 0.5267 on average. A shift that large says nothing, by itself,
about comparisons made under one common convention: an offset shared by every filter would
leave every ranking intact. This module scores each classical filter's *own* output -- its
native mask, no retention argument -- at the same five slice sizes, on the same first 10^6
events `native_oracle` uses, and asks whether the ordering of the filters moves with the
convention.

Two readings, both against the benchmark's own 30,000-event convention:

* **Per recording.** The filters evaluable at all five sizes (at least one complete
  100,000-event slice) are ranked at each size; Kendall tau against the 30,000 ranking and the
  number of discordant pairs are reported, with whether the top filter changes.
* **Corpus mean, paired.** For every pair of filters, the mean MESR difference over the
  recordings on which both are evaluable at *every* size, so each pair keeps one cohort across
  the sweep. A pair flips when that difference changes sign relative to 30,000.

The unfiltered stream is also re-swept on the same 10^6-event support, so the spread and the
filter gaps it is set against are read on the same events.

Writes `results/slice_ranking_dnd21.json`. Run:

    python -m dataset_assessment.slice_ranking
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import kendalltau

from .denoisors import CLASSICAL, score_events
from .esr import mesr
from .measure_metric_deps import SLICES
from .native_oracle import prepare
from .readers import iter_dnd21

REFERENCE = 30_000
OUT = Path(__file__).resolve().parents[1] / "results/slice_ranking_dnd21.json"


def scores_by_slice(x: np.ndarray, y: np.ndarray, width: int, height: int,
                    slices: Sequence[int] = SLICES) -> Dict[int, float]:
    """MESR of one retained stream at each slice size; NaN below one complete slice."""

    return {s: (mesr(x, y, width, height, slice_size=s) if len(x) >= s else float("nan"))
            for s in slices}


def rank_agreement(table: Mapping[str, Mapping[int, float]],
                   slices: Sequence[int] = SLICES, reference: int = REFERENCE) -> Dict:
    """Kendall tau and discordant pairs of each size's ranking against the reference size.

    `table` maps method -> {slice size -> MESR}. Only methods finite at every size enter, so
    every size ranks the same set.
    """

    cohort = sorted(m for m, row in table.items()
                    if all(np.isfinite(row[s]) for s in slices))
    out: Dict = {"cohort": cohort, "by_slice": {}}
    if len(cohort) < 2:
        return out
    ref = np.array([table[m][reference] for m in cohort])
    top_ref = cohort[int(np.argmax(ref))]
    for s in slices:
        cur = np.array([table[m][s] for m in cohort])
        discordant = sum(1 for i in range(len(cohort)) for j in range(i + 1, len(cohort))
                         if np.sign(ref[i] - ref[j]) != np.sign(cur[i] - cur[j]))
        tau = kendalltau(ref, cur).statistic if len(cohort) > 2 else (
            1.0 if discordant == 0 else -1.0)
        out["by_slice"][s] = {"tau": float(tau), "discordant_pairs": discordant,
                              "pairs": len(cohort) * (len(cohort) - 1) // 2,
                              "top": cohort[int(np.argmax(cur))],
                              "top_changes": cohort[int(np.argmax(cur))] != top_ref}
    return out


def paired_flips(per_recording: Mapping[str, Mapping[str, Mapping[int, float]]],
                 slices: Sequence[int] = SLICES, reference: int = REFERENCE) -> List[Dict]:
    """Mean paired difference of every filter pair at each size, on one cohort per pair.

    `per_recording` maps recording -> method -> {slice size -> MESR}. A pair's cohort is the
    recordings on which both methods are finite at every size.
    """

    methods = sorted({m for rows in per_recording.values() for m in rows})
    out = []
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            shared = [r for r, rows in per_recording.items()
                      if a in rows and b in rows
                      and all(np.isfinite(rows[a][s]) and np.isfinite(rows[b][s])
                              for s in slices)]
            if not shared:
                continue
            diff = {s: float(np.mean([per_recording[r][a][s] - per_recording[r][b][s]
                                      for r in shared])) for s in slices}
            out.append({"a": a, "b": b, "n_recordings": len(shared), "diff": diff,
                        "flips": [s for s in slices
                                  if np.sign(diff[s]) != np.sign(diff[reference])]})
    return out


def measure(max_events: int = 1_000_000, max_recordings: Optional[int] = None,
            methods: Sequence[str] = CLASSICAL) -> Dict:
    started = time.perf_counter()
    recordings: List[Dict] = []
    per_recording: Dict[str, Dict[str, Dict[int, float]]] = {}
    for index, rec in enumerate(iter_dnd21()):
        if max_recordings is not None and index >= max_recordings:
            break
        rec, _ = prepare(rec, max_events)
        x, y = rec.events[:, 1], rec.events[:, 2]
        unfiltered = scores_by_slice(x, y, rec.width, rec.height)
        table: Dict[str, Dict[int, float]] = {}
        kept: Dict[str, int] = {}
        for method in methods:
            accepted = score_events(method, rec) == 0.0
            kept[method] = int(accepted.sum())
            table[method] = scores_by_slice(x[accepted], y[accepted], rec.width, rec.height)
        per_recording[rec.name] = table
        finite_u = [v for v in unfiltered.values() if np.isfinite(v)]
        recordings.append({
            "recording": rec.name, "kept": kept,
            "unfiltered": unfiltered,
            "unfiltered_spread": float(max(finite_u) - min(finite_u)),
            "filters": table,
            "filter_spread": {m: float(np.nanmax(list(row.values()))
                                       - np.nanmin(list(row.values())))
                              for m, row in table.items()
                              if np.isfinite(list(row.values())).sum() >= 2},
            "agreement": rank_agreement(table)})
        print(f"{rec.name}: cohort {len(recordings[-1]['agreement']['cohort'])}", flush=True)

    pairs = paired_flips(per_recording)
    return {"dataset": "dnd21", "max_events": max_events, "slices": list(SLICES),
            "reference": REFERENCE, "recordings": recordings, "pairs": pairs,
            "summary": summarise(recordings, pairs),
            "wall_seconds": time.perf_counter() - started}


def summarise(recordings: Sequence[Dict], pairs: Sequence[Dict]) -> Dict:
    """The counts the paper quotes."""

    ranked = [r for r in recordings if len(r["agreement"]["cohort"]) >= 2]
    reordered = [r["recording"] for r in ranked
                 if any(v["discordant_pairs"] for v in r["agreement"]["by_slice"].values())]
    top_changed = [r["recording"] for r in ranked
                   if any(v["top_changes"] for v in r["agreement"]["by_slice"].values())]
    taus = [v["tau"] for r in ranked for s, v in r["agreement"]["by_slice"].items()
            if s != REFERENCE]
    return {
        "recordings": len(recordings),
        "ranked_recordings": len(ranked),
        "cohort_sizes": [len(r["agreement"]["cohort"]) for r in recordings],
        "recordings_reordered": len(reordered), "reordered": reordered,
        "recordings_top_changed": len(top_changed), "top_changed": top_changed,
        "min_tau": float(min(taus)) if taus else float("nan"),
        "mean_tau": float(np.mean(taus)) if taus else float("nan"),
        "unfiltered_spread_mean": float(np.mean([r["unfiltered_spread"] for r in recordings])),
        "pairs": len(pairs),
        "pairs_flipping": [f"{p['a']}-{p['b']}@{p['flips']}" for p in pairs if p["flips"]],
        "widest_pair_gap_at_reference": max(
            (abs(p["diff"][REFERENCE]) for p in pairs), default=float("nan")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    result = measure(max_recordings=args.max_recordings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps(result["summary"], indent=2, default=float))


if __name__ == "__main__":
    main()
