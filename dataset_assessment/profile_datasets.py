"""Does a method-independent statistic separate synthetic from real noise?

This is the paper's dataset-assessment claim. It is a HYPOTHESIS TEST, not a demo: if the
statistics do not separate, say so and drop section 5 rather than tuning statistics until
they do.

Corpora, all capped at CAP events so no single recording dominates:

  synthetic  DND21 (346x260, injected 1-10 Hz/px), DVSCLEAN (1280x720, simulated)
  real       E-MLB (346x260, ND-filtered scenes), DVSD22 (346x260, falling drops),
             Pure_BA (346x260, signal-free captures - real background activity with no
             signal to confound the statistics)

DVSD22 and Pure_BA were not in the original plan. They matter because without them the
"real" group is E-MLB alone, and a two-group test with one real corpus cannot distinguish
"synthetic differs from real" from "synthetic differs from E-MLB".

Every corpus is profiled in full. Earlier revisions capped E-MLB at 40 recordings and
Pure_BA at 20 for speed, and neither cap was a neutral subsample: corpora are iterated in
name order, so E-MLB's first 40 are 40 daylight recordings and no night ones, and Pure_BA's
first 20 are the lowest 20 grades of a monotonic background-activity gradient, understating
its event rate by 43x. A `limit=` here silently selects a stratum, so there is none. The
whole run costs ~220 s.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .readers import (iter_dnd21, iter_dvsd22, iter_dvsclean, iter_emlb,
                      iter_pure_ba_noise)
from .regime import regime_profile

CAP = 300_000
STATS = ("spatial_autocorr", "local_global_coupling", "hot_pixel_share", "event_rate_per_px")
OUT = Path(__file__).resolve().parents[1] / "results/noise_regimes.json"


def _capped(recording):
    events = recording.events[:CAP]
    labels = None if recording.labels is None else recording.labels[:len(events)]
    return replace(recording, events=events, labels=labels)


def _collect(corpus: str, iterator, limit: int | None = None):
    profiles = []
    for index, rec in enumerate(iterator):
        if limit is not None and index >= limit:
            break
        profile = regime_profile(_capped(rec))
        profile["corpus"] = corpus
        profiles.append(profile)
        print(f"  [{corpus} {index}] {rec.name}: "
              f"autocorr={profile['spatial_autocorr']:.4f} "
              f"coupling={profile['local_global_coupling']:.4f}", flush=True)
    return profiles


def _two_group_test(synth, real, stat: str) -> dict:
    """Mann-Whitney U plus rank-biserial effect size for one statistic."""

    from scipy.stats import mannwhitneyu

    s = np.array([p[stat] for p in synth], dtype=float)
    r = np.array([p[stat] for p in real], dtype=float)
    s, r = s[np.isfinite(s)], r[np.isfinite(r)]
    if len(s) < 2 or len(r) < 2:
        return {"synthetic_n": int(len(s)), "real_n": int(len(r)),
                "p_value": float("nan"), "rank_biserial": float("nan"),
                "separates": False}
    u = mannwhitneyu(s, r, alternative="two-sided")
    # rank-biserial effect size = 1 - 2U/(n1*n2); |.|=1 means perfect separation
    effect = 1 - 2 * u.statistic / (len(s) * len(r))
    return {"synthetic_mean": float(s.mean()), "real_mean": float(r.mean()),
            "synthetic_n": int(len(s)), "real_n": int(len(r)),
            "p_value": float(u.pvalue), "rank_biserial": float(effect),
            "separates": bool(u.pvalue < 0.01 and abs(effect) > 0.8)}


def _leave_one_corpus_out(profiles) -> dict:
    """Re-run every test with each corpus withheld in turn.

    The verdict is a two-group comparison over five corpora of very unequal size and
    character, so "does any single corpus carry the result?" is a question a reader will
    ask. Answering it here means the answer is an artifact rather than a claim: the paper
    quotes these numbers instead of a hand-computed sensitivity check.
    """

    out: dict = {}
    for dropped in sorted({p["corpus"] for p in profiles}):
        kept = [p for p in profiles if p["corpus"] != dropped]
        synth = [p for p in kept if p["synthetic"]]
        real = [p for p in kept if not p["synthetic"]]
        if not synth or not real:
            continue
        out[dropped] = {stat: _two_group_test(synth, real, stat) for stat in STATS}
    return out


def main() -> None:
    started = time.perf_counter()
    profiles = []
    profiles += _collect("DND21", iter_dnd21())
    profiles += _collect("DVSCLEAN", iter_dvsclean())
    profiles += _collect("E-MLB", iter_emlb(max_events=CAP))
    profiles += _collect("DVSD22", iter_dvsd22(max_events=CAP))
    profiles += _collect("Pure_BA", iter_pure_ba_noise(max_events=CAP))

    synth = [p for p in profiles if p["synthetic"]]
    real = [p for p in profiles if not p["synthetic"]]

    verdict = {stat: _two_group_test(synth, real, stat) for stat in STATS}
    sensitivity = _leave_one_corpus_out(profiles)

    per_corpus = {}
    for corpus in sorted({p["corpus"] for p in profiles}):
        rows = [p for p in profiles if p["corpus"] == corpus]
        per_corpus[corpus] = {"n": len(rows), "synthetic": rows[0]["synthetic"],
                              **{stat: float(np.nanmean([r[stat] for r in rows]))
                                 for stat in STATS}}

    payload = {"cap_events": CAP, "profiles": profiles, "per_corpus": per_corpus,
               "verdict": verdict, "leave_one_corpus_out": sensitivity,
               "separation_bar": {"p_value": 0.01, "abs_rank_biserial": 0.8,
                                  "registered": "before the run, in the task plan"},
               "wall_seconds": time.perf_counter() - started}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=float) + "\n")

    print("\n=== per corpus ===")
    for corpus, row in per_corpus.items():
        print(f"  {corpus:<11s} n={row['n']:<3d} synthetic={str(row['synthetic']):<5s} "
              + "  ".join(f"{s}={row[s]:.4f}" for s in STATS))
    print("\n=== synthetic vs real ===")
    for stat, row in verdict.items():
        print(f"  {stat:<22s} synth={row['synthetic_mean']:.4f} real={row['real_mean']:.4f} "
              f"p={row['p_value']:.3e} rank_biserial={row['rank_biserial']:+.3f} "
              f"separates={row['separates']}")
    print("\n=== leave one corpus out (does any single corpus carry the result?) ===")
    for dropped, rows in sensitivity.items():
        best = max(rows.items(), key=lambda kv: abs(kv[1]["rank_biserial"]))
        print(f"  without {dropped:<11s} strongest={best[0]:<22s} "
              f"p={best[1]['p_value']:.3e} rank_biserial={best[1]['rank_biserial']:+.3f} "
              f"separates={best[1]['separates']}")
    print(f"\nn_synthetic={len(synth)} n_real={len(real)}  wall={payload['wall_seconds']:.1f}s")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
