"""Quantify what MESR depends on besides denoising quality (paper section 3).

Three dependencies, each a separate experiment:

1. RETENTION. For one fixed ranking, sweep r and report the full MESR@r curve. The spread
   of this curve is an upper bound on how much a paper's unstated operating point can move
   its reported number.
2. RESOLUTION. This experiment was written to confirm that `ln = K - sum(1-M/N)^n` grows
   with `K = W*H`. It does not: empty pixels contribute exactly 1 each, K cancels, and ESR
   is invariant to the declared sensor size. The experiment is kept because that invariance
   is the evidence, and it is *measured* here rather than asserted. What actually differs
   across sensors is occupancy, so the second half of the experiment decomposes real 346x260
   and real 1280x720 streams into (ntss, ln, occupied_px) to show where the gap comes from.
3. SLICE SIZE. The 30,000-event slicing is a convention; report sensitivity to it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .esr import mesr, mesr_curve
from .protocol import decompose_esr
from .readers import iter_dnd21, iter_dvsclean

RETENTIONS = np.round(np.arange(0.05, 1.001, 0.05), 3)
SLICES = (10_000, 20_000, 30_000, 50_000, 100_000)
OUT = Path(__file__).resolve().parents[1] / "results/metric_dependencies.json"


def rank_by_label(labels: np.ndarray, rng: np.random.Generator, target_auc: float = 0.93):
    """A synthetic ranking with a KNOWN AUC, so retention is the only free variable.

    For two unit-variance Gaussians separated by `d`, AUC = Phi(d/sqrt(2)), so the shift
    that realises `target_auc` is d = sqrt(2) * Phi^-1(target_auc). Verified by assertion
    rather than asserted by comment.
    """

    from scipy.stats import norm
    from sklearn.metrics import roc_auc_score

    noise = labels == 1
    scores = rng.normal(0.0, 1.0, len(labels))
    scores[noise] += np.sqrt(2) * norm.ppf(target_auc)
    realised = roc_auc_score(labels, scores)
    assert abs(realised - target_auc) < 0.01, f"AUC {realised:.4f} != {target_auc}"
    return scores


def _spread_summary(spreads, draws: int = 10_000, seed: int = 20260726) -> dict:
    """Mean per-recording spread with a percentile bootstrap interval over recordings.

    An influence quantity needs an uncertainty statement, which a single recording cannot
    give. `n` is the number of recordings that contributed, not the number of grid points.
    """

    array = np.asarray([s for s in spreads if np.isfinite(s)], dtype=float)
    if len(array) < 2:
        return {"mean": float(array[0]) if len(array) else float("nan"),
                "lo": float("nan"), "hi": float("nan"), "n": int(len(array))}
    generator = np.random.default_rng(seed)
    means = generator.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return {"mean": float(array.mean()), "sd": float(array.std(ddof=1)),
            "min": float(array.min()), "max": float(array.max()),
            "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)),
            "n": int(len(array)), "draws": int(draws)}


def main() -> None:
    rng = np.random.default_rng(0)
    payload = {}

    # 1. Retention dependence, on a labeled dataset with a fixed ranking.
    #
    # Run over EVERY DND21 recording, not one. An influence quantity quoted from a single
    # recording is an anecdote: it carries no uncertainty and a reader cannot tell whether
    # the next recording would halve it. The per-recording spreads are bootstrapped below
    # and the paper quotes the mean with its interval.
    # The first recording's curve is computed inside this loop and reused below, rather
    # than recomputed afterwards: a second `rank_by_label` call would draw from an rng the
    # loop has already advanced, silently changing a published number (it moved 0.4256 ->
    # 0.4283 when this was written the other way).
    recordings = list(iter_dnd21())
    per_recording, curves = [], {}
    for rec_i in recordings:
        scores_i = rank_by_label(rec_i.labels, rng)
        curve_i = mesr_curve(scores_i, rec_i.x, rec_i.y, rec_i.width, rec_i.height,
                             RETENTIONS)
        curves[rec_i.name] = curve_i
        usable_i = [c for c in curve_i if c["evaluable"]]
        if not usable_i:
            continue
        lo = min(c["mesr"] for c in usable_i)
        hi = max(c["mesr"] for c in usable_i)
        per_recording.append({
            "recording": rec_i.name, "min_mesr": lo, "max_mesr": hi, "spread": hi - lo,
            "argmax_r": max(usable_i, key=lambda c: c["mesr"])["r"],
            "argmin_r": min(usable_i, key=lambda c: c["mesr"])["r"],
        })

    rec = recordings[0]
    curve = curves[rec.name]
    usable = [c for c in curve if c["evaluable"]]
    spreads = [r["spread"] for r in per_recording]
    payload["retention_dependence"] = {
        "recording": rec.name,
        "fixed_ranking_auc": 0.93,
        "curve": curve,
        "min_mesr": min(c["mesr"] for c in usable),
        "max_mesr": max(c["mesr"] for c in usable),
        "spread": max(c["mesr"] for c in usable) - min(c["mesr"] for c in usable),
        "argmax_r": max(usable, key=lambda c: c["mesr"])["r"],
        "per_recording": per_recording,
        "spread_over_recordings": _spread_summary(spreads),
        "note": ("Spread is an upper bound on how far an unstated operating point can move a "
                 "published MESR for a FIXED denoiser quality. `spread_over_recordings` is "
                 "the quantity the paper quotes; the single-recording `spread` above is kept "
                 "for continuity with the curve stored beside it."),
    }

    # 2a. Declared-resolution invariance: identical events, different declared sensor size.
    sub = rec.events[:120_000]
    res_rows = []
    for width, height in ((346, 260), (640, 480), (1280, 720), (4096, 4096)):
        inside = (sub[:, 1] < width) & (sub[:, 2] < height)
        res_rows.append({"width": width, "height": height, "K": width * height,
                         "mesr": mesr(sub[inside, 1], sub[inside, 2], width, height),
                         "events": int(inside.sum())})
    values = np.array([r["mesr"] for r in res_rows], dtype=float)
    spread = float(values.max() - values.min())

    # 2b. Where a cross-sensor difference actually comes from: occupancy, not K.
    dnd21_slice = rec.events[:30_000]
    hd = next(iter_dvsclean())
    hd_slice = hd.events[:30_000]
    occupancy_rows = [
        {"stream": rec.name, "sensor": [rec.width, rec.height],
         "esr": float(np.sqrt(decompose_esr(dnd21_slice[:, 1], dnd21_slice[:, 2],
                                            rec.width, rec.height)["ntss"]
                              * decompose_esr(dnd21_slice[:, 1], dnd21_slice[:, 2],
                                              rec.width, rec.height)["ln"])),
         **decompose_esr(dnd21_slice[:, 1], dnd21_slice[:, 2], rec.width, rec.height)},
        {"stream": hd.name, "sensor": [hd.width, hd.height],
         "esr": float(np.sqrt(decompose_esr(hd_slice[:, 1], hd_slice[:, 2],
                                            hd.width, hd.height)["ntss"]
                              * decompose_esr(hd_slice[:, 1], hd_slice[:, 2],
                                              hd.width, hd.height)["ln"])),
         **decompose_esr(hd_slice[:, 1], hd_slice[:, 2], hd.width, hd.height)},
    ]
    for row in occupancy_rows:
        row["occupied_px_share_of_K"] = row["occupied_px"] / (row["sensor"][0] * row["sensor"][1])
        row["ln_over_occupied"] = row["ln"] / row["occupied_px"] if row["occupied_px"] else float("nan")

    payload["resolution_dependence"] = {
        "declared_sensor_rows": res_rows,
        "declared_sensor_spread": spread,
        "declared_sensor_invariant": bool(spread < 1e-9),
        "occupancy_rows": occupancy_rows,
        "note": ("2a: same events, same denoising, four declared sensor sizes. Any nonzero "
                 "spread would be a metric artefact. Measured spread is reported above; K "
                 "cancels analytically because empty pixels contribute (1-M/N)^0 = 1. "
                 "2b: real 346x260 vs real 1280x720 slices of 30,000 events, decomposed. "
                 "The cross-sensor gap lives in ntss/ln/occupancy, i.e. in the data, not in "
                 "a W*H term in the metric."),
    }

    # 3. Slice-size sensitivity, again over every recording rather than one.
    slice_per_recording = []
    for rec_i in recordings:
        row_values = [mesr(rec_i.x, rec_i.y, rec_i.width, rec_i.height, slice_size=s)
                      for s in SLICES]
        finite = [v for v in row_values if np.isfinite(v)]
        if len(finite) < 2:
            continue
        slice_per_recording.append({
            "recording": rec_i.name,
            "rows": [{"slice": s, "mesr": v} for s, v in zip(SLICES, row_values)],
            "spread": float(max(finite) - min(finite)),
            "monotone_increasing": bool(all(a <= b for a, b in zip(finite, finite[1:]))),
        })

    payload["slice_dependence"] = {
        "recording": rec.name,
        "rows": [{"slice": s,
                  "mesr": mesr(rec.x, rec.y, rec.width, rec.height, slice_size=s)}
                 for s in SLICES],
        "per_recording": slice_per_recording,
        "spread_over_recordings": _spread_summary(
            [r["spread"] for r in slice_per_recording]),
        "monotone_increasing_share": (
            sum(r["monotone_increasing"] for r in slice_per_recording)
            / len(slice_per_recording) if slice_per_recording else float("nan")),
    }
    slice_values = np.array([r["mesr"] for r in payload["slice_dependence"]["rows"]],
                            dtype=float)
    payload["slice_dependence"]["spread"] = float(np.nanmax(slice_values)
                                                  - np.nanmin(slice_values))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=float) + "\n")

    print("retention spread : %.4f  (argmax r = %.2f, min %.4f, max %.4f)" % (
        payload["retention_dependence"]["spread"],
        payload["retention_dependence"]["argmax_r"],
        payload["retention_dependence"]["min_mesr"],
        payload["retention_dependence"]["max_mesr"]))
    print("declared-resolution spread : %.3e  (invariant = %s)" % (
        spread, payload["resolution_dependence"]["declared_sensor_invariant"]))
    for row in occupancy_rows:
        print("  %-26s K=%-8d occupied=%-7d (%.3f of K)  ntss=%.5f  ln=%8.1f  esr=%.4f" % (
            row["stream"], row["sensor"][0] * row["sensor"][1], row["occupied_px"],
            row["occupied_px_share_of_K"], row["ntss"], row["ln"], row["esr"]))
    print("slice-size spread : %.4f" % payload["slice_dependence"]["spread"])
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
