"""Three properties of ESR that the paper asserted and did not measure.

Section IV-A claimed ESR "is a monotone function of how concentrated a sample is" and
explained the null's sign by hot-pixel share alone. Neither survives contact with
Eq.~(1), and this module replaces both with computations run against the released
`esr.esr`, so the paper's statements trace to an artifact like every other number in it.

**1. Monotone in concentration -- false.** Write the slice as its per-pixel count vector.
`ntss` is a sum of the convex `n(n-1)`, so it is Schur-convex: concentrating the counts can
only raise it. `ell_n = K - sum_p c^{n_p}` with `c = 1 - M/N` is a sum of the convex `c^n`
subtracted from a constant, so it is Schur-*concave*: concentrating the counts can only
lower it. ESR is the geometric mean of the two, so majorisation fixes neither factor's
product and the direction is genuinely indeterminate. `monotonicity_counterexample` exhibits
a pair where the majorising (strictly more concentrated) vector scores *lower*.

**2. Permutation invariance -- true, and provable.** Both factors read the count vector as a
multiset; neither reads pixel coordinates. So ESR is invariant under any bijection of the
pixel grid, and cannot distinguish a compact blob from the same counts scattered across the
sensor. That bounds what "structural" can mean for this metric.

**3. The sign of the null needs nonstationarity, not hot pixels.** Under independent
stationary Poisson rates, uniform thinning multiplies every rate by `r` and leaves the pixel
probabilities `lambda_p / sum lambda_q` unchanged; a fixed-count slice is multinomial with
the same parameter before and after, however concentrated the rates are. Hot-pixel share
alone therefore predicts nothing. `stationary_vs_nonstationary` runs the controlled 2x2 --
{stationary, drifting scene} x {hot-dominated, scene-dominated} -- and shows the sign is
carried by the *contrast* between a stationary hot component and a nonstationary scene
component, which is the condition the paper needs and did not state.

Writes `results/esr_properties.json`. Run:

    python -m dataset_assessment.esr_properties
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .cap_sensitivity import UNIT_OF_SCALE
from .esr import SLICE, esr, mesr

#: DAVIS346, the sensor behind E-MLB, DVSD22 and DND21's host recordings.
WIDTH, HEIGHT = 346, 260

OUT = Path(__file__).resolve().parents[1] / "results" / "esr_properties.json"


def _stream_to_xy(pixels: np.ndarray, width: int = WIDTH) -> Tuple[np.ndarray, np.ndarray]:
    """Flat pixel indices to the (x, y) pair `esr` expects."""

    return (pixels % width).astype(np.int64), (pixels // width).astype(np.int64)


def esr_of_counts(counts: Sequence[int], width: int = WIDTH,
                  height: int = HEIGHT) -> Dict[str, float]:
    """ESR of one slice given its per-pixel count vector, via the released implementation.

    The counts are laid onto distinct pixels; which ones is irrelevant by property 2.
    """

    pixels = np.concatenate([np.full(int(c), p, dtype=np.int64)
                             for p, c in enumerate(counts) if c])
    x, y = _stream_to_xy(pixels, width)
    n = len(pixels)
    m = int(n * 2 / 3)
    arr = np.asarray(counts, dtype=np.float64)
    return {
        "counts": [int(c) for c in counts],
        "n_events": n,
        "ntss": float((arr * (arr - 1)).sum() / n / (n - 1)),
        "ell_n": float(np.sum(1.0 - (1 - m / n) ** arr[arr > 0])),
        "esr": esr(x, y, width, height),
    }


def _majorises(a: Sequence[int], b: Sequence[int]) -> bool:
    """Does count vector `a` majorise `b`? Equal totals, and every partial sum at least."""

    sa = np.sort(np.asarray(a, dtype=np.int64))[::-1]
    sb = np.sort(np.asarray(b, dtype=np.int64))[::-1]
    k = max(len(sa), len(sb))
    sa = np.pad(sa, (0, k - len(sa)))
    sb = np.pad(sb, (0, k - len(sb)))
    return bool(sa.sum() == sb.sum() and np.all(np.cumsum(sa) >= np.cumsum(sb)))


def monotonicity_counterexample() -> Dict:
    """A strictly more concentrated slice that scores strictly lower.

    Both vectors hold N = 30,000 events, the slice size the benchmark uses, so this is not a
    small-sample artefact. `[30000]` majorises `[15000, 7500, 7500]` -- it is more
    concentrated in the only order that makes the word precise -- and scores below it.
    """

    concentrated = esr_of_counts([30_000])
    spread = esr_of_counts([15_000, 7_500, 7_500])
    return {
        "concentrated": concentrated,
        "spread": spread,
        "concentrated_majorises_spread": _majorises(concentrated["counts"],
                                                    spread["counts"]),
        "esr_gap": spread["esr"] - concentrated["esr"],
        "ntss_favours_concentrated": concentrated["ntss"] > spread["ntss"],
        "ell_n_favours_spread": spread["ell_n"] > concentrated["ell_n"],
    }


def permutation_invariance(trials: int = 200, seed: int = 0) -> Dict:
    """ESR before and after a random bijection of the pixel grid, on random slices."""

    rng = np.random.default_rng(seed)
    k = WIDTH * HEIGHT
    worst = 0.0
    for _ in range(trials):
        pixels = rng.integers(0, k, size=SLICE)
        # A concentrated draw too: uniform pixels alone would not exercise a blob.
        if rng.random() < 0.5:
            pixels = rng.integers(0, 400, size=SLICE) + rng.integers(0, k - 400)
        perm = rng.permutation(k)
        a = esr(*_stream_to_xy(pixels), WIDTH, HEIGHT)
        b = esr(*_stream_to_xy(perm[pixels]), WIDTH, HEIGHT)
        worst = max(worst, abs(a - b))
    return {"trials": trials, "max_abs_difference": float(worst),
            "note": "zero to floating point; both factors read the count multiset only"}


def _synthetic_stream(rng: np.random.Generator, length: int, hot_share: float,
                      drifting: bool, n_hot: int = 20, span: int = 500) -> np.ndarray:
    """An ordered event stream: a stationary hot component plus a scene component.

    The scene component occupies `span` contiguous pixels at any moment. When `drifting`, its
    support sweeps the sensor over the stream, so a slice spanning a longer stretch of the
    stream sees a proportionally larger scene support -- the nonstationary case. When not, the
    support is fixed and the whole stream is i.i.d., which is the stationary Poisson case the
    thinning argument must survive and does not distinguish.
    """

    k = WIDTH * HEIGHT
    hot_pixels = rng.choice(k, size=n_hot, replace=False)
    is_hot = rng.random(length) < hot_share
    out = np.empty(length, dtype=np.int64)
    out[is_hot] = rng.choice(hot_pixels, size=int(is_hot.sum()))

    n_scene = int((~is_hot).sum())
    position = np.nonzero(~is_hot)[0]
    if drifting:
        base = (position / max(length - 1, 1) * (k - span)).astype(np.int64)
    else:
        base = np.full(n_scene, (k - span) // 2, dtype=np.int64)
    out[~is_hot] = base + rng.integers(0, span, size=n_scene)
    return out


def _thin(rng: np.random.Generator, stream: np.ndarray, r: float) -> np.ndarray:
    """Uniform random discard carrying no information about which events matter."""

    if r >= 1.0:
        return stream
    keep = rng.random(len(stream)) < r
    return stream[keep]


def _sign(deltas: Dict[str, float]) -> str:
    """Does thinning raise, lower or not move MESR, at the resolution the paper claims?"""

    extreme = max(deltas.values(), key=abs)
    if abs(extreme) <= UNIT_OF_SCALE:
        return "flat"
    return "rises" if extreme > 0 else "falls"


def stationary_vs_nonstationary(length: int = 3_000_000,
                                retentions: Sequence[float] = (1.0, 0.5, 0.25, 0.10),
                                hot_shares: Sequence[float] = (0.20, 0.01),
                                seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> Dict:
    """Does the nonselective null raise MESR? Measured over the controlled 2x2.

    Hot share is held fixed down each column, so any difference between the stationary and
    drifting rows is attributable to the scene component's stationarity alone -- which is the
    contrast the paper's mechanism needs and hot-pixel share cannot supply.
    """

    cells: List[Dict] = []
    for drifting in (False, True):
        for hot_share in hot_shares:
            by_r: Dict[str, List[float]] = {f"r={r:.2f}": [] for r in retentions}
            for seed in seeds:
                rng = np.random.default_rng(seed)
                stream = _synthetic_stream(rng, length, hot_share, drifting)
                for r in retentions:
                    kept = _thin(np.random.default_rng(seed + 10_000), stream, r)
                    x, y = _stream_to_xy(kept)
                    by_r[f"r={r:.2f}"].append(mesr(x, y, WIDTH, HEIGHT))
            means = {k: float(np.mean(v)) for k, v in by_r.items()}
            sds = {k: float(np.std(v, ddof=1)) for k, v in by_r.items()}
            base = means[f"r={max(retentions):.2f}"]
            deltas = {k: v - base for k, v in means.items()}
            cells.append({
                "scene": "drifting (nonstationary)" if drifting else "fixed (stationary)",
                "hot_share": hot_share,
                "mesr_mean": means, "mesr_sd": sds,
                "delta_over_unthinned": deltas,
                "max_delta": float(max(deltas.values())),
                "min_delta": float(min(deltas.values())),
                # Judged against the paper's own unit of scale rather than against zero: a
                # move the published EDformer-EDmamba gap could not resolve is not a sign.
                "sign": _sign(deltas),
            })
    return {"length": length, "retentions": list(retentions), "seeds": list(seeds),
            "n_hot_pixels": 20, "scene_span_pixels": 500, "cells": cells}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--length", type=int, default=3_000_000)
    args = parser.parse_args()

    result = {
        "monotonicity": monotonicity_counterexample(),
        "permutation_invariance": permutation_invariance(),
        "stationarity": stationary_vs_nonstationary(length=args.length),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float) + "\n")

    m = result["monotonicity"]
    print(f"monotonicity: ESR{tuple(m['concentrated']['counts'])} = "
          f"{m['concentrated']['esr']:.5f} vs ESR{tuple(m['spread']['counts'])} = "
          f"{m['spread']['esr']:.5f}  (majorises: "
          f"{m['concentrated_majorises_spread']})")
    print(f"permutation invariance: max |dESR| over "
          f"{result['permutation_invariance']['trials']} trials = "
          f"{result['permutation_invariance']['max_abs_difference']:.3g}")
    print("stationarity 2x2 (Delta MESR against the unthinned stream):")
    for c in result["stationarity"]["cells"]:
        row = "  ".join(f"{k} {v:+.4f}" for k, v in c["delta_over_unthinned"].items())
        print(f"  {c['scene']:26s} hot={c['hot_share']:.2f}  {row}   -> {c['sign']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
