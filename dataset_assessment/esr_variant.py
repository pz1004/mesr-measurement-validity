"""How far apart are the two ESR classes the field ships?

`cuke-emlb/python/src/utils/metric.py` defines two classes. `EventStructuralRatio` at `:109`
is the official one, and `esr.esr` reproduces it bit-exact. `EventStructuralRatioV2` at `:24`
sits above it in the same file, and EDformer's repository ships the same variant beside its
own metric. They differ only in class name, so selecting the wrong one is a plausible
mistake, and neither release warns about it.

The paper asserted that the variant returns values "three orders of magnitude out", reading
the `1000 *` in its return statement as the whole story. It is not. The variant makes four
changes at once:

    ntss = (n*n).sum() / (N*N)          official: n(n-1) summed, divided by N(N-1)
    ln   = (K - (0.5**n).sum()) / K     official: no /K, and base (1 - M/N) ~ 1/3
    n    = median_filter(surface, 3)    official: the raw count surface
    return 1000 * sqrt(ntss * ln)       official: sqrt(ntss * ln)

The `/K` alone cancels most of the `1000` -- their product is `1000/sqrt(K)`, which is
`3.33` on a 346x260 sensor and `1.04` on a 1280x720 one -- and the size-3 median filter then
erases every isolated pixel of a sparse slice, which no closed form covers. So the ratio is
data-dependent and has to be measured. It is measured here, on real slices from every corpus
the paper uses.

The answer matters for how the hazard should be described. A variant that returned 1000x
would be caught by anyone who looked at the number. One that returns the same order of
magnitude will not be.

Writes `results/esr_variant.json`. Run:

    python -m dataset_assessment.esr_variant
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from numpy.lib.stride_tricks import as_strided

from .esr import SLICE, esr
from .run_benchmark import CAPPED_LOADERS, LOADERS

OUT = Path(__file__).resolve().parents[1] / "results" / "esr_variant.json"

#: Corpora to sample, in the order the paper introduces them.
CORPORA = ("dnd21", "dvsclean", "emlb", "pure_ba", "dvsd22")


def median_filter(data: np.ndarray, size: int = 3) -> np.ndarray:
    """Verbatim from `cuke-emlb/python/src/utils/metric.py:13`, including its constant pad."""

    temp = np.pad(data, [(size // 2, size // 2)], mode="constant")
    view = as_strided(temp, shape=(data.shape[0], data.shape[1], size, size),
                      strides=(temp.strides[0], temp.strides[1],
                               temp.strides[0], temp.strides[1]))
    return np.median(view, axis=(2, 3))


def esr_v2(x: np.ndarray, y: np.ndarray, width: int, height: int) -> float:
    """`EventStructuralRatioV2._calc_esr`, on the accumulator surface it would have built.

    The accumulator is configured with unit contribution, no decay and unbounded potential
    (`metric.py:29-33`), so its potential surface is exactly the per-pixel event count.
    """

    surface = np.zeros((height, width), dtype=np.float64)
    np.add.at(surface, (y.astype(np.int64), x.astype(np.int64)), 1.0)
    n = median_filter(surface, size=3)
    n_events = len(x)
    pixels = width * height
    ntss = (n * n).sum() / (n_events * n_events)
    ln = (pixels - (0.5 ** n).sum()) / pixels
    return float(1000 * np.sqrt(ntss * ln))


def ratios_for_recording(x: np.ndarray, y: np.ndarray, width: int, height: int,
                         max_slices: int, slice_size: int = SLICE) -> List[Dict]:
    """Both classes on the same complete slices, so nothing but the formula differs."""

    out: List[Dict] = []
    for start in range(0, len(x) - slice_size + 1, slice_size):
        if len(out) >= max_slices:
            break
        xs, ys = x[start:start + slice_size], y[start:start + slice_size]
        official = esr(xs, ys, width, height)
        variant = esr_v2(xs, ys, width, height)
        if np.isfinite(official) and official > 0 and np.isfinite(variant):
            out.append({"official": official, "variant": variant,
                        "ratio": variant / official})
    return out


def survey(corpora: Sequence[str] = CORPORA, recordings: int = 3, slices: int = 4,
           max_events: int = 200_000) -> Dict:
    """Per-corpus variant/official ratio, on real slices at each corpus's own sensor."""

    per_corpus: Dict[str, Dict] = {}
    for name in corpora:
        loader = LOADERS[name]
        kwargs = {"max_events": max_events} if name in CAPPED_LOADERS else {}
        rows: List[Dict] = []
        sensor = None
        scored = 0
        for index, rec in enumerate(loader(**kwargs)):
            # Pure_BA's series starts with recordings shorter than one 30,000-event slice,
            # so count recordings that yield a slice rather than recordings seen.
            if scored >= recordings or index >= 40:
                break
            cells = ratios_for_recording(rec.x[:max_events], rec.y[:max_events],
                                         rec.width, rec.height, slices)
            if not cells:
                continue
            sensor = [rec.width, rec.height]
            scored += 1
            rows.extend({"recording": rec.name, **row} for row in cells)
        if not rows:
            continue
        ratios = np.array([r["ratio"] for r in rows])
        pixels = sensor[0] * sensor[1]
        per_corpus[name] = {
            "sensor": sensor,
            "slices": len(rows),
            "ratio": {"median": float(np.median(ratios)),
                      "min": float(ratios.min()), "max": float(ratios.max())},
            # The closed form that would hold if the /K and the x1000 were the only
            # differences. It is not the answer, and showing it says why.
            "thousand_over_sqrt_k": float(1000 / np.sqrt(pixels)),
            "cells": rows,
        }
    every = np.array([r["ratio"] for c in per_corpus.values() for r in c["cells"]])
    return {
        "corpora": per_corpus,
        "overall": {"slices": int(every.size),
                    "min": float(every.min()), "max": float(every.max()),
                    "median": float(np.median(every))},
        # The variant is not a rescaling: it overshoots on dense slices and collapses on
        # sparse ones, so the error has neither a fixed sign nor a fixed size. On the
        # sparsest corpus the size-3 median filter empties the surface outright and the
        # variant returns exactly zero where the official class returns its largest values.
        "collapsed_to_zero": int((every == 0).sum()),
        "spread_in_orders_of_magnitude": (
            float(np.log10(every.max() / every[every > 0].min()))
            if (every > 0).any() else float("nan")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--recordings", type=int, default=3)
    parser.add_argument("--slices", type=int, default=4)
    args = parser.parse_args()

    result = survey(recordings=args.recordings, slices=args.slices)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float) + "\n")

    print(f"{'corpus':10s} {'sensor':>12s} {'slices':>7s} "
          f"{'variant/official':>18s} {'1000/sqrt(K)':>13s}")
    for name, c in result["corpora"].items():
        r = c["ratio"]
        print(f"{name:10s} {str(c['sensor']):>12s} {c['slices']:7d} "
              f"{r['median']:8.3f} [{r['min']:.3f}, {r['max']:.3f}] "
              f"{c['thousand_over_sqrt_k']:13.3f}")
    o = result["overall"]
    print(f"\noverall over {o['slices']} slices: {o['min']:.5f} to {o['max']:.3f}, "
          f"median {o['median']:.3f}")
    print(f"the variant is not a rescaling: over the nonzero slices its error spans "
          f"{result['spread_in_orders_of_magnitude']:.1f} orders of magnitude and changes "
          f"direction with slice density; it returns exactly zero on "
          f"{result['collapsed_to_zero']} of {o['slices']} slices")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
