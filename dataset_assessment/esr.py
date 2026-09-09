"""Event Structural Ratio — the official cuke-emlb definition, verified bit-exact.

Ported from `native3d_ssm/phase5_emlb_mesr.py`, which was checked against
`cuke-emlb/python/src/utils/metric.py:109 EventStructuralRatio._calc_esr` (maxdiff 0.00e+00)
and against EDformer's vendored copy under its pinned dv-processing 1.7.9 (maxdiff 1.4e-07,
float32 accumulator precision).

Do NOT substitute the `EventStructuralRatioV2` variant that lives at `:24` in the same
cuke-emlb file (and again as `EDformer/metrics.py`): it median-filters the potential surface,
divides `ln` by K, sums n^2 rather than n(n-1), and returns 1000*sqrt(ntss*ln). This was once
described here as "numbers ~1000x off", which is wrong -- the /K cancels most of the 1000,
and what is left is the median filter. `esr_variant.py` measures it: the variant runs from
3.4x the official value on E-MLB down to *zero* on the sparse DVSD22 slices, where the size-3
median empties the surface. It is not a rescaling and there is no factor to divide out.

On K: `ln = K - sum_all_px (1-M/N)^n`. An empty pixel has n = 0 and therefore contributes
exactly 1, so K cancels and

    ln = sum_occupied_px [1 - (1-M/N)^n]  <=  #occupied pixels.

ESR is consequently *exactly* invariant to the declared sensor size (asserted in
`tests/test_esr.py::test_esr_is_invariant_to_the_declared_sensor_size`). The only way W*H
enters is as a ceiling on the occupied-pixel count, which binds only when the sensor is small
enough to saturate, K <~ N_slice = 30,000. See `protocol.sensor_saturation_ratio`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

SLICE = 30_000
BLOCK = 10_240


def esr(x: np.ndarray, y: np.ndarray, width: int, height: int) -> float:
    """ESR of one slice. NaN when fewer than 2 events."""

    n_events = len(x)
    if n_events < 2:
        return float("nan")
    pixels = width * height
    counts = np.bincount(y.astype(np.int64) * width + x.astype(np.int64),
                         minlength=pixels).astype(np.float64)
    m = int(n_events * 2 / 3)
    eps = np.spacing(1)
    ntss = (counts * (counts - 1)).sum() / (n_events + eps) / (n_events - 1 + eps)
    ln = pixels - ((1 - m / n_events) ** counts).sum()
    return float(np.sqrt(ntss * ln))


def mesr(x: np.ndarray, y: np.ndarray, width: int, height: int,
         slice_size: int = SLICE) -> float:
    """Mean ESR over consecutive complete slices; incomplete tail dropped."""

    scores = [esr(x[i:i + slice_size], y[i:i + slice_size], width, height)
              for i in range(0, len(x) - slice_size + 1, slice_size)]
    return float(np.mean(scores)) if scores else float("nan")


def retain_mask(scores: np.ndarray, retention: float, block: int = BLOCK) -> np.ndarray:
    """Keep the lowest-scoring `retention` fraction within each block."""

    keep = np.zeros(len(scores), dtype=bool)
    for start in range(0, len(scores), block):
        chunk = scores[start:start + block]
        if len(chunk) == 0:
            continue
        n_keep = max(1, int(round(len(chunk) * retention)))
        order = np.argsort(chunk, kind="stable")[:n_keep]
        keep[start + order] = True
    return keep


def mesr_curve(scores: np.ndarray, x: np.ndarray, y: np.ndarray,
               width: int, height: int, retentions: Sequence[float],
               block: int = BLOCK, slice_size: int = SLICE) -> List[Dict]:
    """MESR at each retention, with an explicit evaluability flag.

    `evaluable=False` means the retained stream held fewer than one complete slice, so MESR
    is undefined there. Silently dropping these points makes an optimum look pinned to the
    grid floor when it is only pinned to the *measurable* floor.
    """

    out: List[Dict] = []
    for retention in retentions:
        keep = retain_mask(scores, float(retention), block)
        kept = int(keep.sum())
        value = mesr(x[keep], y[keep], width, height, slice_size)
        out.append({"r": float(retention), "mesr": value, "kept_events": kept,
                    "evaluable": bool(np.isfinite(value))})
    return out
