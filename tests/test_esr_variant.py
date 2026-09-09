"""The paper said the unused ESR variant returns values "three orders of magnitude out".

It does not, and the reason it does not is the point. The variant makes four changes at once
-- a size-3 median filter on the count surface, `ln` divided by `K`, `n^2` in place of
`n(n-1)`, and a `1000` multiplier -- and the `/K` cancels most of the `1000`. What is left is
the median filter, which erases every isolated pixel. So the variant *overshoots* on dense
slices and *collapses* on sparse ones, and there is no fixed factor at all.

These tests exist so the corrected claim cannot silently regress to the old one. The old
claim was wrong about a third party's released code, printed in the article body.
"""

import numpy as np
import pytest

from dataset_assessment.esr import esr
from dataset_assessment.esr_variant import esr_v2, median_filter

WIDTH, HEIGHT = 346, 260


def _naive_median_filter(data, size=3):
    """Reference for `median_filter`, written the slow obvious way."""

    pad = size // 2
    padded = np.pad(data, pad, mode="constant")
    out = np.zeros_like(data, dtype=float)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            out[i, j] = np.median(padded[i:i + size, j:j + size])
    return out


def _surface_to_xy(surface):
    ys, xs = np.nonzero(surface)
    x = np.repeat(xs, surface[ys, xs].astype(int))
    y = np.repeat(ys, surface[ys, xs].astype(int))
    return x.astype(np.int64), y.astype(np.int64)


def test_median_filter_matches_a_naive_reference():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 5, size=(12, 15)).astype(np.float64)
    assert np.array_equal(median_filter(data), _naive_median_filter(data))


def test_median_filter_erases_an_isolated_pixel():
    """The mechanism behind the collapse: 8 of 9 neighbours are empty, so the median is 0."""

    data = np.zeros((5, 5))
    data[2, 2] = 900.0
    assert median_filter(data).sum() == 0.0


def test_variant_reproduces_the_released_formula():
    """Hand-computed against `cuke-emlb/python/src/utils/metric.py:100-107`."""

    surface = np.zeros((HEIGHT, WIDTH))
    surface[10:14, 10:14] = 3.0                       # a block that survives the filter
    x, y = _surface_to_xy(surface)
    filtered = median_filter(surface)
    pixels = WIDTH * HEIGHT
    expected = 1000 * np.sqrt(((filtered * filtered).sum() / (len(x) ** 2))
                              * ((pixels - (0.5 ** filtered).sum()) / pixels))
    assert esr_v2(x, y, WIDTH, HEIGHT) == pytest.approx(expected, rel=1e-12)


def test_the_variant_is_not_a_constant_multiple_of_the_official_class():
    """The claim the paper now makes, and the one it used to make, in one assertion."""

    ratios = []
    rng = np.random.default_rng(7)
    for density in (0.02, 0.2):                       # sparse and dense slices
        occupied = max(1, int(WIDTH * HEIGHT * density))
        surface = np.zeros((HEIGHT, WIDTH))
        flat = rng.choice(WIDTH * HEIGHT, size=occupied, replace=False)
        surface.flat[flat] = np.maximum(1, rng.poisson(30_000 / occupied, occupied))
        x, y = _surface_to_xy(surface)
        official = esr(x, y, WIDTH, HEIGHT)
        assert np.isfinite(official) and official > 0
        ratios.append(esr_v2(x, y, WIDTH, HEIGHT) / official)
    assert max(ratios) / max(min(ratios), 1e-12) > 10, ratios


def test_a_sparse_slice_collapses_the_variant_while_the_official_class_does_not():
    """DVSD22's regime: isolated pixels, and the variant returns ~0 where ESR returns a normal value."""

    rng = np.random.default_rng(11)
    surface = np.zeros((HEIGHT, WIDTH))
    flat = rng.choice(WIDTH * HEIGHT, size=6_000, replace=False)
    surface.flat[flat] = 5.0                          # scattered, never adjacent in practice
    x, y = _surface_to_xy(surface)
    official = esr(x, y, WIDTH, HEIGHT)
    variant = esr_v2(x, y, WIDTH, HEIGHT)
    assert official > 0.5
    assert variant < 0.01 * official
