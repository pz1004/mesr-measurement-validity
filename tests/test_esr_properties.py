"""Section IV-A's mechanism was stated at a generality Eq.~(1) does not support.

Two claims were wrong and one useful claim was missing:

* "ESR is a monotone function of how concentrated a sample is" -- false. `ntss` is
  Schur-convex and `ell_n` is Schur-concave, so majorisation pushes the two factors in
  opposite directions and their product has no fixed sign.
* "the null's direction tracks hot-pixel share" -- underdetermined. Under stationary rates,
  uniform thinning leaves the pixel distribution unchanged at any concentration, so the share
  alone predicts nothing; the sign needs a *nonstationary* scene component to contrast with.
* ESR is invariant to a bijection of the pixel grid. Provable in one line and it bounds what
  "structural" can mean.

A regression on any of these would restore a mechanism the paper cannot support, which is the
one defect that would let a reviewer discard the empirical result along with the explanation.
"""

import numpy as np
import pytest

from dataset_assessment.cap_sensitivity import UNIT_OF_SCALE
from dataset_assessment.esr import esr
from dataset_assessment.esr_properties import (WIDTH, HEIGHT, _majorises, _stream_to_xy,
                                               _synthetic_stream, _thin, esr_of_counts,
                                               monotonicity_counterexample,
                                               permutation_invariance,
                                               stationary_vs_nonstationary)
from dataset_assessment.esr import mesr


def test_majorisation_helper_orders_concentration():
    assert _majorises([30_000], [15_000, 7_500, 7_500])
    assert not _majorises([15_000, 7_500, 7_500], [30_000])
    assert _majorises([10, 10], [10, 10])              # reflexive
    assert not _majorises([10, 10], [10, 10, 10])      # different totals


def test_a_strictly_more_concentrated_slice_can_score_strictly_lower():
    """The counterexample, computed by the released `esr`, not by prose."""

    m = monotonicity_counterexample()
    assert m["concentrated_majorises_spread"]
    assert m["concentrated"]["esr"] == pytest.approx(1.0)
    assert m["spread"]["esr"] == pytest.approx(1.06063, abs=5e-6)
    assert m["spread"]["esr"] > m["concentrated"]["esr"]


def test_the_two_factors_move_in_opposite_directions_under_concentration():
    """Why the counterexample exists at all, and why it is not a special case.

    `ntss` sums the convex `n(n-1)` and `ell_n` subtracts a sum of the convex `c^n`, so
    concentrating counts raises the first and lowers the second. Their product is then free.
    """

    rng = np.random.default_rng(0)
    for _ in range(50):
        k = int(rng.integers(2, 12))
        spread = rng.multinomial(30_000, np.ones(k) / k)
        # A Robin-Hood step in reverse: move mass from a poorer to a richer pixel.
        order = np.argsort(spread)[::-1]
        rich, poor = order[0], order[-1]
        if spread[poor] == 0:
            continue
        concentrated = spread.copy()
        step = int(spread[poor])
        concentrated[rich] += step
        concentrated[poor] -= step
        assert _majorises(concentrated, spread)
        a = esr_of_counts(concentrated)
        b = esr_of_counts(spread)
        assert a["ntss"] >= b["ntss"] - 1e-12, "ntss must be Schur-convex"
        assert a["ell_n"] <= b["ell_n"] + 1e-12, "ell_n must be Schur-concave"


def test_esr_is_invariant_to_a_bijection_of_the_pixel_grid():
    """It reads the count multiset, so no spatial arrangement is visible to it."""

    r = permutation_invariance(trials=40, seed=3)
    assert r["max_abs_difference"] < 1e-12


def test_a_blob_and_the_same_counts_scattered_score_identically():
    """The concrete consequence: 'structural' cannot mean 'spatially arranged'."""

    rng = np.random.default_rng(7)
    counts = rng.multinomial(30_000, np.ones(300) / 300)
    blob = np.repeat(np.arange(300), counts)                       # 300 adjacent pixels
    scattered = np.repeat(rng.choice(WIDTH * HEIGHT, 300, replace=False), counts)
    assert esr(*_stream_to_xy(blob), WIDTH, HEIGHT) == pytest.approx(
        esr(*_stream_to_xy(scattered), WIDTH, HEIGHT))


@pytest.mark.parametrize("hot_share", [0.20, 0.01])
def test_a_stationary_stream_is_unmoved_by_thinning_however_hot(hot_share):
    """The reviewer's Poisson argument, run: concentration alone predicts no direction."""

    rng = np.random.default_rng(0)
    stream = _synthetic_stream(rng, 600_000, hot_share, drifting=False)
    full = mesr(*_stream_to_xy(stream), WIDTH, HEIGHT)
    thinned = _thin(np.random.default_rng(1), stream, 0.25)
    assert abs(mesr(*_stream_to_xy(thinned), WIDTH, HEIGHT) - full) <= UNIT_OF_SCALE


def test_the_sign_needs_a_drifting_scene_and_then_tracks_the_hot_share():
    """Both ingredients are load-bearing, which is the conditional the paper must state."""

    result = stationary_vs_nonstationary(length=600_000, retentions=(1.0, 0.25),
                                         seeds=(0, 1))
    by_cell = {(c["scene"], c["hot_share"]): c for c in result["cells"]}
    assert by_cell[("fixed (stationary)", 0.20)]["sign"] == "flat"
    assert by_cell[("fixed (stationary)", 0.01)]["sign"] == "flat"
    assert by_cell[("drifting (nonstationary)", 0.20)]["sign"] == "rises"
    assert by_cell[("drifting (nonstationary)", 0.01)]["sign"] == "falls"


def test_published_properties_match_the_artifact():
    """The numbers the paper quotes come from `results/esr_properties.json`."""

    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "results" / "esr_properties.json"
    if not path.is_file():
        pytest.skip("esr_properties.json not generated")
    r = json.loads(path.read_text())
    assert r["monotonicity"]["spread"]["esr"] == pytest.approx(1.06063, abs=5e-6)
    assert r["permutation_invariance"]["max_abs_difference"] < 1e-12
    signs = {(c["scene"], c["hot_share"]): c["sign"] for c in r["stationarity"]["cells"]}
    assert signs[("fixed (stationary)", 0.20)] == "flat"
    assert signs[("drifting (nonstationary)", 0.20)] == "rises"
    assert signs[("drifting (nonstationary)", 0.01)] == "falls"
