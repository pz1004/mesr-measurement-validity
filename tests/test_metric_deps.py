"""Guards on the influence-quantity measurement.

Two things are pinned here. First, the shape of the broadened statistic: an influence
quantity must come with an interval over recordings, not a single recording's spread.
Second, the RNG-ordering bug that changed a published number without failing anything --
see the E1 dependency note in the working record.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from dataset_assessment.measure_metric_deps import _spread_summary, rank_by_label

RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def deps():
    return json.loads((RESULTS / "metric_dependencies.json").read_text())


def test_rank_by_label_realises_the_requested_auc():
    """At the tolerance the function asserts (0.01), this needs a realistic sample.

    500+500 lands at AUC 0.9137 for a 0.93 target -- inside sampling noise, outside the
    tolerance. Real recordings carry 10^5-10^6 events, so the assertion is safe there;
    the test uses 20,000 to stay honest about why.
    """

    rng = np.random.default_rng(0)
    labels = np.array([0] * 10_000 + [1] * 10_000)
    for target in (0.80, 0.93, 0.99):
        rank_by_label(labels, rng, target_auc=target)  # asserts internally


def test_rank_by_label_consumes_the_generator():
    """The property behind E1: two calls on the same rng do not agree.

    This is why the single-recording curve must be taken from the loop rather than
    recomputed after it.
    """

    rng = np.random.default_rng(0)
    labels = np.array([0] * 200 + [1] * 200)
    first = rank_by_label(labels, rng)
    second = rank_by_label(labels, rng)
    assert not np.allclose(first, second)


def test_single_recording_spread_matches_the_stored_curve(deps):
    """E1 regression: the headline single-recording spread must equal its own curve.

    If the curve and the spread are computed from different rng draws, they disagree and
    the paper quotes a number no stored artifact supports.
    """

    block = deps["retention_dependence"]
    usable = [c["mesr"] for c in block["curve"] if c["evaluable"]]
    assert block["min_mesr"] == pytest.approx(min(usable))
    assert block["max_mesr"] == pytest.approx(max(usable))
    assert block["spread"] == pytest.approx(max(usable) - min(usable))


@pytest.mark.parametrize("block", ["retention_dependence", "slice_dependence"])
def test_influence_quantities_carry_an_interval_over_recordings(deps, block):
    summary = deps[block]["spread_over_recordings"]
    assert summary["n"] >= 10, "an influence quantity needs more than one recording"
    assert summary["lo"] <= summary["mean"] <= summary["hi"]
    assert summary["lo"] > 0, "a spread whose interval includes zero is not an influence"
    assert summary["min"] <= summary["mean"] <= summary["max"]


def test_per_recording_rows_cover_every_recording(deps):
    for block in ("retention_dependence", "slice_dependence"):
        rows = deps[block]["per_recording"]
        assert len(rows) == deps[block]["spread_over_recordings"]["n"]
        assert len({r["recording"] for r in rows}) == len(rows), "duplicate recording"


def test_slice_size_monotonicity_is_measured_not_asserted(deps):
    """The draft called the slice-size curve 'monotone increasing' from one recording."""

    assert deps["slice_dependence"]["monotone_increasing_share"] == pytest.approx(1.0)


def test_spread_summary_degrades_gracefully():
    assert np.isnan(_spread_summary([])["mean"])
    single = _spread_summary([0.5])
    assert single["mean"] == pytest.approx(0.5) and single["n"] == 1
    assert np.isnan(single["lo"])
