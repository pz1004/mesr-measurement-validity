import numpy as np
import pytest

from dataset_assessment.esr import esr
from dataset_assessment.protocol import (
    decompose_esr, delta_over_raw, protocol_row, summarise_curve,
)


def curve_fixture():
    # Interior optimum at r=0.4; r=0.05 unevaluable.
    return [
        {"r": 0.05, "mesr": float("nan"), "kept_events": 900, "evaluable": False},
        {"r": 0.20, "mesr": 0.90, "kept_events": 40_000, "evaluable": True},
        {"r": 0.40, "mesr": 1.00, "kept_events": 80_000, "evaluable": True},
        {"r": 0.80, "mesr": 0.85, "kept_events": 160_000, "evaluable": True},
    ]


def test_delta_over_raw_is_a_difference():
    assert delta_over_raw(0.95, 0.81) == pytest.approx(0.14)


def test_decompose_esr_reconstructs_esr():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 346, 30_000)
    y = rng.integers(0, 260, 30_000)
    parts = decompose_esr(x, y, 346, 260)
    assert np.sqrt(parts["ntss"] * parts["ln"]) == pytest.approx(esr(x, y, 346, 260), rel=1e-12)


def test_decompose_esr_does_not_depend_on_the_declared_sensor_size():
    """Both factors are K-free, which is why the protocol reports them instead of a
    sqrt(W*H) normalisation."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 346, 30_000)
    y = rng.integers(0, 260, 30_000)
    small = decompose_esr(x, y, 346, 260)
    large = decompose_esr(x, y, 1280, 720)
    assert small["ntss"] == large["ntss"]
    assert small["ln"] == large["ln"]
    assert small["occupied_px"] == large["occupied_px"]


def test_decompose_esr_bounds_ln_by_the_occupied_pixel_count():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 60, 30_000)
    y = rng.integers(0, 60, 30_000)
    parts = decompose_esr(x, y, 346, 260)
    assert 0 < parts["ln"] <= parts["occupied_px"]
    assert parts["occupied_px"] <= 60 * 60


def test_summarise_curve_finds_interior_optimum_and_ignores_unevaluable():
    s = summarise_curve(curve_fixture())
    assert s["r_star"] == 0.40
    assert s["mesr_star"] == pytest.approx(1.00)
    assert s["evaluable_range"] == (0.20, 0.80)
    assert s["optimum_at_evaluable_floor"] is False


def test_summarise_curve_flags_a_floor_pinned_optimum():
    pinned = [
        {"r": 0.20, "mesr": 1.10, "kept_events": 40_000, "evaluable": True},
        {"r": 0.40, "mesr": 1.00, "kept_events": 80_000, "evaluable": True},
    ]
    assert summarise_curve(pinned)["optimum_at_evaluable_floor"] is True


def test_summarise_curve_area_is_retention_weighted():
    s = summarise_curve(curve_fixture())
    # trapezoid over r in [0.2, 0.8] divided by the width, so it is a mean not a sum
    assert 0.85 < s["auc_over_r"] < 1.00


def test_summarise_curve_does_not_use_a_deprecated_numpy_alias():
    """np.trapz raises DeprecationWarning on numpy >= 2.0 and will be removed."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        summarise_curve(curve_fixture())


def test_protocol_row_is_self_describing():
    row = protocol_row("KNoise", curve_fixture(), raw_mesr=0.81, width=346, height=260)
    for key in ("method", "sensor", "r_star", "mesr_star", "delta_over_raw_at_r_star",
                "auc_over_r", "evaluable_range", "optimum_at_evaluable_floor", "raw_mesr"):
        assert key in row
    assert row["delta_over_raw_at_r_star"] == pytest.approx(0.19)
    assert "normalised_mesr_star" not in row, (
        "dividing MESR by sqrt(W*H) manufactures a resolution dependence the metric "
        "does not have; see plan/dataset_assessment_notes.md finding F-LN")


def test_protocol_row_carries_the_decomposition_when_given_one():
    row = protocol_row("KNoise", curve_fixture(), raw_mesr=0.81, width=346, height=260,
                       decomposition={"ntss": 0.01, "ln": 1000.0, "occupied_px": 2000,
                                      "n_events": 30_000})
    assert row["ln"] == 1000.0 and row["occupied_px"] == 2000


def test_all_nan_curve_raises_rather_than_returning_junk():
    with pytest.raises(ValueError):
        summarise_curve([{"r": 0.1, "mesr": float("nan"), "kept_events": 5,
                          "evaluable": False}])
