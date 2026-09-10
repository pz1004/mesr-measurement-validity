"""Tests for the downstream statistics helpers.

These are the functions section 8's claims rest on, and two of them had defects that only a
degenerate input exposes: a partial correlation that returned an arbitrary finite number
when its control was collinear, and a summary that dropped its own keys when nothing was
computable. Both are pinned below. The paper argues that a metric returning a plausible
number in a case where it is undefined is a silent-error hazard; the same standard applies
to our own code.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataset_assessment.downstream_gesture import (PAPER_METHODS, partial_spearman,
                                                   summarise_conditions,
                                                   within_retention_correlations)


def _condition(method: str, retention: float, mesr: float, accuracy: float,
               per_seed=None) -> dict:
    row = {"method": method, "retention": retention, "mesr": mesr, "accuracy": accuracy,
           "accuracy_sd": 0.01}
    if per_seed is not None:
        row["accuracy_per_seed"] = {str(s): a for s, a in per_seed.items()}
    return row


# --- partial correlation ------------------------------------------------------------------

def test_partial_spearman_removes_a_confound_the_raw_correlation_shows():
    """`x` and `y` are independent given the control, and both track it. The raw
    correlation is therefore large and the partial correlation must not be. This is
    exactly section 8's situation: retention moves accuracy, and a raw MESR-accuracy
    correlation over all conditions inherits that."""

    rng = np.random.default_rng(0)
    control = np.repeat([0.4, 0.6, 0.8, 1.0], 8).astype(float)
    x = control + rng.normal(0, 0.05, len(control))
    y = control + rng.normal(0, 0.05, len(control))
    assert np.corrcoef(x, y)[0, 1] > 0.9              # confounded
    assert abs(partial_spearman(x, y, control)["rho"]) < 0.4   # confound removed


def test_partial_spearman_keeps_a_relationship_that_is_not_the_confound():
    """The complement: when `y` really does track `x` beyond the control, the partial
    correlation must stay large rather than being explained away."""

    rng = np.random.default_rng(1)
    control = np.repeat([0.4, 0.6, 0.8, 1.0], 8).astype(float)
    x = control + rng.normal(0, 0.3, len(control))
    y = 2 * x + rng.normal(0, 0.05, len(control))
    assert partial_spearman(x, y, control)["rho"] > 0.8


def test_partial_spearman_is_nan_when_the_control_is_collinear():
    """The regression: identical x, y and control leaves a 0/0 whose floating-point
    residues divide to an arbitrary finite value. It must be NaN, not a number."""

    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = partial_spearman(values, values, values)
    assert np.isnan(result["rho"])
    assert "undefined_because" in result


def test_partial_spearman_is_nan_for_a_constant_control():
    result = partial_spearman([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 1.0, 4.0, 3.0, 5.0],
                              [1.0] * 5)
    assert np.isnan(result["rho"])


def test_partial_spearman_is_nan_below_four_points():
    assert np.isnan(partial_spearman([1, 2, 3], [1, 2, 3], [3, 2, 1])["rho"])


def test_partial_spearman_reports_the_sample_size_it_used():
    control = [0.4, 0.4, 0.6, 0.6, 0.8, 0.8]
    assert partial_spearman([1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5], control)["n"] == 6


# --- within-retention combination -----------------------------------------------------

def test_within_retention_always_publishes_its_summary_keys():
    """The regression: callers print `mean_rho` and `fisher_p` unguarded, so an
    uncomputable run must yield NaN rather than a missing key."""

    for rows in ([], [_condition("raw", 0.4, 1.0, 0.5)]):
        result = within_retention_correlations(rows)
        assert "mean_rho" in result and "fisher_p" in result
        assert np.isnan(result["mean_rho"]) and np.isnan(result["fisher_p"])


def test_within_retention_excludes_the_shared_r1_point():
    """At r = 1.0 every method is the unfiltered stream, so that level carries no
    between-method information and must not enter the combination."""

    rows = [_condition(m, r, i + 1.0, 0.4 + 0.01 * i)
            for r in (0.4, 0.6, 1.0) for i, m in enumerate(PAPER_METHODS)]
    assert set(within_retention_correlations(rows)["per_retention"]) == {"0.4", "0.6"}


def test_within_retention_per_seed_is_computed_from_each_seed_not_the_mean():
    """A signal present only in the mean must show up as absent per seed. Here seed 1
    ranks with MESR and seed 2 ranks against it, so their mean_rho must differ in sign."""

    rows = []
    for i, m in enumerate(PAPER_METHODS):
        rows.append(_condition(m, 0.4, float(i), 0.0,
                               per_seed={1: float(i), 2: float(-i)}))
    per_seed = within_retention_correlations(rows)["per_seed"]
    assert per_seed["1"]["mean_rho"] > 0.9
    assert per_seed["2"]["mean_rho"] < -0.9


def test_within_retention_skips_a_level_with_too_few_methods():
    rows = [_condition("raw", 0.4, 1.0, 0.5), _condition("dwf", 0.4, 2.0, 0.6)]
    assert within_retention_correlations(rows)["per_retention"] == {}


# --- the scoped summary ---------------------------------------------------------------

def test_summarise_refuses_to_correlate_too_few_conditions():
    result = summarise_conditions([_condition("raw", 0.4, 1.0, 0.5)])
    assert "error" in result and "mesr_vs_accuracy" not in result


def test_summarise_drops_conditions_with_an_unevaluable_mesr():
    rows = [_condition(m, r, float("nan") if r == 0.4 else 1.0 + i, 0.4 + 0.01 * i)
            for r in (0.4, 0.6) for i, m in enumerate(PAPER_METHODS)]
    summary = summarise_conditions(rows)
    assert summary["n_conditions"] == len(PAPER_METHODS)
    assert summary["mesr_vs_accuracy"]["all"]["n"] == len(PAPER_METHODS)


def test_summarise_reports_the_methods_it_actually_covered():
    rows = [_condition(m, r, 1.0 + i, 0.4 + 0.01 * i)
            for r in (0.4, 0.6) for i, m in enumerate(("raw", "dwf", "red", "ts"))]
    assert summarise_conditions(rows)["methods"] == ["dwf", "raw", "red", "ts"]


@pytest.mark.parametrize("field", ("median", "max"))
def test_summarise_reports_the_seed_spread_that_bounds_any_difference(field):
    rows = [_condition(m, r, 1.0 + i, 0.4 + 0.01 * i)
            for r in (0.4, 0.6) for i, m in enumerate(PAPER_METHODS)]
    assert summarise_conditions(rows)["accuracy_sd_across_seeds"][field] == pytest.approx(0.01)


def test_sensitivity_reports_what_the_forty_conditions_could_have_detected():
    """A failure to reject is only informative next to the effect the test could resolve."""

    from dataset_assessment.downstream_gesture import detectable_rho
    sensitivity = detectable_rho(40)
    assert 0.30 < sensitivity["min_detectable_rho"] < 0.32
    assert sensitivity["power"]["rho_03"] < 0.5      # underpowered for a weak association
    assert sensitivity["power"]["rho_05"] > 0.9      # well powered for a moderate one
    # More points must never make the test less sensitive.
    assert detectable_rho(80)["min_detectable_rho"] < sensitivity["min_detectable_rho"]


def test_shipped_downstream_artifact_carries_the_sensitivity_block():
    import json
    import pathlib
    results = pathlib.Path(__file__).resolve().parents[1] / "results"
    payload = json.loads((results / "downstream_gesture.json").read_text())
    sensitivity = payload["spearman_mesr_vs_accuracy"]["sensitivity"]
    assert sensitivity["n"] == 40
    reported = payload["spearman_mesr_vs_accuracy"]["all"]
    # The reported rho sits below what this design can resolve - which is the point.
    assert abs(reported["rho"]) < sensitivity["min_detectable_rho"]


def _grid(n_methods=8, retentions=(0.2, 0.5, 1.0)):
    """One condition per (method, retention). At r = 1.0 every method is the same input."""

    rows = []
    for m in range(n_methods):
        for r in retentions:
            shared = r >= 1.0
            rows.append({"method": f"m{m}", "retention": r,
                         "mesr": 1.0 if shared else 1.0 + m * 0.1 + r,
                         "accuracy": 0.45 if shared else 0.40 + m * 0.01,
                         "accuracy_sd": 0.05, "accuracy_per_seed": {}})
    return rows


def test_the_r1_endpoint_is_one_input_counted_once_per_method():
    """Eight identical cells are one condition, and pooling them counts it eight times."""

    from dataset_assessment.downstream_gesture import summarise_conditions

    rows = _grid()
    s = summarise_conditions(rows)["mesr_vs_accuracy"]
    assert s["all"]["n"] == 24
    assert s["distinct_inputs"]["n"] == 17          # 16 below r=1, plus the endpoint once
    assert s["distinct_inputs"]["duplicate_cells_collapsed"] == 8
    assert s["excluding_r1"]["n"] == 16             # the endpoint dropped entirely


def test_the_deduplicated_correlation_is_reported_alongside_the_pooled_one():
    """Both must survive: the pooled value is what was published, the distinct one is what
    the association is estimated on. Dropping either would hide a change of conclusion."""

    from dataset_assessment.downstream_gesture import summarise_conditions

    s = summarise_conditions(_grid())["mesr_vs_accuracy"]
    for key in ("all", "excluding_r1", "distinct_inputs"):
        assert {"rho", "p", "n"} <= set(s[key])


def test_the_partial_correlation_is_reported_on_both_weightings():
    """The pre-designated test is a partial correlation, and it inherits the same defect the
    pooled correlation had: the r = 1 endpoint enters once per method, so the control is
    given as many copies of one deterministic condition as there are methods. Both
    weightings must be published, because they are different estimands and neither is
    automatically the intended one."""

    from dataset_assessment.downstream_gesture import summarise_conditions

    s = summarise_conditions(_grid())["mesr_vs_accuracy"]
    pooled = s["partial_given_retention"]
    distinct = s["partial_given_retention_distinct"]
    assert pooled["n"] == 24
    assert distinct["n"] == 17
    assert distinct["duplicate_cells_collapsed"] == 7


def test_deduplicating_the_endpoint_does_not_overturn_the_downstream_verdict():
    """Section 9 reports that the pre-designated within-condition test fails to reject. That
    verdict must survive the endpoint repair, or the section's conclusion changes. It does:
    the correlation rises from +0.261 to +0.333 and stays above 0.05."""

    import json
    from pathlib import Path

    path = (Path(__file__).resolve().parents[1] / "results" / "downstream_gesture.json")
    if not path.is_file():
        pytest.skip("downstream_gesture.json not generated")
    s = json.loads(path.read_text())["spearman_mesr_vs_accuracy"]
    pooled = s["partial_given_retention"]
    distinct = s["partial_given_retention_distinct"]
    assert pooled["n"] == 40 and distinct["n"] == 33
    assert pooled["rho"] == pytest.approx(0.261, abs=5e-4)
    assert distinct["rho"] == pytest.approx(0.333, abs=5e-4)
    assert distinct["rho"] > pooled["rho"]          # the pooled value was biased downward
    assert distinct["p"] > 0.05                     # and the verdict is unchanged


def test_published_downstream_correlations_match_the_artifact():
    """+0.135 over 40 pooled cells and +0.292 over 33 distinct inputs, both quoted."""

    import json
    from pathlib import Path

    path = (Path(__file__).resolve().parents[1] / "results" / "downstream_gesture.json")
    if not path.is_file():
        pytest.skip("downstream_gesture.json not generated")
    s = json.loads(path.read_text())["spearman_mesr_vs_accuracy"]
    assert s["all"]["n"] == 40
    assert s["all"]["rho"] == pytest.approx(0.135, abs=5e-4)
    assert s["distinct_inputs"]["n"] == 33
    assert s["distinct_inputs"]["rho"] == pytest.approx(0.292, abs=5e-4)
    # The headline claim has to survive the repair, and it does: still not significant.
    assert s["distinct_inputs"]["p"] > 0.05
