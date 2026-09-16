"""The paired readout must answer the question the marginal one only appears to.

The claim under test in the paper is that the seed moves accuracy more than the filter does.
Stated on marginal standard deviations that claim is not falsifiable by the design: two
conditions sharing seeds can both swing widely while their difference is stable. These tests
pin the two cases apart, because a regression that silently reverted to marginal spreads
would restore exactly the reasoning this module exists to replace.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataset_assessment.downstream_gesture import PAPER_METHODS
from dataset_assessment.downstream_paired import (paired_differences, summarise)


def _condition(method: str, retention: float, per_seed) -> dict:
    return {"method": method, "retention": retention, "mesr": 1.0,
            "accuracy": float(np.mean(per_seed)),
            "accuracy_per_seed": {str(i): float(a) for i, a in enumerate(per_seed)}}


def test_a_shared_seed_effect_cancels_in_the_difference():
    """The case the marginal comparison gets wrong: a large common seed swing, and a
    perfectly stable gap. Pairing must resolve it."""

    a, b = PAPER_METHODS[0], PAPER_METHODS[1]
    swing = np.array([0.0, 0.30, -0.30])
    rows = paired_differences([_condition(a, 0.4, 0.5 + swing),
                               _condition(b, 0.4, 0.4 + swing)])
    assert len(rows) == 1
    row = rows[0]
    assert {row["a"], row["b"]} == {a, b}      # pairs are emitted in name order
    assert row["marginal_sd_a"] > 0.2          # each condition swings hugely
    assert row["sd_of_difference"] == pytest.approx(0.0, abs=1e-12)
    assert abs(row["mean_difference"]) == pytest.approx(0.1)
    assert row["resolved"] and row["sign_consistent"]


def test_an_independent_seed_effect_leaves_the_difference_unresolved():
    """The complement: when the seed moves the two conditions independently and the gap is
    small against that noise, the pair must not be reported as resolved."""

    a, b = PAPER_METHODS[0], PAPER_METHODS[1]
    rows = paired_differences([_condition(a, 0.4, [0.50, 0.80, 0.20]),
                               _condition(b, 0.4, [0.79, 0.21, 0.51])])
    assert not rows[0]["resolved"]
    assert not rows[0]["sign_consistent"]


def test_the_shared_r1_endpoint_is_excluded():
    """At r = 1 every method is the same unfiltered stream, so every difference is zero and
    would count as sign-consistent noise-free evidence. It must not enter."""

    rows = paired_differences([_condition(m, 1.0, [0.45, 0.45, 0.45]) for m in PAPER_METHODS])
    assert rows == []


def test_only_the_reported_methods_enter():
    """The extended grid adds two filters from a sibling project. No claim in the paper
    covers them, so they must not move these counts."""

    rows = paired_differences([_condition(PAPER_METHODS[0], 0.4, [0.5, 0.6, 0.7]),
                               _condition(PAPER_METHODS[1], 0.4, [0.4, 0.5, 0.6]),
                               _condition("native3d", 0.4, [0.9, 0.9, 0.9])])
    assert {r["a"] for r in rows} | {r["b"] for r in rows} == set(PAPER_METHODS[:2])


def test_a_cell_without_per_seed_accuracies_is_dropped_rather_than_guessed():
    conditions = [_condition(PAPER_METHODS[0], 0.4, [0.5, 0.6, 0.7]),
                  {"method": PAPER_METHODS[1], "retention": 0.4, "accuracy": 0.5}]
    assert paired_differences(conditions) == []


def test_the_summary_reports_both_spreads_it_compares():
    a, b = PAPER_METHODS[0], PAPER_METHODS[1]
    swing = np.array([0.0, 0.30, -0.30])
    summary = summarise([_condition(a, 0.4, 0.5 + swing), _condition(b, 0.4, 0.4 + swing)])
    assert summary["n_pairs"] == 1 and summary["n_resolved"] == 1
    assert summary["median_marginal_sd"] > summary["median_paired_sd"]


def test_the_summary_publishes_its_keys_when_nothing_is_computable():
    """A caller printing n_pairs must not meet a KeyError on an empty grid."""

    summary = summarise([])
    assert summary["n_pairs"] == 0 and "error" in summary
