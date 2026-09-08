"""Tests for the denoiser adapter.

`denoisors.py` was the one module carrying a load-bearing assumption with no test behind
it: `_kept_indices` recovers which input events a cuke-emlb filter kept by an ordered
subsequence merge, and every classical method's score depends on that merge being exact.
Its own docstring flags the assumption ("generateEvents() returns an exact ordered
subsequence"); these tests pin the behaviour, including the failure path, so a change in
the third-party build is caught here rather than showing up as a silently wrong benchmark.

Nothing here needs the cuke-emlb build: the merge, the null, the oracle and the
operating-point predicates are all pure Python over numpy arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataset_assessment.denoisors import (CLASSICAL, NATIVE3D, ORACLE, RANDOM_NULL,
                                          _kept_indices, _oracle_scores,
                                          _random_null_scores, has_native_operating_point,
                                          is_null, is_oracle, native_retention,
                                          score_events)
from dataset_assessment.readers import Recording


def _events(rows) -> np.ndarray:
    return np.asarray(rows, dtype=np.int64)


def _as_kept(events: np.ndarray) -> np.ndarray:
    """Repackage `[t, x, y, p]` rows in the structured form `generateEvents()` returns."""

    kept = np.empty(len(events), dtype=[("timestamp", "<i8"), ("x", "<i8"),
                                        ("y", "<i8"), ("polarity", "<i8")])
    kept["timestamp"], kept["x"] = events[:, 0], events[:, 1]
    kept["y"], kept["polarity"] = events[:, 2], events[:, 3]
    return kept


def _recording(events: np.ndarray, labels=None) -> Recording:
    return Recording(events, labels, 8, 8, "unit-test", True)


# --- the ordered-subsequence merge ------------------------------------------------------

def test_kept_indices_recovers_a_strict_subsequence():
    events = _events([[10, 1, 1, 0], [20, 2, 2, 1], [30, 3, 3, 0], [40, 4, 4, 1]])
    indices = _kept_indices(events, _as_kept(events[[0, 2, 3]]))
    assert indices.tolist() == [0, 2, 3]


def test_kept_indices_disambiguates_events_sharing_a_coordinate():
    """Three events share `(t, x, y, p)`. A set of keys could not tell them apart; the
    ordered merge assigns the first two, which is what preserving input order means."""

    events = _events([[10, 1, 1, 0], [10, 1, 1, 0], [10, 1, 1, 0], [99, 7, 7, 1]])
    indices = _kept_indices(events, _as_kept(events[[0, 1]]))
    assert indices.tolist() == [0, 1]


def test_kept_indices_keeps_everything_when_nothing_was_dropped():
    events = _events([[1, 0, 0, 0], [2, 1, 0, 1], [3, 2, 0, 0]])
    assert _kept_indices(events, _as_kept(events)).tolist() == [0, 1, 2]


def test_kept_indices_raises_when_order_is_not_preserved():
    """The failure the docstring warns about: if `generateEvents()` ever reorders, scores
    would be attributed to the wrong events. That must raise, never silently truncate."""

    events = _events([[10, 1, 1, 0], [20, 2, 2, 1], [30, 3, 3, 0]])
    with pytest.raises(RuntimeError, match="ordered subsequence"):
        _kept_indices(events, _as_kept(events[[2, 0]]))


def test_kept_indices_raises_on_an_event_that_is_not_in_the_input():
    events = _events([[10, 1, 1, 0], [20, 2, 2, 1]])
    with pytest.raises(RuntimeError, match="ordered subsequence"):
        _kept_indices(events, _as_kept(_events([[99, 5, 5, 1]])))


# --- the two reference rows -------------------------------------------------------------

def test_random_null_is_reproducible_and_ignores_the_events():
    """The null must depend on nothing but its seed, or it is not a null."""

    a = _random_null_scores(_recording(_events([[1, 0, 0, 0]] * 64)))
    b = _random_null_scores(_recording(_events([[9, 7, 7, 1]] * 64)))
    assert np.array_equal(a, b)
    assert a.min() >= 0.0 and a.max() < 1.0


def test_oracle_scores_rank_signal_below_noise():
    """Lower score = more likely signal, so `retain_mask` keeps signal first."""

    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    scores = _oracle_scores(_recording(_events([[1, 0, 0, 0]] * 4), labels))
    assert scores.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_oracle_refuses_an_unlabelled_recording():
    with pytest.raises(ValueError, match="no labels"):
        _oracle_scores(_recording(_events([[1, 0, 0, 0]])))


def test_raw_scores_every_event_identically():
    scores = score_events("raw", _recording(_events([[1, 0, 0, 0]] * 5)))
    assert len(np.unique(scores)) == 1


def test_score_events_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        score_events("no_such_denoiser", _recording(_events([[1, 0, 0, 0]])))


# --- operating points -------------------------------------------------------------------

def test_native_retention_is_the_kept_share_of_a_binary_score():
    assert native_retention(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)) == 0.5


def test_native_retention_is_nan_for_a_continuous_score():
    """A uniform ranking has a whole curve, not an operating point. Returning 0.0 here
    would drag a meaningless point into the native-operating-point ranking."""

    assert np.isnan(native_retention(np.linspace(0, 1, 16, dtype=np.float32)))


def test_native_retention_is_nan_for_a_three_level_score():
    assert np.isnan(native_retention(np.array([0.0, 0.5, 1.0], dtype=np.float32)))


def test_only_keep_drop_methods_claim_a_native_operating_point():
    assert has_native_operating_point("raw")
    assert has_native_operating_point(ORACLE)
    assert all(has_native_operating_point(m) for m in CLASSICAL)
    # A continuous scorer belongs on the curve but in no native-point comparison.
    assert not has_native_operating_point(RANDOM_NULL)
    assert not has_native_operating_point(NATIVE3D)


def test_nulls_and_oracle_are_flagged_so_they_cannot_enter_a_ranking():
    assert is_null("raw") and is_null(RANDOM_NULL)
    assert is_oracle(ORACLE)
    assert not any(is_null(m) or is_oracle(m) for m in CLASSICAL)
