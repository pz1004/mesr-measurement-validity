"""Tests for the denoiser adapter.

`denoisors.py` was the one module carrying a load-bearing assumption with no test behind
it: `_kept_indices` recovers which input events a cuke-emlb filter kept by an ordered
subsequence merge, and every classical method's score depends on that merge being exact.
It is exact only where no two input events are identical, so the filters are fed their
input in packets without identical events (`_packets`, `_decisions`). These tests pin the
merge, its limit on identical events, the packets, and the failure path, so a change in
the third-party build is caught here rather than showing up as a silently wrong benchmark.

Only the last test needs the cuke-emlb build, and it is skipped without it: it checks the
packet route against the filters themselves. Everything else is pure Python over numpy.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from dataset_assessment import denoisors
from dataset_assessment.denoisors import (CLASSICAL, NATIVE3D, ORACLE, RANDOM_NULL,
                                          _classical_scores, _decisions, _kept_indices,
                                          _oracle_scores, _packets, _random_null_scores,
                                          _to_storage, available_methods,
                                          has_native_operating_point, is_null, is_oracle,
                                          native_retention, score_events)
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


def test_kept_indices_matches_identical_events_to_their_earliest_copies():
    """Three events share `(t, x, y, p)`. The merge assigns kept copies to the earliest
    identical events, which is right here only because the first two are the ones kept."""

    events = _events([[10, 1, 1, 0], [10, 1, 1, 0], [10, 1, 1, 0], [99, 7, 7, 1]])
    indices = _kept_indices(events, _as_kept(events[[0, 1]]))
    assert indices.tolist() == [0, 1]


def test_kept_indices_cannot_see_which_identical_copy_survived():
    """Two identical events, the first labelled signal and the second noise: a filter keeping
    the second outputs exactly what keeping the first would. The merge returns the first, so
    on a whole stream it would count the noise event as signal. Hence the packets."""

    events = _events([[10, 1, 1, 0], [10, 1, 1, 0], [20, 2, 2, 1]])
    assert _kept_indices(events, _as_kept(events[[1, 2]])).tolist() == [0, 2]


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


# --- packets: no two identical events in one ---------------------------------------------

class _Storage:
    """Stands in for `dv_toolkit.EventStorage`: `slice(start, length)` and `size()`."""

    def __init__(self, events: np.ndarray, offset: int = 0):
        self.events, self.offset = events, offset

    def slice(self, start: int, length: int) -> "_Storage":
        return _Storage(self.events[start:start + length], self.offset + start)

    def size(self) -> int:
        return len(self.events)


class _Output:
    def __init__(self, kept: np.ndarray):
        self.kept = kept

    def numpy(self) -> np.ndarray:
        return self.kept


class _Filter:
    """Keeps a fixed set of input positions, deciding over whatever `accept` buffered, the
    way `EventFilterBase::generateEvents()` decides over its buffer."""

    def __init__(self, keep):
        self.keep, self.buffer = set(keep), None

    def accept(self, storage: _Storage) -> None:
        self.buffer = storage

    def generateEvents(self) -> _Output:
        rows = [i for i in range(self.buffer.size()) if self.buffer.offset + i in self.keep]
        out, self.buffer = _Output(_as_kept(self.buffer.events[rows])), None
        return out


def test_packets_never_hold_two_identical_events():
    events = _events([[10, 1, 1, 0], [10, 1, 1, 0], [10, 2, 2, 0], [10, 1, 1, 0],
                      [20, 3, 3, 1], [20, 3, 3, 1]])
    packets = _packets(events)
    assert packets == [(0, 1), (1, 3), (3, 5), (5, 6)]
    for start, stop in packets:
        rows = [tuple(row) for row in events[start:stop].tolist()]
        assert len(rows) == len(set(rows))


def test_packets_cover_the_input_in_order():
    events = _events([[t // 3, t % 2, 0, 0] for t in range(30)])
    packets = _packets(events)
    assert packets[0][0] == 0 and packets[-1][1] == len(events)
    assert all(a[1] == b[0] for a, b in zip(packets, packets[1:]))


def test_packets_are_as_few_as_possible():
    """A repeat forces a cut only after its earlier copy: `a b b a` needs one cut, not two."""

    a, b = [10, 1, 1, 0], [10, 2, 2, 0]
    assert _packets(_events([a, b, b, a])) == [(0, 2), (2, 4)]


def test_a_stream_without_identical_events_is_one_packet():
    """Same timestamp and pixel, different polarity: distinct events, so one packet."""

    events = _events([[1, 0, 0, 0], [1, 0, 0, 1], [2, 0, 0, 0]])
    assert _packets(events) == [(0, 3)]
    assert _packets(_events([[1, 0, 0, 0]])) == [(0, 1)]
    assert _packets(np.empty((0, 4), dtype=np.int64)) == []


def test_decisions_recover_the_identical_copy_a_filter_kept():
    """The case the whole-stream merge gets wrong: the filter keeps the second copy."""

    events = _events([[10, 1, 1, 0], [10, 1, 1, 0], [20, 2, 2, 1]])
    assert _decisions(_Filter(keep=[1, 2]), events, _Storage(events)).tolist() == [1, 2]
    assert _decisions(_Filter(keep=[0, 2]), events, _Storage(events)).tolist() == [0, 2]
    assert _decisions(_Filter(keep=[]), events, _Storage(events)).tolist() == []


class _ShortStorage(_Storage):
    def slice(self, start: int, length: int) -> _Storage:
        return _Storage(self.events[start:start + length - 1], self.offset + start)


def test_decisions_refuse_a_slice_that_does_not_line_up():
    events = _events([[10, 1, 1, 0], [10, 1, 1, 0], [20, 2, 2, 1]])
    with pytest.raises(RuntimeError, match="would not line up"):
        _decisions(_Filter(keep=[1]), events, _ShortStorage(events))


def test_classical_scores_score_the_copy_the_filter_kept(monkeypatch):
    """The adapter end to end, without the build: a filter that keeps the second of two
    identical events must leave the first scored as dropped. A whole-stream merge scores
    them the other way round, `[0, 1, 0]`, which is the fault this pins."""

    events = _events([[10, 1, 1, 0], [10, 1, 1, 0], [20, 2, 2, 1]])
    configs = types.ModuleType("configs")
    configs.denoisors = types.SimpleNamespace(dwf=lambda resolution, params: _Filter([1, 2]))
    monkeypatch.setitem(sys.modules, "configs", configs)
    monkeypatch.setattr(denoisors, "_to_storage", _Storage)
    assert _classical_scores("dwf", _recording(events)).tolist() == [1.0, 0.0, 0.0]


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


# --- the packet route against the build ------------------------------------------------

def _stream_with_identical_events(n: int = 20_000, seed: int = 0) -> np.ndarray:
    """Events on a 64x64 grid at coarse timestamps, a tenth of them repeated verbatim; the
    repeats sort to the end of their timestamp's run, so most are not adjacent copies."""

    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, n * 20, n)) // 100 * 100
    events = np.column_stack([t, rng.integers(0, 64, n), rng.integers(0, 64, n),
                              rng.integers(0, 2, n)]).astype(np.int64)
    events = np.concatenate([events, events[rng.choice(n, n // 10, replace=False)]])
    return events[np.argsort(events[:, 0], kind="stable")]


def test_packets_give_each_filter_its_own_decisions():
    """Feeding a filter its input in packets must change no decision, and must attribute
    each kept event to the copy the filter kept. The build is the only authority on both,
    so each filter is also run one event at a time - where attribution is trivial - and
    once over the whole stream; the adapter must match the first index for index and the
    second value for value."""

    if "dwf" not in available_methods():
        pytest.skip("needs the cuke-emlb build")
    configs = pytest.importorskip("configs.denoisors")
    pytest.importorskip("modules.python.double_window_filter")

    events = _stream_with_identical_events()
    assert len(_packets(events)) > 1
    recording = Recording(events, None, 64, 64, "identical-events", True)
    storage = _to_storage(events)
    for name in CLASSICAL:
        kept = np.flatnonzero(_classical_scores(name, recording) == 0.0)

        model = getattr(configs, name)((64, 64), {})
        single = []
        for index in range(len(events)):
            model.accept(storage.slice(index, 1))
            if len(model.generateEvents().numpy()):
                single.append(index)
        assert kept.tolist() == single, name

        model = getattr(configs, name)((64, 64), {})
        model.accept(_to_storage(events))
        one_pass = model.generateEvents().numpy()
        rows = np.column_stack([one_pass["timestamp"], one_pass["x"], one_pass["y"],
                                one_pass["polarity"] > 0]).astype(np.int64)
        assert np.array_equal(events[kept], rows), name
