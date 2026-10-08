"""The mark shuffle keeps every count and timestamp, and removes order below its window."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from dataset_assessment.null_shuffle import CORPORA, OUT, WINDOWS, shuffle_marks


def _stream(n: int = 5_000, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([np.arange(n) * 10, rng.integers(0, 32, n), rng.integers(0, 24, n),
                            rng.integers(0, 2, n)]).astype(np.int64)


def _marks(rows: np.ndarray) -> Counter:
    return Counter(map(tuple, rows[:, 1:].tolist()))


def test_timestamps_stay_and_each_window_keeps_its_marks():
    events = _stream()
    shuffled = shuffle_marks(events, 1_000, seed=7, name="a")
    assert np.array_equal(shuffled[:, 0], events[:, 0])
    for start in range(0, len(events), 1_000):
        assert _marks(shuffled[start:start + 1_000]) == _marks(events[start:start + 1_000])
    assert not np.array_equal(shuffled, events)


def test_window_zero_is_the_recording_and_none_is_one_window():
    events = _stream()
    assert shuffle_marks(events, 0, seed=7) is events
    whole = shuffle_marks(events, None, seed=7, name="a")
    assert _marks(whole) == _marks(events)
    # A whole-stream shuffle moves marks across what any shorter window would hold.
    assert _marks(whole[:1_000]) != _marks(events[:1_000])


def test_the_shuffle_is_reproducible_per_recording():
    events = _stream()
    a = shuffle_marks(events, 1_000, seed=7, name="a")
    assert np.array_equal(a, shuffle_marks(events, 1_000, seed=7, name="a"))
    assert not np.array_equal(a, shuffle_marks(events, 1_000, seed=7, name="b"))
    assert not np.array_equal(a, shuffle_marks(events, 1_000, seed=8, name="a"))


def test_the_artifact_reproduces_the_benchmark_unshuffled():
    if not OUT.exists():
        pytest.skip("null_shuffle.json not present")
    payload = json.loads(OUT.read_text())
    assert payload["windows"] == list(WINDOWS)
    assert tuple(payload["corpora"]) == CORPORA
    for block in payload["corpora"].values():
        assert block["unshuffled_cells_matching_benchmark"] == 2 * 19 * block["recordings"]


def test_each_control_scores_the_shuffled_stream(monkeypatch):
    from dataset_assessment import null_shuffle as ns
    from dataset_assessment.readers import Recording

    seen = []

    def score(control, recording):
        seen.append(recording.events.copy())
        return np.zeros(len(recording.events), dtype=np.float32)

    monkeypatch.setattr(ns.denoisors, "score_events", score)
    monkeypatch.setattr(ns, "WINDOWS", (None,))
    events = _stream(n=500)
    ns._one(Recording(events, None, 32, 24, "a", False))
    for seed in ns.SHUFFLE_SEEDS:
        shuffled = shuffle_marks(events, None, seed, "a")
        assert sum(np.array_equal(s, shuffled) for s in seen) == len(ns.CONTROLS)


def test_a_failed_run_leaves_the_artifact_in_place(monkeypatch, tmp_path):
    """Progress goes to a sidecar; only a run through every corpus replaces the artifact."""

    import sys

    from dataset_assessment import null_shuffle as ns

    out = tmp_path / "null_shuffle.json"
    out.write_text("complete\n")
    partial = tmp_path / "null_shuffle.partial.json"
    monkeypatch.setattr(sys, "argv", ["null_shuffle", "--out", str(out)])

    def analyse(dataset, workers):
        if dataset == CORPORA[-1]:
            raise AssertionError("unshuffled Delta against the benchmark's")
        return {"recordings": 0}

    monkeypatch.setattr(ns, "analyse", analyse)
    with pytest.raises(AssertionError):
        ns.main()
    assert out.read_text() == "complete\n"
    assert tuple(json.loads(partial.read_text())["corpora"]) == CORPORA[:-1]

    monkeypatch.setattr(ns, "analyse", lambda dataset, workers: {"recordings": 0})
    ns.main()
    assert tuple(json.loads(out.read_text())["corpora"]) == CORPORA
    assert not partial.exists()


def test_window_zero_must_match_the_benchmark_cell_for_cell():
    from dataset_assessment.null_matched_span import GRID
    from dataset_assessment.null_shuffle import CONTROLS, check_unshuffled

    curve = [{"r": r, "mesr": 1.0 + r, "evaluable": True} for r in GRID]
    records = {"a": {"raw_mesr": 1.5, "methods": {c: {"curve": curve} for c in CONTROLS}}}
    run = {"window": 0, **{c: [1.0 + r - 1.5 for r in GRID] for c in CONTROLS}}
    assert check_unshuffled(records, [{"recording": "a", "runs": [run]}]) == 2 * len(GRID)
    run[CONTROLS[1]] = [v + 1e-9 for v in run[CONTROLS[1]]]
    with pytest.raises(AssertionError, match="unshuffled"):
        check_unshuffled(records, [{"recording": "a", "runs": [run]}])
    with pytest.raises(AssertionError, match="not in the benchmark"):
        check_unshuffled(records, [{"recording": "b", "runs": [run]}])


def test_the_summary_counts_positive_means_per_window_and_control():
    from dataset_assessment.null_matched_span import GRID
    from dataset_assessment.null_shuffle import CONTROLS, SHUFFLE_SEEDS, summarise

    runs = []
    for window in WINDOWS:
        for seed in (SHUFFLE_SEEDS[:1] if window == 0 else SHUFFLE_SEEDS):
            sign = 1.0 if window == 0 else -1.0
            runs.append({"window": window, "seed": seed,
                         **{c: [sign * 0.1] * len(GRID) for c in CONTROLS}})
    rows = summarise([{"recording": "a", "runs": runs}], reference_r=GRID[0])
    unshuffled = [r for r in rows if r["window"] == 0]
    shuffled = [r for r in rows if r["window"] is None]
    assert all(r["positive"] == [len(GRID)] for r in unshuffled)
    assert all(r["positive"] == [0, 0, 0] and r["at_reference_max"] == -0.1 for r in shuffled)
