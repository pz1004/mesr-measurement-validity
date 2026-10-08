"""Counting identical events, and the shape of the identity audit's artifact.

The audit's verdict - the packet route changes no filter decision - is checked against the
build in `tests/test_denoisors.py`; here the counting rule is pinned on a hand-built stream.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dataset_assessment.identity_audit import (OUT, identical_groups, summarise_groups,
                                                summarise_routes)


def test_groups_count_only_complete_repeats_and_flag_mixed_labels():
    events = np.array([[10, 1, 1, 0], [10, 1, 1, 0],    # a group, signal + noise
                       [10, 1, 1, 1],                   # same pixel, other polarity
                       [20, 2, 2, 1], [20, 2, 2, 1], [20, 2, 2, 1],  # a group, all noise
                       [30, 3, 3, 0]], dtype=np.int64)
    labels = np.array([0, 1, 0, 1, 1, 1, 0], dtype=np.int64)
    groups = identical_groups(events, labels)
    assert (groups["groups"], groups["events_in_groups"], groups["largest"]) == (2, 5, 3)
    assert groups["mixed_groups"] == 1


def test_a_stream_without_repeats_has_no_groups():
    events = np.array([[1, 0, 0, 0], [1, 0, 0, 1], [2, 0, 0, 0]], dtype=np.int64)
    assert identical_groups(events)["groups"] == 0
    assert "mixed_groups" not in identical_groups(events)


def test_summaries_add_up_over_recordings():
    per = [{"events": 5, "groups": 2, "events_in_groups": 4, "largest": 2, "mixed_groups": 1},
           {"events": 3, "groups": 0, "events_in_groups": 0, "largest": 1, "mixed_groups": 0}]
    summary = summarise_groups(per)
    assert (summary["groups"], summary["mixed_groups"], summary["recordings_with_groups"]) \
        == (2, 1, 1)
    cells = [{"moved": 0, "tp_scored": 10, "tp_scored_whole_merge": 10},
             {"moved": 3, "tp_scored": 10, "tp_scored_whole_merge": 11}]
    routes = summarise_routes(cells)
    assert (routes["cells_moved"], routes["tp_changed_cells"], routes["max_tp_overcount"]) \
        == (1, 1, 1)


def test_compare_routes_measures_what_a_whole_stream_merge_misattributes(monkeypatch):
    """A filter keeping the second of two identical events, signal then noise: the packet
    route scores one signal event, a whole-stream merge two."""

    from dataset_assessment import identity_audit as ia
    from dataset_assessment.readers import Recording

    events = np.array([[10, 1, 1, 0], [10, 1, 1, 0], [20, 2, 2, 1]], dtype=np.int64)
    recording = Recording(events, np.array([0, 1, 0]), 4, 4, "stub", True)
    monkeypatch.setattr(ia.denoisors, "_classical_scores",
                        lambda name, rec: np.array([1.0, 0.0, 0.0], dtype=np.float32))
    monkeypatch.setattr(ia, "_one_pass", lambda name, rec: events[[1, 2]])
    monkeypatch.setattr(ia, "SLICE", 1)
    cell = ia.compare_routes("dwf", recording)
    assert cell["moved"] == 1
    assert (cell["tp_scored"], cell["tp_scored_whole_merge"]) == (1, 2)

    monkeypatch.setattr(ia, "_one_pass", lambda name, rec: events[[2]])
    with pytest.raises(RuntimeError, match="differs from the one-pass output"):
        ia.compare_routes("dwf", recording)


#: Every input the experiments read, in the order `main` audits them.
CORPORA = ("dnd21", "dnd21_hotpixel", "dvsclean", "dvsclean_hotpixel", "gesture_test_clips",
           "gesture_trained_samples", "dvsd22", "pure_ba", "emlb")


def test_the_artifact_compares_every_filter_on_every_input():
    if not OUT.exists():
        pytest.skip("identity_audit.json not present")
    payload = json.loads(OUT.read_text())
    corpora = payload["corpora"]
    assert tuple(corpora) == CORPORA
    for block in corpora.values():
        assert block["routes"]["cells"] == (block["summary"]["recordings"]
                                            * len(payload["filters"]))
    for name in CORPORA[:4]:
        assert corpora[name]["routes"]["cells"] == 60
    assert corpora["dvsclean"]["summary"]["groups"] == 0


def test_a_failed_run_leaves_the_artifact_in_place(monkeypatch, tmp_path):
    """Progress goes to a sidecar; only a run through every input replaces the artifact."""

    import sys

    from dataset_assessment import identity_audit as ia

    out = tmp_path / "identity_audit.json"
    out.write_text("complete\n")
    partial = tmp_path / "identity_audit.partial.json"
    monkeypatch.setattr(sys, "argv", ["identity_audit", "--out", str(out)])
    monkeypatch.setattr(ia.denoisors, "available_methods", lambda: ["dwf"])
    for loader in ("_labelled", "_capped", "_gesture_clips", "_gesture_samples"):
        monkeypatch.setattr(ia, loader, lambda *args: [])

    def audit(name, recordings, routes, workers):
        if name == "dvsd22":
            raise RuntimeError("the packet output differs from the one-pass output")
        return {"summary": {}}

    monkeypatch.setattr(ia, "audit_corpus", audit)
    with pytest.raises(RuntimeError):
        ia.main()
    assert out.read_text() == "complete\n"
    assert tuple(json.loads(partial.read_text())["corpora"]) == CORPORA[:6]

    monkeypatch.setattr(ia, "audit_corpus", lambda *args: {"summary": {}})
    ia.main()
    assert tuple(json.loads(out.read_text())["corpora"]) == CORPORA
    assert not partial.exists()
