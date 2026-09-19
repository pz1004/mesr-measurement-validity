"""EDformer enters the pipeline through cached scores, and must enter it the way it is released.

`edformer.py` runs the released model (it needs torch and CUDA); everything else reads its
cached per-event sigmoid. What must hold, without torch:

* **The released decision.** Keep iff sigmoid < 0.005; events past the last full 4096-event
  block, which the release never scores, are rejected.
* **Scores belong to their recording.** A cache that does not have one score per event is
  refused rather than silently misaligned.
* **Adding a method moves nothing already there.** `native_emlb.merge` carries every existing
  cell over verbatim and refuses a clash or a changed input.
* **The artifacts agree.** The manifest's kept counts match the cache, and the merged native
  table carries an EDformer row with its matched-null margin.
"""

import hashlib
import json

import numpy as np
import pytest

from dataset_assessment import denoisors, edformer
from dataset_assessment.analyze import RESULTS
from dataset_assessment.native_emlb import merge
from dataset_assessment.readers import Recording


def _recording(n, name="E-MLB/D-END/Test-ND00-1"):
    events = np.column_stack([np.arange(n), np.zeros(n), np.zeros(n), np.ones(n)]).astype(np.int64)
    return Recording(events, None, 346, 260, name, False)


def test_released_decision_and_tail():
    scores = np.array([0.001, 0.00499, 0.005, 0.5, edformer.TAIL_SCORE], dtype=np.float32)
    assert edformer.keep_mask(scores).tolist() == [True, True, False, False, False]


def test_native_scores_read_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(edformer, "CACHE", tmp_path)
    rec = _recording(4)
    path = edformer.cache_path(rec.name, denoisors.EDFORMER_CAP)
    path.parent.mkdir(parents=True)
    np.save(path, np.array([0.001, 0.9, edformer.TAIL_SCORE, 0.004], dtype=np.float32))
    assert denoisors.score_events("edformer_native", rec).tolist() == [0.0, 1.0, 1.0, 0.0]
    assert denoisors.native_retention(denoisors.score_events("edformer_native", rec)) == 0.5
    assert denoisors.has_native_operating_point("edformer_native")
    assert not denoisors.has_native_operating_point("edformer")


def test_misaligned_cache_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(edformer, "CACHE", tmp_path)
    rec = _recording(5)
    path = edformer.cache_path(rec.name, denoisors.EDFORMER_CAP)
    path.parent.mkdir(parents=True)
    np.save(path, np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError):
        denoisors.score_events("edformer", rec)


def _record(name, cells, raw=1.0, events=10):
    return {"recording": name, "events": events, "raw_mesr": raw,
            "cells": [{"method": m, "mesr": v} for m, v in cells], "unevaluable": []}


def test_merge_carries_existing_cells_verbatim():
    old = [_record("a", [("dwf", 0.3)]), _record("b", [("red", 0.7)])]
    new = [_record("a", [("edformer_native", 0.5)]), _record("b", [])]
    out = merge(old, new)
    assert out[0]["cells"][0] == old[0]["cells"][0]
    assert [c["method"] for c in out[0]["cells"]] == ["dwf", "edformer_native"]
    with pytest.raises(ValueError):
        merge(old, [_record("a", [("dwf", 0.1)]), _record("b", [])])
    with pytest.raises(ValueError):
        merge(old, [_record("a", [], raw=2.0), _record("b", [])])


def _load(name):
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    return json.loads(path.read_text())


def test_manifest_matches_cache():
    manifest = _load("edformer_emlb_manifest.json")
    assert manifest["threshold"] == 0.005 and manifest["seq_len"] == 4096
    checked = 0
    for row in manifest["records"][:8]:
        path = edformer.cache_path(row["recording"], manifest["cap"])
        if not path.is_file():
            continue
        scores = np.load(path)
        assert len(scores) == row["events"]
        assert int(edformer.keep_mask(scores).sum()) == row["kept"]
        assert hashlib.sha256(scores.tobytes()).hexdigest() == row["sha256"]
        checked += 1
    if not checked:
        pytest.skip("score cache not present")


def test_native_table_carries_edformer_with_its_null():
    payload = _load("native_emlb.json")
    row = payload["summary"]["by_method"].get("edformer_native")
    if row is None:
        pytest.skip("EDformer not merged yet")
    assert row["evaluable"] + row["unevaluable"] == payload["summary"]["recordings"]
    margin = row["filter_minus_null"]
    assert margin["lo"] <= margin["mean"] <= margin["hi"]
