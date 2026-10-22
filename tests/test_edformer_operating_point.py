"""The operating-point sweep with EDformer's scores is the synthetic sweep with scores swapped.

What must hold without torch: the cached whole-recording scores are refused unless aligned;
the spread is max minus min over the *evaluable* retentions only; and the ROC AUC is computed
over the events EDformer scores, ignoring the tail the release never scores.
"""

import numpy as np
import pytest

from dataset_assessment import edformer, edformer_operating_point as op
from dataset_assessment.esr import SLICE
from dataset_assessment.readers import Recording


def _stream(n=120_000, seed=0):
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.3).astype(np.int64)            # 1 = noise
    x = rng.integers(0, 64, n)
    y = rng.integers(0, 48, n)
    scores = np.where(labels == 1, rng.uniform(0.4, 1.0, n), rng.uniform(0.0, 0.6, n))
    scores = scores.astype(np.float32)
    scores[-100:] = edformer.TAIL_SCORE                          # the unscored tail
    return labels, x, y, scores


def test_sweep_row_spread_is_over_evaluable_points_only():
    labels, x, y, scores = _stream()
    row = op.sweep_row("t", scores, labels, x, y, 64, 48)
    usable = [c["mesr"] for c in row["curve"] if c["evaluable"]]
    assert row["spread"] == pytest.approx(max(usable) - min(usable))
    assert row["evaluable_retentions"] == len(usable) < len(row["curve"])
    assert all(c["kept_events"] >= SLICE for c in row["curve"] if c["evaluable"])
    assert row["scored"] == len(scores) - 100


def test_auc_ignores_the_tail():
    labels, x, y, scores = _stream()
    from sklearn.metrics import roc_auc_score
    row = op.sweep_row("t", scores, labels, x, y, 64, 48)
    assert row["roc_auc"] == pytest.approx(roc_auc_score(labels[:-100], scores[:-100]))


def test_misaligned_or_missing_cache_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(edformer, "CACHE", tmp_path)
    events = np.zeros((10, 4), dtype=np.int64)
    rec = Recording(events, np.zeros(10, dtype=np.int64), 346, 260, "DND21/1hz_driving", True)
    with pytest.raises(ValueError):
        op.scores_for(rec)
    path = edformer.cache_path(rec.name, op.WHOLE)
    path.parent.mkdir(parents=True)
    np.save(path, np.zeros(9, dtype=np.float32))
    with pytest.raises(ValueError):
        op.scores_for(rec)
