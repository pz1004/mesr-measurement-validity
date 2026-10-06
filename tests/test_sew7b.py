"""Tests for the frozen classifier's integrator and the path to its evaluator.

The released DVS Gesture classifier was trained on frames built by SpikingJelly's
`split_by='number'` rule: the first `bins - 1` frames take `N // bins` consecutive events each
and the last takes the remainder, with every event in the last frame when `N < bins`. Frames
built any other way would score the checkpoint on inputs it never saw, so the rule is pinned
here on hand-countable streams. The probe must also run the evaluator shipped in this
repository, not one found elsewhere on the path.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from dataset_assessment.downstream_gesture_frozen import CHECKPOINT, EVALUATOR, PROJECT_ROOT
from dataset_assessment.sew7b.frames import events_to_number_frames


def _events(n: int, start: int = 1_000) -> np.ndarray:
    """`n` events one microsecond apart, alternating polarity, on a 4x4 grid."""

    index = np.arange(n, dtype=np.int64)
    return np.column_stack([start + index, index % 4, (index // 4) % 4, index % 2])


def test_each_frame_takes_a_fixed_count_and_the_last_the_remainder():
    frames = events_to_number_frames(_events(35), 1_000, 2_000, bins=16, width=4, height=4)
    assert frames.shape == (16, 2, 4, 4)
    assert frames.sum(axis=(1, 2, 3)).tolist() == [2] * 15 + [5]


def test_fewer_events_than_bins_all_land_in_the_last_frame():
    frames = events_to_number_frames(_events(5), 1_000, 2_000, bins=16, width=4, height=4)
    assert frames[:15].sum() == 0 and frames[15].sum() == 5


def test_only_events_inside_the_interval_are_integrated():
    frames = events_to_number_frames(_events(40), 1_010, 1_030, bins=4, width=4, height=4)
    assert frames.sum() == 20
    assert frames.sum(axis=(1, 2, 3)).tolist() == [5, 5, 5, 5]


def test_polarity_selects_the_channel():
    frames = events_to_number_frames(_events(32), 1_000, 2_000, bins=16, width=4, height=4)
    assert frames[:, 0].sum() == frames[:, 1].sum() == 16


def test_unordered_events_are_refused():
    events = _events(20)[::-1].copy()
    with pytest.raises(ValueError, match="ordered"):
        events_to_number_frames(events, 1_000, 2_000, bins=4, width=4, height=4)


def test_the_probe_runs_the_evaluator_shipped_here():
    spec = importlib.util.find_spec(EVALUATOR)
    assert spec is not None and spec.origin is not None
    assert Path(spec.origin).resolve().is_relative_to(PROJECT_ROOT / "dataset_assessment" / "sew7b")
    assert CHECKPOINT.is_relative_to(PROJECT_ROOT)


#: SHA-256 of the files that produced the frozen-classifier artifacts, with line endings
#: normalised so a checkout that converts them still matches. A deliberate change to one of
#: these files must update its digest here, and say why.
VENDORED = {
    "snn7b.py": "eed285d1dd634079",
    "train_dvsgesture.py": "fa7bf5d04b6ecd32",
    "evaluate_dvsgesture_variants.py": "13193418f51fd0be",
}


@pytest.mark.parametrize("name", sorted(VENDORED))
def test_the_vendored_files_are_the_ones_that_produced_the_artifacts(name):
    data = (PROJECT_ROOT / "dataset_assessment" / "sew7b" / name).read_bytes()
    digest = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
    assert digest.startswith(VENDORED[name])


@pytest.mark.filterwarnings("ignore:`torch.jit.script` is deprecated:DeprecationWarning")
def test_the_evaluator_imports_where_spikingjelly_is_installed():
    pytest.importorskip("spikingjelly")
    module = importlib.import_module(EVALUATOR)
    assert callable(module.main)
