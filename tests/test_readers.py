import numpy as np
import pytest

from dataset_assessment.readers import (
    Recording, iter_dnd21, iter_dvsd22, iter_dvsclean, iter_ed24, iter_emlb,
    iter_pure_ba_noise,
)


def assert_valid(rec: Recording):
    assert rec.events.ndim == 2 and rec.events.shape[1] == 4
    assert rec.events.dtype == np.int64
    assert np.all(np.diff(rec.events[:, 0]) >= 0), "timestamps must be non-decreasing"
    assert rec.events[:, 1].max() < rec.width and rec.events[:, 2].max() < rec.height
    assert rec.events[:, 1].min() >= 0 and rec.events[:, 2].min() >= 0
    assert set(np.unique(rec.events[:, 3])).issubset({0, 1})
    if rec.labels is not None:
        assert rec.labels.shape == (len(rec.events),)
        assert rec.labels.dtype == np.int64
        assert set(np.unique(rec.labels)).issubset({0, 1})


def test_dnd21_first_recording_is_labeled_and_synthetic():
    rec = next(iter_dnd21())
    assert_valid(rec)
    assert rec.labels is not None and rec.synthetic is True
    assert (rec.width, rec.height) == (346, 260)


def test_dvsclean_is_labeled_synthetic_and_hd():
    rec = next(iter_dvsclean())
    assert_valid(rec)
    assert rec.labels is not None and rec.synthetic is True
    assert (rec.width, rec.height) == (1280, 720)


def test_dvsclean_timestamps_are_microseconds_not_seconds():
    """The file stores float32 SECONDS; a reader that forgets to scale collapses the
    stream into a handful of distinct timestamps and silently breaks any time-windowed
    statistic."""
    rec = next(iter_dvsclean())
    span_us = int(rec.events[-1, 0] - rec.events[0, 0])
    assert span_us > 100_000, f"stream spans only {span_us} us; timestamps not scaled"
    assert len(np.unique(rec.events[:, 0])) > len(rec.events) // 100


def test_emlb_is_unlabeled_and_real():
    rec = next(iter_emlb(max_events=60_000))
    assert_valid(rec)
    assert rec.labels is None and rec.synthetic is False
    assert len(rec.events) <= 60_000


def test_ed24_is_labeled_and_synthetic():
    rec = next(iter_ed24(scenes=("Bicycle_01",), levels=("0.0",)))
    assert_valid(rec)
    assert rec.labels is not None and rec.synthetic is True
    assert (rec.width, rec.height) == (346, 260)


def test_pure_ba_noise_is_all_noise():
    rec = next(iter_pure_ba_noise(max_events=60_000))
    assert_valid(rec)
    assert rec.labels is not None
    assert rec.labels.min() == 1 and rec.labels.max() == 1
    assert rec.synthetic is False


def test_dvsd22_reads_aedat2_from_a_davis346():
    rec = next(iter_dvsd22(max_events=200_000))
    assert_valid(rec)
    assert rec.labels is None and rec.synthetic is False
    assert (rec.width, rec.height) == (346, 260)
    assert len(rec.events) == 200_000


def test_dvsd22_exposes_the_drop_frequency_factor():
    """The controlled signal-rate axis is the reason this corpus is in the paper, so it
    must survive into the Recording rather than being buried in the filename."""
    names = [rec.name for rec in iter_dvsd22(max_events=2_000)]
    assert len(names) >= 11
    assert any("45Hz" in n for n in names) and any("110Hz" in n for n in names)
    assert not any("calibration" in n for n in names)


def test_recording_is_immutable():
    rec = next(iter_dnd21())
    with pytest.raises(Exception):
        rec.width = 1
