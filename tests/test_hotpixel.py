"""The hot-pixel rule must be the one the paper already declares, not a new one tuned to fit.

`regime.hot_pixel_share` is the statistic the paper reports per corpus: the share of events
emitted by pixels whose count reaches the 99.9th percentile of the *occupied* pixels' counts.
`hotpixel` promotes that same predicate to a removal step, so the two must agree exactly --
otherwise the experiment answers a question about a threshold nobody stated.

Nothing here may read MESR. A rule that could be tuned to the outcome would make the
before/after comparison worthless, which is the whole reason the removal reuses a published
threshold instead of choosing one.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

from dataset_assessment.hotpixel import hot_pixel_mask, remove_hot_pixels
from dataset_assessment.regime import hot_pixel_share


@dataclass
class _Rec:
    """The fields `remove_hot_pixels` touches. A dataclass, because it uses `replace`."""

    events: np.ndarray
    labels: Optional[np.ndarray]
    width: int
    height: int


def _events(pixels):
    """(N,4) t,x,y,p from a list of (x, y), timestamps in arrival order."""

    return np.array([[i, x, y, 1] for i, (x, y) in enumerate(pixels)], dtype=np.int64)


def _stream(width=8, height=8, hot=(3, 4), hot_count=500, spread=2000, seed=20260915):
    """One very busy pixel against `spread` events scattered over the rest of the sensor."""

    rng = np.random.default_rng(seed)
    pixels = [hot] * hot_count
    while len(pixels) < hot_count + spread:
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        if (x, y) != hot:
            pixels.append((x, y))
    rng.shuffle(pixels)
    return _events(pixels)


def test_the_removed_share_is_exactly_the_reported_statistic():
    """`share_removed` must equal `regime.hot_pixel_share` on the same stream."""

    events = _stream()
    rec = _Rec(events, None, 8, 8)
    _, summary = remove_hot_pixels(rec)
    assert summary["share_removed"] == pytest.approx(hot_pixel_share(events, 8, 8))


def test_removal_drops_exactly_the_flagged_events():
    events = _stream()
    mask = hot_pixel_mask(events, 8, 8)
    trimmed, summary = remove_hot_pixels(_Rec(events, None, 8, 8))
    assert summary["events_removed"] == int(mask.sum())
    assert len(trimmed.events) == len(events) - int(mask.sum())
    assert np.array_equal(trimmed.events, events[~mask])


def test_labels_travel_with_their_events():
    """A label array that stopped matching its events would corrupt every downstream count."""

    events = _stream()
    labels = np.arange(len(events)) % 2
    mask = hot_pixel_mask(events, 8, 8)
    trimmed, _ = remove_hot_pixels(_Rec(events, labels, 8, 8))
    assert np.array_equal(trimmed.labels, labels[~mask])


def test_empty_pixels_do_not_set_the_threshold():
    """The percentile is over occupied pixels, so a sparse sensor cannot collapse it."""

    events = _stream(width=64, height=64)               # 4096 pixels, few occupied
    dense = hot_pixel_mask(events, 8, 8)
    sparse = hot_pixel_mask(events, 64, 64)
    assert int(dense.sum()) > 0 and int(sparse.sum()) > 0


def test_a_uniform_stream_loses_little_and_stays_applied():
    """With no hot pixel there is nothing much to remove, and the rule must not empty it."""

    rng = np.random.default_rng(7)
    events = _events([(int(x), int(y)) for x, y in rng.integers(0, 8, size=(4000, 2))])
    trimmed, summary = remove_hot_pixels(_Rec(events, None, 8, 8))
    assert summary["applied"]
    assert len(trimmed.events) > 0.9 * len(events)


def test_a_single_pixel_stream_is_reported_rather_than_emptied():
    """Removal that would delete everything is refused, and says so."""

    events = _events([(1, 1)] * 100)
    rec = _Rec(events, None, 8, 8)
    trimmed, summary = remove_hot_pixels(rec)
    assert summary["applied"] is False
    assert len(trimmed.events) == len(events)
