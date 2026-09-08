"""Method-independent statistics that characterise a dataset's NOISE, not a denoiser.

Design requirement: these must be comparable across sensor resolutions, because the paper's
central dataset claim compares 346x260 real data against 1280x720 synthetic data. Every
statistic below is either a correlation (already dimensionless) or normalised by the number
of occupied pixels rather than by W*H.

`spatial_autocorrelation`
    Within short time windows, the share of events whose 8-neighbourhood also fired.
    Spatially uniform background activity has a low value regardless of sensor size;
    coherent structure (edges, motion) has a high one.

`local_global_coupling`
    Correlation, across time windows, between per-window local co-activation and total
    window activity. This is the sensor-agnostic analogue of the parent project's operand
    correlation: it asks whether "more events overall" also means "more locally coherent
    events" (real scenes) or not (uniform injection).

`hot_pixel_share`
    Share of all events emitted by the busiest 0.1% of *occupied* pixels. Dead pixels on a
    large sensor cannot dilute it.

`event_rate_per_px`
    Events per pixel per second. The one statistic that does divide by W*H, because it is a
    density by definition.

Implementation note: the neighbour statistic is computed over the sparse set of occupied
pixels, never over a W*H surface. On 1280x720 a dense surface would cost ~1 MB of allocation
and 7 M operations per 1 ms window; the sparse form costs O(n log n) in the window's own
event count and is bit-identical (asserted in tests/test_regime.py).
"""

from __future__ import annotations

from typing import Dict, Iterator, Tuple

import numpy as np

NEIGHBOUR_OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                          if not (dy == 0 and dx == 0))


def _window_slices(timestamps: np.ndarray, window_us: int,
                   min_events: int = 8) -> Iterator[Tuple[int, int]]:
    """Yield [start, stop) index pairs for fixed-duration windows holding >= min_events.

    Window membership is computed by integer division rather than by building an edge
    array, so the cost does not depend on the recording's duration and absolute
    microsecond timestamps of order 1e15 lose no precision.
    """

    if len(timestamps) == 0:
        return
    bins = (timestamps - timestamps[0]) // window_us
    boundaries = np.flatnonzero(np.diff(bins)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(timestamps)]))
    for start, stop in zip(starts, stops):
        if stop - start >= min_events:
            yield int(start), int(stop)


def _neighbour_share(x: np.ndarray, y: np.ndarray, width: int, height: int) -> float:
    """Share of events with at least one 8-neighbour occupied in the same window."""

    if len(x) == 0:
        return float("nan")
    linear = y * width + x
    occupied = np.unique(linear)
    has_neighbour = np.zeros(len(x), dtype=bool)
    for dy, dx in NEIGHBOUR_OFFSETS:
        nx, ny = x + dx, y + dy
        inside = (nx >= 0) & (nx < width) & (ny >= 0) & (ny < height)
        candidate = ny * width + nx
        position = np.searchsorted(occupied, candidate)
        np.clip(position, 0, len(occupied) - 1, out=position)
        has_neighbour |= inside & (occupied[position] == candidate)
    return float(has_neighbour.mean())


def spatial_autocorrelation(events: np.ndarray, width: int, height: int,
                            window_us: int = 1000) -> float:
    """Mean neighbour-occupancy share over time windows. Dimensionless, sensor-agnostic."""

    t, x, y = events[:, 0], events[:, 1], events[:, 2]
    shares = [_neighbour_share(x[a:b], y[a:b], width, height)
              for a, b in _window_slices(t, window_us)]
    return float(np.mean(shares)) if shares else float("nan")


def local_global_coupling(events: np.ndarray, width: int, height: int,
                          window_us: int = 1000) -> float:
    """Correlation between per-window local coherence and per-window total activity."""

    t, x, y = events[:, 0], events[:, 1], events[:, 2]
    local, total = [], []
    for a, b in _window_slices(t, window_us):
        local.append(_neighbour_share(x[a:b], y[a:b], width, height))
        total.append(b - a)
    if len(local) < 3:
        return float("nan")
    local_arr, total_arr = np.asarray(local), np.asarray(total, dtype=float)
    if local_arr.std() == 0 or total_arr.std() == 0:
        return 0.0
    return float(np.corrcoef(local_arr, total_arr)[0, 1])


def hot_pixel_share(events: np.ndarray, width: int, height: int,
                    percentile: float = 99.9) -> float:
    """Share of all events emitted by the busiest 0.1% of occupied pixels."""

    counts = np.bincount(events[:, 2] * width + events[:, 1],
                         minlength=width * height)
    counts = counts[counts > 0]
    if counts.size == 0:
        return float("nan")
    threshold = np.percentile(counts, percentile)
    return float(counts[counts >= threshold].sum() / counts.sum())


def event_rate_per_px(events: np.ndarray, width: int, height: int) -> float:
    duration_us = float(events[-1, 0] - events[0, 0])
    if duration_us <= 0:
        return 0.0
    return float(len(events) / (duration_us * 1e-6) / (width * height))


def regime_profile(recording) -> Dict:
    """All four statistics for one Recording, plus its identity."""

    e, w, h = recording.events, recording.width, recording.height
    return {
        "name": recording.name,
        "synthetic": recording.synthetic,
        "spatial_autocorr": spatial_autocorrelation(e, w, h),
        "local_global_coupling": local_global_coupling(e, w, h),
        "event_rate_per_px": event_rate_per_px(e, w, h),
        "hot_pixel_share": hot_pixel_share(e, w, h),
        "events": int(len(e)),
    }
