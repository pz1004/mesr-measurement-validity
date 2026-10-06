"""The released count-bin frame integrator for DVS Gesture.

`events_to_number_frames` is the one function `downstream_gesture_frozen` takes from the
study's DVS Gesture utilities. It is copied unchanged and needs numpy alone; the rest of that
module reads the raw corpus through readers this package does not use.
"""

from __future__ import annotations

import numpy as np


def events_to_number_frames(
    events: np.ndarray,
    start_us: int,
    end_us: int,
    bins: int = 16,
    width: int = 128,
    height: int = 128,
    dtype=np.uint16,
) -> np.ndarray:
    """Integrate a labeled interval using the released count-bin convention.

    This matches SpikingJelly 0.0.0.0.12's ``split_by='number'`` rule: the
    first ``bins - 1`` frames receive ``N // bins`` consecutive events and the
    final frame receives the remainder.  Boundaries are therefore determined
    independently for every retained event stream, as in the published DVS
    Gesture training pipeline.
    """

    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must have shape [N, 4]")
    if bins <= 0 or width <= 0 or height <= 0 or start_us >= end_us:
        raise ValueError("Invalid frame integration configuration")
    if len(events) and np.any(np.diff(events[:, 0].astype(np.int64, copy=False)) < 0):
        raise ValueError("events must be ordered by timestamp")
    left = int(np.searchsorted(events[:, 0], start_us, side="left"))
    right = int(np.searchsorted(events[:, 0], end_us, side="left"))
    selected = events[left:right]
    shape = (bins, 2, height, width)
    if not len(selected):
        return np.zeros(shape, dtype=dtype)
    x = selected[:, 1].astype(np.int64, copy=False)
    y = selected[:, 2].astype(np.int64, copy=False)
    polarity = selected[:, 3].astype(np.int64, copy=False)
    if (
        np.any(x < 0)
        or np.any(x >= width)
        or np.any(y < 0)
        or np.any(y >= height)
        or np.any((polarity != 0) & (polarity != 1))
    ):
        raise ValueError("Event coordinate or polarity lies outside the sensor")

    events_per_bin = len(selected) // bins
    if events_per_bin == 0:
        # The historical implementation gives every event to the final bin
        # because all earlier [j_l, j_r) ranges are empty when N < bins.
        number_bin = np.full(len(selected), bins - 1, dtype=np.int64)
    else:
        number_bin = np.arange(len(selected), dtype=np.int64) // events_per_bin
        number_bin = np.minimum(number_bin, bins - 1)
    flat_index = (((number_bin * 2 + polarity) * height + y) * width + x)
    counts = np.bincount(flat_index, minlength=int(np.prod(shape)))
    maximum = np.iinfo(np.dtype(dtype)).max
    if counts.max(initial=0) > maximum:
        raise OverflowError(f"Frame count exceeds {np.dtype(dtype)} capacity")
    return counts.reshape(shape).astype(dtype, copy=False)
