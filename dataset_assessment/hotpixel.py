"""Does the null's MESR gain survive the preprocessing the metric's own release recommends?

`ding2024emlb` records that ESR "is easily affected by hot pixels" and recommends eliminating
them in preprocessing. The two corpora on which a label-blind null *raises* the corpus mean --
DVSD22 and Pure_BA -- are unfiltered streams, so the gain reported in the paper belongs to a
regime that recommendation excludes. Whether it survives the recommendation is a separate
question, and this module is the experiment that asks it.

**The rule is the one already declared, not a new one.** `regime.hot_pixel_share` calls a pixel
hot when its event count reaches the 99.9th percentile of the occupied pixels' counts; that is
the statistic the paper already reports per corpus. This module promotes the same predicate to
a removal step. Nothing here reads MESR, so the rule cannot be tuned to the outcome, and it is
applied identically to every procedure compared downstream.

**Removal happens after the event cap**, on exactly the stream the benchmark would otherwise
have scored, so the before/after cohort is the same recordings in the same order and the
comparison is within-recording.

Either outcome is informative. If the gain persists, the practical claim strengthens: the
response is not an artefact of pixels the release already tells users to drop. If it
attenuates, the boundary is a real one and the paper's claim is bounded to unfiltered input.

Used by `run_benchmark --hot-pixel-removal`; the summary it returns is written beside each
record so the exclusions are auditable rather than implicit.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

import numpy as np

#: The percentile `regime.hot_pixel_share` uses. Changing it changes what the paper reports.
PERCENTILE = 99.9


def hot_pixel_mask(events: np.ndarray, width: int, height: int,
                   percentile: float = PERCENTILE) -> np.ndarray:
    """Boolean over events: True where the event came from a hot pixel.

    Hot is defined exactly as in `regime.hot_pixel_share` -- a pixel whose count reaches the
    `percentile` of the *occupied* pixels' counts. Empty pixels are excluded from the
    percentile, as they are there, so the threshold does not collapse on a sparse sensor.
    """

    flat = events[:, 2] * width + events[:, 1]
    counts = np.bincount(flat, minlength=width * height)
    occupied = counts[counts > 0]
    if occupied.size == 0:
        return np.zeros(len(events), dtype=bool)
    threshold = np.percentile(occupied, percentile)
    return counts[flat] >= threshold


def remove_hot_pixels(recording, percentile: float = PERCENTILE) -> Tuple[object, Dict]:
    """The recording with hot-pixel events dropped, and what was dropped.

    Returns the recording unchanged when removal would empty it, with `applied` False, so a
    degenerate recording is reported rather than silently turned into a different experiment.
    """

    events = recording.events
    hot = hot_pixel_mask(events, recording.width, recording.height, percentile)
    n_hot = int(hot.sum())
    flat = events[:, 2] * recording.width + events[:, 1]
    summary = {
        "percentile": percentile,
        "events_before": int(len(events)),
        "events_removed": n_hot,
        "share_removed": n_hot / len(events) if len(events) else float("nan"),
        "hot_pixels": int(len(np.unique(flat[hot]))) if n_hot else 0,
        "occupied_pixels": int(len(np.unique(flat))),
        "applied": True,
    }
    keep = ~hot
    if not keep.any():
        summary["applied"] = False
        summary["events_after"] = int(len(events))
        return recording, summary

    summary["events_after"] = int(keep.sum())
    trimmed = replace(
        recording,
        events=events[keep],
        labels=None if recording.labels is None else recording.labels[keep],
    )
    return trimmed, summary
