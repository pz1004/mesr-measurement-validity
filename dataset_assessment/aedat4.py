"""AEDAT-4 reader, copied verbatim from the parent project.

Source: `native3d_ssm/phase5_emlb_mesr.py` in the companion repository, the same file this
project ported `esr()`/`mesr()` from. It is vendored here rather than imported because
`native3d_ssm/__init__.py` eagerly imports the parent project's model code, which pulls in
torch — a heavy dependency for a 25-line reader, and one that would make this release depend
on a different paper's method.

The function is unmodified. Equivalence against the original was verified byte-for-byte on
an E-MLB recording; see `REPRODUCE.md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


def read_aedat4(path: Path, max_events: Optional[int]) -> Optional[np.ndarray]:
    import dv_processing as dv

    recording = dv.io.MonoCameraRecording(str(path))
    chunks: List[np.ndarray] = []
    total = 0
    while recording.isRunning():
        batch = recording.getNextEventBatch()
        if batch is None:
            continue
        array = batch.numpy()
        chunks.append(array)
        total += len(array)
        if max_events is not None and total >= max_events:
            break
    if not chunks:
        return None
    events = np.concatenate(chunks)
    if max_events is not None:
        events = events[:max_events]
    # Some released recordings contain overlapping packets. A stable sort keeps
    # source order for timestamp ties and changes no event value, matching the
    # repair already used by filter_analytic_aedat.py.
    if len(events) > 1 and np.any(np.diff(events["timestamp"].astype(np.int64)) < 0):
        events = events[np.argsort(events["timestamp"], kind="mergesort")]
        read_aedat4.repairs += 1
    return events


read_aedat4.repairs = 0
