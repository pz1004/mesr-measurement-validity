"""Unified readers. Every dataset becomes the same `Recording` shape.

Column order is `[t, x, y, p]` int64 with p in {0,1}, t non-decreasing and in MICROSECONDS,
because the metric and every denoiser adapter assume it. The `synthetic` flag is the paper's
primary grouping variable, so it is part of the record rather than inferred later.

On-disk formats, all confirmed against the files on this machine (see
`plan/dataset_assessment_notes.md`, section F-FORMATS):

DND21      `ECCV2024_datasets/AUC_test/<rate>/<scene>.txt` - whitespace text WITH a header
           row `timestamp x y polarity label`; label 1 = noise; 346x260; synthetic injection.
ED24       `ECCV2024_datasets/ED24/<Scene>/<Scene>_<level>.csv` - same five whitespace
           columns and header; 100 scenes x 41 levels; 346x260; synthetic injection.
DVSCLEAN   `DVSCLEAN/simulated_data/<seq>_<level>.hdf5` - arrays live under an `events`
           GROUP, timestamps are float32 SECONDS, and polarity is a CONTINUOUS float32 that
           the parent project thresholds at >= 0.5
           (`native3d_ssm/evaluate_dvsclean_analytic.py:85`); 1280x720; synthetic.
E-MLB      `E-MLB/<part>/<Scene>/<Scene>-ND<xx>-<rep>.aedat4` - AEDAT-4, dv-processing;
           346x260; real sensor; unlabeled.
Pure_BA    `ECCV2024_datasets/Pure_BA_noise/<level>-<stamp>.aedat4` - AEDAT-4; 346x260; real
           sensor pointed at a signal-free scene, so every event is a false positive and the
           labels are all 1. File size grows ~40x from level 0.0 to 4.0.
DVSD22 `DVSD22/Recordings/<session>/Davis346blue_<date>[_<n>x<f>Hz].aedat` - jAER
           AEDAT-2.0 (big-endian int32 address + int32 timestamp pairs), DAVIS346, 346x260;
           real sensor; unlabeled. Recordings of falling water drops with an independent
           physical ground truth in `DVSD22/Data and configuration/*.CSV`. Released as
           "DVSD22 - Dynamic Vision Sensor Disdrometer 2022" (sensors.ini.ch/datasets) with
           Micev et al., Atmos. Meas. Tech. 17(1):335-357, 2024, doi 10.5194/amt-17-335-2024.
           NOT an event-denoising benchmark: that paper applies no filtering of any kind.
           Older working notes call this corpus DROPLETS22 and its directory DND22.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DND21_RATES = ("1hz", "3hz", "5hz", "7hz", "10hz")
DND21_SCENES = {"hotel-bar": "mix_result.txt", "driving": "driving_mix_result.txt"}
DVSCLEAN_SEQUENCES = ("MAH00446", "MAH00454", "MAH00455", "MAH00456", "MAH00465")
EMLB_PARTS = ("D-END", "N-END")
EMLB_ND = ("ND00", "ND04", "ND16", "ND64")

# jAER DAVIS raw AER address layout, validated against every DVSD22 recording: decoding all
# 13 files this way yields x in [0,345], y in [0,259] with zero out-of-range words.
AEDAT2_X_SHIFT, AEDAT2_X_MASK = 12, 0x003FF000
AEDAT2_Y_SHIFT, AEDAT2_Y_MASK = 22, 0x7FC00000
AEDAT2_POL_SHIFT = 11


@dataclass(frozen=True)
class Recording:
    events: np.ndarray            # (N,4) int64: t, x, y, p
    labels: Optional[np.ndarray]  # (N,) int64, 1 = noise, or None
    width: int
    height: int
    name: str
    synthetic: bool

    @property
    def x(self) -> np.ndarray:
        return self.events[:, 1]

    @property
    def y(self) -> np.ndarray:
        return self.events[:, 2]


def _sorted(events: np.ndarray, labels: Optional[np.ndarray]):
    """Stable-sort by timestamp if needed. Changes no event value."""

    if len(events) > 1 and np.any(np.diff(events[:, 0]) < 0):
        order = np.argsort(events[:, 0], kind="mergesort")
        events = events[order]
        labels = None if labels is None else labels[order]
    return events, labels


def _from_whitespace_table(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a `timestamp x y polarity label` table into (events, labels)."""

    frame = pd.read_csv(path, sep=r"\s+", dtype=np.int64, engine="c")
    values = frame.to_numpy(copy=False)
    events = np.ascontiguousarray(values[:, :4].astype(np.int64))
    labels = np.ascontiguousarray(values[:, 4].astype(np.int64))
    return _sorted(events, labels)


def _from_aedat4(path: Path, max_events: Optional[int]) -> Optional[np.ndarray]:
    from .aedat4 import read_aedat4

    packet = read_aedat4(path, max_events)
    if packet is None:
        return None
    events = np.column_stack([
        packet["timestamp"].astype(np.int64),
        packet["x"].astype(np.int64),
        packet["y"].astype(np.int64),
        (packet["polarity"].astype(np.int64) > 0).astype(np.int64)])
    return np.ascontiguousarray(events)


def iter_dnd21(root: Path = PROJECT_ROOT / "ECCV2024_datasets/AUC_test",
               rates: Sequence[str] = DND21_RATES) -> Iterator[Recording]:
    for rate in rates:
        for scene, fname in DND21_SCENES.items():
            events, labels = _from_whitespace_table(root / rate / fname)
            yield Recording(events, labels, 346, 260, f"DND21/{rate}_{scene}", True)


def iter_ed24(root: Path = PROJECT_ROOT / "ECCV2024_datasets/ED24",
              scenes: Optional[Sequence[str]] = None,
              levels: Optional[Sequence[str]] = None) -> Iterator[Recording]:
    """ED24: 100 scenes x 41 injected-noise levels. Both axes are caller-capped because the
    full corpus is 4,100 files."""

    scene_dirs = ([root / s for s in scenes] if scenes
                  else sorted(p for p in root.iterdir() if p.is_dir()))
    for scene_dir in scene_dirs:
        paths = sorted(scene_dir.glob(f"{scene_dir.name}_*.csv"))
        if levels is not None:
            wanted = {f"{scene_dir.name}_{lv}.csv" for lv in levels}
            paths = [p for p in paths if p.name in wanted]
        for path in paths:
            events, labels = _from_whitespace_table(path)
            yield Recording(events, labels, 346, 260, f"ED24/{path.stem}", True)


def iter_dvsclean(root: Path = PROJECT_ROOT / "DVSCLEAN/simulated_data",
                  sequences: Sequence[str] = DVSCLEAN_SEQUENCES,
                  levels: Sequence[int] = (50, 100)) -> Iterator[Recording]:
    import h5py

    for seq in sequences:
        for level in levels:
            path = root / f"{seq}_{level}.hdf5"
            with h5py.File(path, "r") as handle:
                group = handle["events"]
                seconds = np.asarray(group["timestamp"]).astype(np.float64)
                # float32 seconds -> int64 microseconds, rebased so the cast is exact.
                micros = np.rint((seconds - seconds[0]) * 1e6).astype(np.int64)
                events = np.column_stack([
                    micros,
                    np.asarray(group["x"]).astype(np.int64),
                    np.asarray(group["y"]).astype(np.int64),
                    # polarity is a continuous float32; the parent thresholds at 0.5.
                    (np.asarray(group["polarity"]).astype(np.float32) >= 0.5).astype(np.int64)])
                labels = np.asarray(group["label"]).astype(np.int64)
            events, labels = _sorted(np.ascontiguousarray(events), labels)
            yield Recording(events, labels, 1280, 720, f"DVSCLEAN/{seq}_{level}", True)


def iter_emlb(root: Path = PROJECT_ROOT / "E-MLB",
              max_events: Optional[int] = None,
              reps: Sequence[int] = (1,),
              scene_stride: int = 1) -> Iterator[Recording]:
    """E-MLB, 48 scenes x 4 ND x 2 lighting parts per repetition.

    `scene_stride` subsamples *scenes*, never ND levels or lighting parts, so a capped run
    still covers the whole noise-level and illumination grid. Capping by recording count
    instead would truncate inside D-END and drop N-END entirely, because iteration runs
    part -> scene -> ND.
    """

    for part in EMLB_PARTS:
        scene_dirs = sorted(p for p in (root / part).iterdir() if p.is_dir())
        for scene_dir in scene_dirs[::scene_stride]:
            for nd in EMLB_ND:
                for rep in reps:
                    path = scene_dir / f"{scene_dir.name}-{nd}-{rep}.aedat4"
                    if not path.exists():
                        continue
                    events = _from_aedat4(path, max_events)
                    if events is None:
                        continue
                    events, _ = _sorted(events, None)
                    yield Recording(events, None, 346, 260,
                                    f"E-MLB/{part}/{scene_dir.name}-{nd}-{rep}", False)


def iter_pure_ba_noise(root: Path = PROJECT_ROOT / "ECCV2024_datasets/Pure_BA_noise",
                       max_events: Optional[int] = None) -> Iterator[Recording]:
    """Signal-free captures: every retained event is a false positive, so labels are all 1."""

    for path in sorted(root.glob("*.aedat4")):
        events = _from_aedat4(path, max_events)
        if events is None:
            continue
        events, _ = _sorted(events, None)
        labels = np.ones(len(events), dtype=np.int64)
        level = path.name.split("-", 1)[0]
        yield Recording(events, labels, 346, 260, f"Pure_BA/{level}", False)


def read_aedat2(path: Path, max_events: Optional[int] = None) -> np.ndarray:
    """Parse a jAER AEDAT-2.0 DAVIS file into `[t, x, y, p]` int64.

    AEDAT-2.0 is a `#`-commented ASCII header followed by big-endian
    `(int32 address, int32 timestamp)` pairs at 1 us resolution. dv-processing reads
    AEDAT-4 only, hence this parser.

    Words whose decoded coordinates fall outside the sensor (APS samples, IMU words,
    external-input markers on recordings that captured them) are dropped; on the DVSD22
    corpus that count is zero.
    """

    with path.open("rb") as handle:
        offset = 0
        while True:
            line = handle.readline()
            if not line.startswith(b"#"):
                handle.seek(offset)
                break
            offset = handle.tell()
        words = np.fromfile(handle, dtype=">u4")

    words = words[: (len(words) // 2) * 2]
    addresses = words[0::2]
    timestamps = words[1::2].astype(np.int64)

    x = ((addresses & AEDAT2_X_MASK) >> AEDAT2_X_SHIFT).astype(np.int64)
    y = ((addresses & AEDAT2_Y_MASK) >> AEDAT2_Y_SHIFT).astype(np.int64)
    polarity = ((addresses >> AEDAT2_POL_SHIFT) & 1).astype(np.int64)

    valid = (x < 346) & (y < 260)
    events = np.column_stack([timestamps[valid], x[valid], y[valid], polarity[valid]])
    if max_events is not None:
        events = events[:max_events]
    return np.ascontiguousarray(events)


def read_aedat31(path: Path, max_events: Optional[int] = None) -> np.ndarray:
    """Parse an AEDAT-3.1 polarity stream (DVS Gesture, DVS128) into `[t, x, y, p]` int64.

    AEDAT-3.1 is a `#`-commented ASCII header followed by packets, each with a 28-byte
    little-endian header `(eventType, eventSource, eventSize, eventTSOffset,
    eventTSOverflow, eventCapacity, eventNumber, eventValid)`. Polarity packets
    (`eventType == 1`) carry 8-byte events: a 32-bit data word and a 32-bit timestamp that
    is relative to `eventTSOverflow << 31`.

    Data word layout: bit 0 validity, bit 1 polarity, bits 2-16 Y, bits 17-31 X. Verified
    on `user01_fluorescent.aedat`: x, y in [0, 127], binary polarity, monotone timestamps,
    2,008,130 events over 40.01 s.
    """

    import struct

    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line.startswith(b"#"):
                handle.seek(offset)
                break
        blob = handle.read()

    chunks, cursor, total = [], 0, 0
    while cursor + 28 <= len(blob):
        (event_type, _source, event_size, _ts_offset, ts_overflow,
         _capacity, number, _valid) = struct.unpack_from("<hhiiiiii", blob, cursor)
        cursor += 28
        body = blob[cursor: cursor + event_size * number]
        cursor += event_size * number
        if event_type != 1 or event_size != 8 or not number:
            continue
        pairs = np.frombuffer(body, dtype="<u4").reshape(-1, 2)
        timestamps = pairs[:, 1].astype(np.int64) | (np.int64(ts_overflow) << 31)
        chunks.append((pairs[:, 0], timestamps))
        total += len(timestamps)
        if max_events is not None and total >= max_events:
            break

    if not chunks:
        return np.empty((0, 4), dtype=np.int64)
    words = np.concatenate([c[0] for c in chunks])
    timestamps = np.concatenate([c[1] for c in chunks])
    valid = (words & 1).astype(bool)
    events = np.column_stack([
        timestamps[valid],
        ((words[valid] >> 17) & 0x7FFF).astype(np.int64),
        ((words[valid] >> 2) & 0x7FFF).astype(np.int64),
        ((words[valid] >> 1) & 1).astype(np.int64)])
    if max_events is not None:
        events = events[:max_events]
    return np.ascontiguousarray(events)


def iter_dvsd22(root: Path = PROJECT_ROOT / "DVSD22/Recordings",
                max_events: Optional[int] = None) -> Iterator[Recording]:
    """Falling-drop DAVIS346 recordings from the DVSD22 disdrometer release (Micev et al.,
    doi 10.5194/amt-17-335-2024). A drop-metrology corpus, not a denoising benchmark.

    Calibration clips are excluded: they record a static caliper and carry no drops. The
    drop-creation frequency stays in the Recording name because it is the corpus's reason
    for being here - a controlled signal-rate axis at a fixed noise regime.
    """

    for session_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        session = "HDD" if "HDD" in session_dir.name else "IV"
        for path in sorted(session_dir.glob("*.aedat")):
            if "calibration" in path.name:
                continue
            stem = path.stem                       # Davis346blue_2021-07-19_2x45Hz
            parts = stem.split("_", 2)
            tag = parts[2] if len(parts) > 2 else "staircase"
            events = read_aedat2(path, max_events)
            events, _ = _sorted(events, None)
            yield Recording(events, None, 346, 260, f"DVSD22/{session}/{tag}", False)
