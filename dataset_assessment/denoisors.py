"""Adapter giving every denoiser the same interface: events in, per-event score out.

Convention: **lower score = more likely signal**, so `esr.retain_mask` semantics are uniform
across methods and a retention `r` always means "keep the r fraction the method likes most".

Classical denoisers in cuke-emlb are BINARY (keep/drop), not scoring. To place them on the
same retention axis as scoring methods, a binary decision is converted into a two-level
score (0.0 kept, 1.0 dropped). This means their MESR@r curve is only defined at their own
operating point plus the trivial endpoints - a limitation that must be stated in the paper,
not hidden: binary methods cannot be swept, which is itself an argument for the protocol.

Two corrections to the boundary as originally planned, both forced by the built API:

1. The class is `dv_toolkit.EventStorage`, not `EventStore`. `EventStore` is
   dv-processing's type, reachable from a storage via `.toEventStore()`; the denoisors'
   `accept()` takes the toolkit storage. Verified against cuke-emlb's own
   `eval_denoisor.py`, which is the authoritative usage.
2. Kept events are recovered by an ordered subsequence merge, not by a set of `(t, x, y)`
   keys. Measured on `DND21/1hz_hotel-bar`: `generateEvents()` returns an exact ordered
   subsequence of the input (198,550/198,550 matched, indices strictly increasing), while
   the same recording contains 54 groups of events sharing a `(t, x, y)` coordinate that a
   set could not disambiguate. The merge is exact and recovers indices, which a set cannot.

Cost at 897,000 events: 0.63 s to fill the storage (`push_back` is per-event from Python,
0.69 us/event, and the toolkit exposes no bulk constructor), 0.03 s to denoise, 0.11 s to
merge. Budget ~0.8 s per (method, recording) at the 1 M cap.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUKE = PROJECT_ROOT / "cuke-emlb"

CLASSICAL = ("dwf", "evflow", "knoise", "red", "ts", "ynoise")

#: Not a denoiser. A label-driven upper bound, available only on labeled recordings, kept
#: out of `available_methods()` so it can never enter a ranking by accident.
ORACLE = "label_oracle"

#: Also not a denoiser. A seeded uniform-random ranking: it removes signal and noise in
#: exactly the proportion they occur. Any MESR gain it shows at r < 1 is a gain from
#: discarding events, not from discriminating them.
#:
#: It is the necessary control for `raw`. `raw` gives every event the same score, so
#: `retain_mask`'s stable argsort keeps the temporally-first r fraction of each block -
#: a periodic decimation that concentrates the survivors in time. Comparing `raw` against
#: `random_null` separates "MESR rewards dropping events" from "MESR rewards the particular
#: temporal structure that tie-breaking happens to produce".
RANDOM_NULL = "random_null"
RANDOM_NULL_SEED = 20260726

#: The zero-parameter analytic denoiser of the sibling project, as two rows.
#:
#: `native3d` is the raw continuous score `s = beta*G - log1p(support)` (the adopted
#: Phase-14b operand, radius-2 scoring ring, radius-1 gate key). It is one of the few
#: methods here that can actually be swept, so unlike the six binary classical filters
#: its AUC_r is a real curve rather than two trivial endpoints.
#:
#: `native3d_gated` is the same score under the method's own frozen `gated_retention`
#: rule, converted to the usual two-level form so it carries a native operating point.
#: Reporting both separates "how well does the score rank" from "where does the method
#: choose to sit".
#:
#: Optional, exactly like the classical methods: it needs the sibling checkout, and is
#: silently absent from `available_methods()` when that is missing.
NATIVE3D = "native3d"
NATIVE3D_GATED = "native3d_gated"
NATIVE3D_METHODS = (NATIVE3D, NATIVE3D_GATED)
NATIVE3D_ROOT = PROJECT_ROOT.parent / "zero-parameter-event-denoising"

#: Frozen by the sibling project; do not retune here. Radius 1 keys the corr(G,R) gate,
#: radius 2 supplies the support ring, and 10,240 is the block the method is defined on.
NATIVE3D_BLOCK = 10_240
NATIVE3D_TAUS_US = (250.0,)
NATIVE3D_GATE_RADIUS = 1
NATIVE3D_SCORE_RADIUS = 2


def available_methods() -> List[str]:
    """Raw is always available; classical methods require the cuke-emlb build (Task 2)."""

    methods = ["raw"]
    try:
        sys.path.insert(0, str(CUKE))
        import dv_toolkit  # noqa: F401
        methods.extend(CLASSICAL)
    except ImportError:
        pass
    if _native3d_available():
        methods.extend(NATIVE3D_METHODS)
    return methods


def _native3d_available() -> bool:
    return (NATIVE3D_ROOT / "native3d_ssm" / "phase10_emlb_blind.py").is_file()


def _raw_scores(recording) -> np.ndarray:
    return np.zeros(len(recording.events), dtype=np.float32)


def _oracle_scores(recording) -> np.ndarray:
    """Perfect per-event knowledge: signal 0, noise 1.

    This is a METRIC ORACLE. It answers "what is the best MESR any ranking could reach at
    this retention", which bounds every method's curve from above. It is not a denoiser and
    must never be reported as one.
    """

    if recording.labels is None:
        raise ValueError(f"{recording.name} has no labels; the oracle is undefined")
    return recording.labels.astype(np.float32)


def _random_null_scores(recording) -> np.ndarray:
    """A seeded uniform ranking, independent of the events. Keeps a random r fraction."""

    rng = np.random.default_rng(RANDOM_NULL_SEED)
    return rng.random(len(recording.events), dtype=np.float32)


#: One-entry memo so scoring `native3d` and `native3d_gated` on the same recording does
#: not run the state bank twice. Keyed by identity of the events array, which the loader
#: keeps alive for the whole per-recording sweep.
_NATIVE3D_CACHE: dict = {}


def _native3d_raw(recording):
    """`(support_score, per_block_corr)` from the sibling project's frozen scorer.

    Imported lazily: `native3d_ssm/__init__.py` pulls torch via its classifier module, and
    this repo is otherwise torch-free. The import cost lands only on runs that ask for it.
    """

    key = (id(recording.events), len(recording.events))
    if key in _NATIVE3D_CACHE:
        return _NATIVE3D_CACHE[key]

    if not _native3d_available():
        raise ValueError(
            f"{NATIVE3D} needs the sibling checkout at {NATIVE3D_ROOT}, which is absent")
    if str(NATIVE3D_ROOT) not in sys.path:
        sys.path.insert(0, str(NATIVE3D_ROOT))
    from native3d_ssm.core import StateConfig
    from native3d_ssm.phase10_emlb_blind import score_stream_both

    events = recording.events
    structured = np.empty(len(events), dtype=[("timestamp", "<i8"), ("x", "<i8"),
                                              ("y", "<i8"), ("polarity", "<i8")])
    structured["timestamp"] = events[:, 0]
    structured["x"] = events[:, 1]
    structured["y"] = events[:, 2]
    structured["polarity"] = events[:, 3]

    gate = StateConfig(width=recording.width, height=recording.height,
                       taus_us=NATIVE3D_TAUS_US, radius=NATIVE3D_GATE_RADIUS)
    score = StateConfig(width=recording.width, height=recording.height,
                        taus_us=NATIVE3D_TAUS_US, radius=NATIVE3D_SCORE_RADIUS)
    _old, _new, support, corrs, _betas = score_stream_both(
        structured, gate, NATIVE3D_BLOCK, score)

    _NATIVE3D_CACHE.clear()
    _NATIVE3D_CACHE[key] = (support.astype(np.float32), corrs)
    return _NATIVE3D_CACHE[key]


def _native3d_scores(recording) -> np.ndarray:
    """The continuous score. Lower = more likely signal, matching this module."""

    support, _corrs = _native3d_raw(recording)
    return support


def _native3d_gated_scores(recording) -> np.ndarray:
    """The deployed method: the same score under its own frozen per-block retention rule.

    Emitted as the usual two-level score so it lands in the native-operating-point
    ranking beside the binary classical filters.
    """

    support, corrs = _native3d_raw(recording)
    if str(NATIVE3D_ROOT) not in sys.path:
        sys.path.insert(0, str(NATIVE3D_ROOT))
    from native3d_ssm.phase10_emlb_blind import adaptive_mask

    keep, _retentions = adaptive_mask(support, corrs, NATIVE3D_BLOCK)
    return np.where(keep, 0.0, 1.0).astype(np.float32)


def _to_storage(events: np.ndarray):
    """Copy an `[t, x, y, p]` int64 array into a `dv_toolkit.EventStorage`."""

    sys.path.insert(0, str(CUKE))
    import dv_toolkit as kit

    storage = kit.EventStorage()
    push = storage.push_back
    for timestamp, x, y, polarity in events.tolist():
        push(timestamp, x, y, bool(polarity))
    return storage


def _kept_indices(events: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """Indices of `kept` within `events`, by ordered subsequence merge.

    `generateEvents()` preserves input order and returns a subsequence, so a single
    forward scan recovers the exact indices even when several events share a `(t, x, y)`
    coordinate. Any failure to consume every kept event means that assumption broke, and
    the caller must hear about it rather than silently scoring the wrong events.
    """

    source = list(zip(events[:, 0].tolist(), events[:, 1].tolist(),
                      events[:, 2].tolist(), events[:, 3].tolist()))
    target = list(zip(kept["timestamp"].astype(np.int64).tolist(),
                      kept["x"].astype(np.int64).tolist(),
                      kept["y"].astype(np.int64).tolist(),
                      (kept["polarity"].astype(np.int64) > 0).astype(np.int64).tolist()))
    indices = np.empty(len(target), dtype=np.int64)
    i = j = 0
    while j < len(target) and i < len(source):
        if source[i] == target[j]:
            indices[j] = i
            j += 1
        i += 1
    if j != len(target):
        raise RuntimeError(
            f"matched only {j} of {len(target)} kept events as an ordered subsequence; "
            "generateEvents() no longer preserves input order, so scores cannot be "
            "attributed to input events. Do not trust any result from this method.")
    return indices


def _classical_scores(name: str, recording) -> np.ndarray:
    """Run a cuke-emlb denoiser and convert its keep/drop decision to a 2-level score."""

    sys.path.insert(0, str(CUKE))
    from configs import denoisors

    model = getattr(denoisors, name)((recording.width, recording.height), {})
    model.accept(_to_storage(recording.events))
    kept = model.generateEvents().numpy()

    scores = np.ones(len(recording.events), dtype=np.float32)
    if len(kept):
        scores[_kept_indices(recording.events, kept)] = 0.0
    return scores


def score_events(method: str, recording) -> np.ndarray:
    if method == "raw":
        return _raw_scores(recording)
    if method == ORACLE:
        return _oracle_scores(recording)
    if method == RANDOM_NULL:
        return _random_null_scores(recording)
    if method == NATIVE3D:
        return _native3d_scores(recording)
    if method == NATIVE3D_GATED:
        return _native3d_gated_scores(recording)
    if method in CLASSICAL:
        return _classical_scores(method, recording)
    raise ValueError(f"unknown method {method!r}; available: {available_methods()}")


def is_oracle(method: str) -> bool:
    return method == ORACLE


def is_null(method: str) -> bool:
    """True for the two reference rows that do no discrimination at all."""

    return method in ("raw", RANDOM_NULL)


def native_retention(scores: np.ndarray) -> float:
    """The method's own operating point: the share it scores 0, i.e. keeps.

    Defined only for methods that emit a keep/drop decision, which is every classical
    denoisor plus `raw` and `label_oracle`. A continuous scorer such as `random_null` has
    no operating point of its own - it has a whole curve - and reporting "share scored
    exactly 0" for it would silently return 0.0 and drag a meaningless point into the
    native-operating-point ranking. NaN says so instead.
    """

    unique = np.unique(scores)
    if unique.size > 2 or not np.isin(unique, (0.0, 1.0)).all():
        return float("nan")
    return float((scores == 0.0).mean())


def has_native_operating_point(method: str) -> bool:
    """True for methods that emit a keep/drop decision rather than a continuous score.

    `random_null` and `native3d` are the exceptions: a uniform ranking has no operating
    point of its own, and `native3d` is a continuous score whose operating point is carried
    by its `native3d_gated` companion row. Both belong on the retention curve but not in
    any native-operating-point comparison.
    """

    return method in CLASSICAL or method in ("raw", ORACLE, NATIVE3D_GATED)
