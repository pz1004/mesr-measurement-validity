"""Which input event did a filter keep? Identical events, and what the packet route recovers.

A classical filter's output carries only event values, `(t, x, y, p)`, so the adapter maps
each output event back to the input by an ordered merge. Inputs hold identical events - the
same `(t, x, y, p)` more than once - and among identical events the merge cannot tell which
copy the filter kept. `denoisors._decisions` therefore feeds each filter its input in
packets that never hold two identical events, where the merge is exact.

This module measures, on the inputs exactly as the experiments read them:

* **How common identical events are.** Per corpus, the groups of identical events, the
  events in them and the largest group; on the labelled corpora, the groups holding both a
  signal and a noise event - the only ones whose attribution can change a label count.
* **That the packet route returns the one-pass output.** For every (recording, filter) the
  experiments run - DND21 and DVSCLEAN as released and with the busiest pixels removed,
  E-MLB, DVSD22, Pure_BA, every DVS Gesture test clip and every trained-probe sample - the
  filter is also run in one pass, and the packet output must equal it event for event; a
  difference raises. Equal outputs cannot show which of two identical copies each route kept:
  that the decisions themselves are unchanged rests on how the filters decide
  (`denoisors`, module docstring), and is checked index for index against a
  one-event-at-a-time run in `tests/test_denoisors.py`.
* **What a whole-stream merge would have attributed differently.** The earliest-copy match
  over the one-pass output, set against the packet route: how many kept events it places on
  a different input event and, on the labelled corpora, how far that moves the signal count
  over the events MESR scores.

Writes `results/identity_audit.json` once every corpus is done; progress after each corpus
goes to `identity_audit.partial.json`, so a failed run never replaces a complete artifact.
Run:

    python -m dataset_assessment.identity_audit --workers 8
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from . import denoisors
from .esr import SLICE

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "identity_audit.json"
MAX_EVENTS = 1_000_000


def identical_groups(events: np.ndarray, labels: Optional[np.ndarray] = None) -> Dict:
    """Groups of two or more identical `(t, x, y, p)` events, and how they are labelled."""

    out = {"events": int(len(events)), "groups": 0, "events_in_groups": 0, "largest": 0}
    if labels is not None:
        out["mixed_groups"] = 0
    if len(events) < 2:
        return out
    order = np.lexsort(events.T[::-1])
    ordered = events[order]
    repeat = np.all(ordered[1:] == ordered[:-1], axis=1)
    starts = np.flatnonzero(np.r_[True, ~repeat])
    sizes = np.diff(np.r_[starts, len(events)])
    grouped = sizes >= 2
    out.update(groups=int(grouped.sum()), events_in_groups=int(sizes[grouped].sum()),
               largest=int(sizes.max()))
    if labels is not None:
        signal = np.add.reduceat((labels[order] == 0).astype(np.int64), starts)
        out["mixed_groups"] = int((grouped & (signal > 0) & (signal < sizes)).sum())
    return out


def _one_pass(name: str, recording) -> np.ndarray:
    """The filter run once over the whole input, as `[t, x, y, p]` rows."""

    denoisors._cuke_first()
    from configs import denoisors as configs

    model = getattr(configs, name)((recording.width, recording.height), {})
    model.accept(denoisors._to_storage(recording.events))
    kept = model.generateEvents().numpy()
    return np.column_stack([kept["timestamp"], kept["x"], kept["y"],
                            kept["polarity"] > 0]).astype(np.int64)


def compare_routes(name: str, recording) -> Dict:
    """One filter on one recording: packet route against one pass and whole-stream merge."""

    events, labels = recording.events, recording.labels
    exact = np.flatnonzero(denoisors._classical_scores(name, recording) == 0.0)
    rows = _one_pass(name, recording)
    if not np.array_equal(events[exact], rows):
        raise RuntimeError(f"{recording.name} {name}: the packet output differs from the "
                           "one-pass output; packets changed a filter decision")
    whole = (denoisors._kept_indices(events, _structured(rows)) if len(rows)
             else np.empty(0, dtype=np.int64))
    cell = {"recording": recording.name, "method": name, "kept": int(len(exact)),
            "moved": int((whole != exact).sum())}
    if labels is not None:
        scored = SLICE * (len(exact) // SLICE)
        cell["scored"] = int(scored)
        cell["tp_scored"] = int((labels[exact[:scored]] == 0).sum())
        cell["tp_scored_whole_merge"] = int((labels[whole[:scored]] == 0).sum())
    return cell


def _structured(rows: np.ndarray) -> np.ndarray:
    kept = np.empty(len(rows), dtype=[("timestamp", "<i8"), ("x", "<i8"),
                                      ("y", "<i8"), ("polarity", "<i8")])
    kept["timestamp"], kept["x"], kept["y"], kept["polarity"] = rows.T
    return kept


def _groups_and_routes(recording):
    """One recording's groups and every classical filter's routes; module level so the pool
    can pickle it."""

    return ({"recording": recording.name,
             **identical_groups(recording.events, recording.labels)},
            [compare_routes(name, recording) for name in denoisors.CLASSICAL])


def _labelled(dataset: str, hot_pixel_removal: bool) -> Iterable:
    from .native_oracle import LOADERS, prepare

    for rec in LOADERS[dataset]():
        rec, _summary = prepare(rec, MAX_EVENTS, hot_pixel_removal)
        yield rec


def _capped(dataset: str) -> Iterable:
    """As `run_benchmark` reads a corpus: through its loader, capped at 10^6 events."""

    from .null_matched_span import _recordings

    return _recordings(dataset)


def _gesture_clips() -> Iterable:
    from .downstream_gesture_frozen import HEIGHT, WIDTH, load_test_clips
    from .readers import Recording

    events, _labels, clips, _coverage = load_test_clips()
    for clip, stream in zip(clips, events):
        yield Recording(stream, None, WIDTH, HEIGHT, clip["clip_id"], False)


def _gesture_samples() -> Iterable:
    from .downstream_gesture import HEIGHT, WIDTH, _trials, load_samples
    from .readers import Recording

    for split in ("trials_to_train.txt", "trials_to_test.txt"):
        events, _labels, _coverage = load_samples(_trials(split))
        for index, stream in enumerate(events):
            yield Recording(stream, None, WIDTH, HEIGHT, f"{split}:{index}", False)


def summarise_groups(per_recording: List[Dict]) -> Dict:
    keys = ("events", "groups", "events_in_groups")
    out = {key: int(sum(r[key] for r in per_recording)) for key in keys}
    out["recordings"] = len(per_recording)
    out["recordings_with_groups"] = int(sum(r["groups"] > 0 for r in per_recording))
    out["largest"] = int(max((r["largest"] for r in per_recording), default=0))
    if per_recording and "mixed_groups" in per_recording[0]:
        out["mixed_groups"] = int(sum(r["mixed_groups"] for r in per_recording))
        out["max_mixed_in_one_recording"] = int(max(r["mixed_groups"] for r in per_recording))
    return out


def summarise_routes(cells: List[Dict]) -> Dict:
    out = {"cells": len(cells), "one_pass_equal": len(cells),
           "cells_moved": int(sum(c["moved"] > 0 for c in cells)),
           "events_moved": int(sum(c["moved"] for c in cells)),
           "max_moved": int(max((c["moved"] for c in cells), default=0))}
    if cells and "tp_scored" in cells[0]:
        changes = [c["tp_scored_whole_merge"] - c["tp_scored"] for c in cells]
        out["tp_changed_cells"] = int(sum(d != 0 for d in changes))
        out["max_tp_overcount"] = int(max(0, max(changes, default=0)))
        out["max_tp_undercount"] = int(max(0, -min(changes, default=0)))
    return out


def audit_corpus(name: str, recordings: Iterable, routes: bool, workers: int) -> Dict:
    started = time.perf_counter()
    per_recording: List[Dict] = []
    cells: List[Dict] = []
    if routes:
        # Spawned, not forked: the parent holds pandas' and MKL's thread pools.
        with get_context("spawn").Pool(workers) as pool:
            for index, (groups, rows) in enumerate(
                    pool.imap(_groups_and_routes, recordings, chunksize=1)):
                per_recording.append(groups)
                cells.extend(rows)
                if index % 48 == 0:
                    print(f"[{name} {index + 1}] {groups['recording']} "
                          f"({time.perf_counter() - started:.0f}s)", flush=True)
    else:
        for rec in recordings:
            per_recording.append({"recording": rec.name,
                                  **identical_groups(rec.events, rec.labels)})
    block = {"summary": summarise_groups(per_recording), "per_recording": per_recording,
             "wall_seconds": time.perf_counter() - started}
    if routes:
        block["routes"] = summarise_routes(cells)
        block["cells"] = cells
    print(f"{name}: {block['summary']}" + (f"; routes {block['routes']}" if routes else ""),
          flush=True)
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if "dwf" not in denoisors.available_methods():
        raise SystemExit("needs the cuke-emlb build: the routes are compared on the filters")
    started = time.perf_counter()
    plan = [
        ("dnd21", lambda: _labelled("dnd21", False)),
        ("dnd21_hotpixel", lambda: _labelled("dnd21", True)),
        ("dvsclean", lambda: _labelled("dvsclean", False)),
        ("dvsclean_hotpixel", lambda: _labelled("dvsclean", True)),
        ("gesture_test_clips", _gesture_clips),
        ("gesture_trained_samples", _gesture_samples),
        ("dvsd22", lambda: _capped("dvsd22")),
        ("pure_ba", lambda: _capped("pure_ba")),
        ("emlb", lambda: _capped("emlb")),
    ]
    payload: Dict[str, Any] = {"max_events": MAX_EVENTS, "filters": list(denoisors.CLASSICAL),
                               "corpora": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.stem + ".partial.json")
    for name, recordings in plan:
        payload["corpora"][name] = audit_corpus(name, recordings(), True, args.workers)
        payload["wall_seconds"] = time.perf_counter() - started
        partial.write_text(json.dumps(payload, indent=2) + "\n")
    partial.replace(args.out)
    print(f"\nWrote {args.out} ({payload['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
