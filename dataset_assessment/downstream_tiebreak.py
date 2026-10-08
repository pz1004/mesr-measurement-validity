"""Does the frozen classifier's verdict on quota-adapted RED depend on the quota's tie-break?

A quota-adapted filter keeps a fixed count in each 10,240-event block: its accepted events
first, then rejected ones, and within each group the earliest, because `esr.retain_mask`
sorts stably. Arrival order is one tie-break of many, and at r = 0.6 it decides which of
RED's accepted events survive the quota. This replaces it with a random order inside each
group and keeps everything else - the block quotas, accepted before rejected, the released
classifier, the frames - so a verdict that survives it does not rest on the tie-break.

For each of ten seeds the six filters are re-selected at r = 0.6 and evaluated beside the
two label-blind controls, whose selections no tie-break touches: the block-prefix control
(every score equal, so arrival order *is* the selection) and the random subsampler (scores
already continuous). Each seed's conditions are named `r060t<seed>__<method>` with the
block-prefix control as `r060t<seed>__unfiltered`, so the released evaluator's own pairing
reports every filter minus the control, resampled over the six test subjects. The random keys
depend on the seed and the clip only, so within a seed the six filters share them: common random
numbers, which pair the filters' random orders the way the clips pair their accuracies.

The arrival-order reference is read from `downstream_gesture_frozen.json`, whose SHA-256 the
report records. Frames are written to a directory that must start empty, and only the files
this run wrote are deleted; the report is rewritten after every seed, with the evaluator's
per-clip predictions.

Writes `results/downstream_gesture_tiebreak.json`. Run:

    python -m dataset_assessment.downstream_tiebreak
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .denoisors import CLASSICAL, RANDOM_NULL, score_events
from .downstream_gesture_frozen import (BINS, CHECKPOINT, HEIGHT, PROJECT_ROOT, SNN_PYTHON,
                                        WIDTH, check_gate, frames_for, load_test_clips,
                                        run_evaluator)
from .esr import mesr, retain_mask
from .readers import Recording
from .sew7b.frames import events_to_number_frames

OUT = PROJECT_ROOT / "results/downstream_gesture_tiebreak.json"
FRAMES = PROJECT_ROOT / "results/downstream_gesture_tiebreak_frames"
FROZEN = PROJECT_ROOT / "results/downstream_gesture_frozen.json"

RETENTION = 0.6
TIEBREAK_SEEDS = tuple(range(20261006, 20261016))
CONTROL = "unfiltered"            # the evaluator's reference name; here the block-prefix control


def tiebroken(accepted: np.ndarray, seed: int, clip_index: int) -> np.ndarray:
    """A keep/drop decision as a score that ranks accepted events first, in random order.

    Accepted events score in [0, 0.5) and rejected ones in [1, 1.5), so `retain_mask` still
    fills each block's quota from the accepted events first; within each group the order is
    a uniform random permutation, keyed on the seed and the clip.
    """

    rng = np.random.default_rng([seed, clip_index])
    return np.where(accepted, 0.0, 1.0) + 0.5 * rng.random(len(accepted))


def condition(seed: int, method: str) -> str:
    return f"r{RETENTION * 100:03.0f}t{seed}__{CONTROL if method == 'raw' else method}"


def _write(path: Path, events: Sequence[np.ndarray], keeps: Sequence[np.ndarray]) -> float:
    """Frames of every clip's kept events to `path`; returns the mean MESR of those events."""

    stack = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16,
                                      shape=(len(events), BINS, 2, HEIGHT, WIDTH))
    values = []
    for index, (clip, keep) in enumerate(zip(events, keeps)):
        stack[index] = frames_for(clip, keep, events_to_number_frames)
        value = mesr(clip[keep][:, 1], clip[keep][:, 2], WIDTH, HEIGHT)
        if np.isfinite(value):
            values.append(value)
    stack.flush()
    del stack
    return float(np.mean(values))


def _manifest(path: Path, labels_path: Path, clips: List[dict], n: int,
              conditions: Dict[str, dict]) -> Path:
    manifest = {"format_version": 1,
                "dataset": "IBM DVS Gesture raw AEDAT 3.1, official test split, uncapped",
                "labels": str(labels_path.resolve()), "clips": clips,
                "shape": [n, BINS, 2, HEIGHT, WIDTH], "conditions": conditions}
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def summarise(rows: List[dict]) -> Dict:
    """Per method over the seeds, and RED's place among the eight methods in each seed."""

    by_method: Dict[str, List[dict]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    methods = {
        method: {"accuracy": [r["accuracy"] for r in group],
                 "accuracy_mean": float(np.mean([r["accuracy"] for r in group])),
                 "accuracy_min": float(min(r["accuracy"] for r in group)),
                 "accuracy_max": float(max(r["accuracy"] for r in group)),
                 "mesr_mean": float(np.mean([r["mesr"] for r in group])),
                 "mesr_min": float(min(r["mesr"] for r in group)),
                 "mesr_max": float(max(r["mesr"] for r in group))}
        for method, group in sorted(by_method.items())}
    seeds = sorted({row["seed"] for row in rows})
    counts = {"red_least_accurate_in_seeds": 0, "red_least_accurate_alone_in_seeds": 0,
              "red_first_by_mesr_in_seeds": 0, "red_first_by_mesr_alone_in_seeds": 0}
    for seed in seeds:
        cell = {row["method"]: row for row in rows if row["seed"] == seed}
        lowest = min(row["accuracy"] for row in cell.values())
        tied = [m for m, row in cell.items() if row["accuracy"] == lowest]
        counts["red_least_accurate_in_seeds"] += "red" in tied
        counts["red_least_accurate_alone_in_seeds"] += tied == ["red"]
        highest = max(row["mesr"] for row in cell.values())
        first = [m for m, row in cell.items() if row["mesr"] == highest]
        counts["red_first_by_mesr_in_seeds"] += "red" in first
        counts["red_first_by_mesr_alone_in_seeds"] += first == ["red"]
    return {"methods": methods, "seeds": len(seeds), **counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TIEBREAK_SEEDS))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--frames-dir", type=Path, default=FRAMES)
    parser.add_argument("--python", type=Path, default=SNN_PYTHON)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"released checkpoint not found at {args.checkpoint}")
    if args.frames_dir.exists() and any(args.frames_dir.iterdir()):
        raise SystemExit(f"{args.frames_dir} is not empty; this run deletes what it writes "
                         "there, so it must start with an empty directory")

    started = time.perf_counter()
    written: List[Path] = []

    def owned(path: Path) -> Path:
        written.append(path)
        return path

    events, labels, clips, _coverage = load_test_clips()
    print(f"{len(events)} clips loaded ({time.perf_counter() - started:.0f}s)", flush=True)
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    labels_path = owned(args.frames_dir / "labels.npy")
    np.save(labels_path, labels)

    def recording(stream: np.ndarray) -> Recording:
        return Recording(stream, None, WIDTH, HEIGHT, "", False)

    # The two controls: no tie-break reaches them, so their frames are built once.
    shared: Dict[str, dict] = {}
    for method in ("raw", RANDOM_NULL):
        keeps = [retain_mask(score_events(method, recording(e)), RETENTION) for e in events]
        path = owned(args.frames_dir / f"control_{method}.npy")
        shared[method] = {"frames": str(path.resolve()), "mesr": _write(path, events, keeps)}
    clean = owned(args.frames_dir / "clean.npy")
    _write(clean, events, [np.ones(len(e), dtype=bool) for e in events])

    accepted = {method: [score_events(method, recording(e)) == 0.0 for e in events]
                for method in CLASSICAL}
    print(f"filters scored ({time.perf_counter() - started:.0f}s)", flush=True)

    arrival, frozen_sha256 = None, None
    if FROZEN.is_file():
        frozen_sha256 = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
        frozen = json.loads(FROZEN.read_text())
        arrival = {row["method"]: {"accuracy": row["accuracy"], "mesr": row["mesr"]}
                   for row in frozen["conditions"] if row["retention"] == RETENTION}

    rows: List[dict] = []
    comparisons: Dict[str, dict] = {}
    predictions: Dict[str, List[int]] = {}
    target = None
    gate, environment = None, None
    methods = ("raw", RANDOM_NULL) + tuple(CLASSICAL)

    def write_report(complete: bool) -> dict:
        report = {
            "protocol": {"retention": RETENTION, "seeds": list(args.seeds),
                         "tie_break": "uniform random order within accepted and within "
                                      "rejected events, per block quota unchanged; keys shared "
                                      "by the six filters within a seed and clip",
                         "controls": "block-prefix control (`unfiltered`) and random "
                                     "subsampler, untouched by the tie-break",
                         "classifier": "released SEW 7B-Net, frozen, fp32 (evaluator --no-amp)",
                         "test_samples": len(events)},
            "complete": complete, "gate": gate, "environment": environment, "rows": rows,
            "comparisons": comparisons, "summary": summarise(rows) if rows else None,
            "arrival_order": arrival, "arrival_order_source_sha256": frozen_sha256,
            "target": target, "predictions": predictions,
            "wall_seconds": time.perf_counter() - started,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report

    for number, seed in enumerate(args.seeds):
        conditions = {condition(seed, m): {"frames": shared[m]["frames"], "method": m,
                                           "retention": RETENTION}
                      for m in ("raw", RANDOM_NULL)}
        mesrs = {m: shared[m]["mesr"] for m in ("raw", RANDOM_NULL)}
        seed_frames = []
        for method in CLASSICAL:
            keeps = [retain_mask(tiebroken(a, seed, i), RETENTION)
                     for i, a in enumerate(accepted[method])]
            path = owned(args.frames_dir / f"{condition(seed, method)}.npy")
            seed_frames.append(path)
            mesrs[method] = _write(path, events, keeps)
            conditions[condition(seed, method)] = {"frames": str(path.resolve()),
                                                   "method": method, "retention": RETENTION}
        if gate is None:
            conditions["clean__unfiltered"] = {"frames": str(clean.resolve()),
                                               "method": "raw", "retention": 1.0}
        manifest = _manifest(owned(args.frames_dir / f"manifest_{seed}.json"), labels_path,
                             clips, len(events), conditions)
        evaluated = run_evaluator(manifest, owned(args.frames_dir / f"evaluation_{seed}.json"),
                                  args.python, args.checkpoint)
        if gate is None:
            gate, environment = check_gate(evaluated), evaluated.get("environment")
            if not gate["reproduced"]:
                raise SystemExit("gate failed; no tie-break condition is reported")
        elif evaluated.get("environment") != environment:
            raise SystemExit(f"the evaluator's environment changed at seed {seed}: "
                             f"{evaluated.get('environment')} against {environment}")
        prefix = condition(seed, "raw").split("__")[0]
        for method in methods:
            name = condition(seed, method)
            cell = evaluated["conditions"][name]
            rows.append({"seed": seed, "method": method, "accuracy": cell["top1"],
                         "macro_f1": cell["macro_f1"], "mesr": mesrs[method]})
            predictions[name] = [int(v) for v in cell["prediction"]]
            if target is None:
                target = [int(v) for v in cell["target"]]
            key = f"{name}_vs_{prefix}__{CONTROL}"
            if method != "raw":
                if key not in evaluated["comparisons"]:
                    raise KeyError(f"the evaluator reported no {key}; its pairing of "
                                   "conditions with the control has changed")
                value = evaluated["comparisons"][key]
                comparisons[key] = {k: value[k] for k in
                                    ("accuracy_delta", "accuracy_delta_95ci", "by_subject",
                                     "cluster_bootstrap_replicates",
                                     "mcnemar_exact_two_sided_p")}
        for path in seed_frames:
            path.unlink()
        write_report(complete=False)
        red = next(r for r in rows if r["seed"] == seed and r["method"] == "red")
        print(f"[seed {number + 1}/{len(args.seeds)}] {seed}: red accuracy "
              f"{red['accuracy']:.4f}, mesr {red['mesr']:.4f} "
              f"({time.perf_counter() - started:.0f}s)", flush=True)

    report = write_report(complete=True)
    for path in written:
        if path.exists():
            path.unlink()
    if not any(args.frames_dir.iterdir()):
        args.frames_dir.rmdir()
    summary = report["summary"]
    print(f"\nred least accurate in {summary['red_least_accurate_in_seeds']} of "
          f"{summary['seeds']} orders ({summary['red_least_accurate_alone_in_seeds']} alone), "
          f"first by MESR in {summary['red_first_by_mesr_in_seeds']} "
          f"({summary['red_first_by_mesr_alone_in_seeds']} alone)")
    print(f"wrote {args.out} ({report['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
