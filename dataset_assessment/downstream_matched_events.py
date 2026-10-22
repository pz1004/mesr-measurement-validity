"""The frozen classifier on exactly the events MESR scores.

`downstream_gesture_frozen` hands the released classifier every event a procedure retains,
while MESR averages complete 30,000-event slices and drops the tail, so the two evaluators
read different event sets: MESR covers 91.4-96.4% of what the classifier sees. This module
removes that difference. For each of the eight procedures at each partial retention
r in {0.4, 0.5, 0.6, 0.8}, the classifier receives the scored prefix, the first
30,000 * floor(k / 30,000) of the k events the procedure retains (`native_oracle.scored_prefix`),
so both evaluators see the same events. Nothing is retrained.

The retained sets are the frozen run's own, and two checks say so. Each condition's MESR,
recomputed on the scored prefix, must equal the value `downstream_gesture_frozen.json` stores
(MESR never reads past the last complete slice, so the prefix and the whole retained set score
identically). And the full-stream RED condition at r = 0.6 is re-evaluated and must return the
stored predictions clip for clip, after the released checkpoint has passed the frozen run's gate.

Cohort. A clip whose retained stream holds no complete slice has no scored events, so neither
evaluator can read it. Each retention is evaluated on the clips that hold one: 287 of the 288
at r = 0.4 and all 288 at the others. Beside each, the full-stream accuracy on the same clips
is read from the frozen run's per-clip predictions
(`downstream_gesture_frozen_evaluation.json`), so the two readings differ only in the events.

Conditions are named `m<r>__<method>`, with the block-prefix control as `m<r>__unfiltered`, so
the released evaluator's own pairing reports every procedure minus the control, resampled over
the six test subjects. Frames are written to a directory that must start empty, and only the
files this run wrote are deleted; the report keeps the evaluator's per-clip predictions.

Writes `results/downstream_matched_events.json`. Run:

    python -m dataset_assessment.downstream_matched_events            # about 80 minutes
    python -m dataset_assessment.downstream_matched_events --smoke    # the gate alone
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .denoisors import score_events
from .downstream_gesture_frozen import (BINS, CHECKPOINT, HEIGHT, PAPER_METHODS, PROJECT_ROOT,
                                        SNN_PYTHON, WIDTH, check_gate, frames_for,
                                        load_test_clips, run_evaluator)
from .esr import mesr, retain_mask
from .native_oracle import scored_prefix
from .readers import Recording
from .sew7b.frames import events_to_number_frames

OUT = PROJECT_ROOT / "results/downstream_matched_events.json"
FRAMES = PROJECT_ROOT / "results/downstream_matched_events_frames"
FROZEN = PROJECT_ROOT / "results/downstream_gesture_frozen.json"
FROZEN_EVALUATION = PROJECT_ROOT / "results/downstream_gesture_frozen_evaluation.json"

RETENTIONS = (0.4, 0.5, 0.6, 0.8)
CONTROL = "unfiltered"            # the evaluator's reference name; here the block-prefix control
#: The full-stream condition re-evaluated against the frozen run's stored predictions.
REPRODUCED = ("red", 0.6)
MESR_TOLERANCE = 1e-12


def condition(method: str, retention: float, scored: bool = True) -> str:
    """`m<r>__<method>` on the scored prefix; `r<r>__<method>` on the whole retained stream.

    The second is the frozen run's own name, so it indexes that run's predictions.
    """

    tag = "m" if scored else "r"
    return f"{tag}{retention * 100:03.0f}__{CONTROL if method == 'raw' else method}"


def cohort(prefixes: Sequence[np.ndarray]) -> List[int]:
    """Indices of the clips whose scored prefix holds any event, the only ones MESR scores."""

    return [index for index, prefix in enumerate(prefixes) if prefix.any()]


def frozen_accuracy(evaluation: Dict, name: str, clips: Sequence[int]) -> float:
    """The frozen run's accuracy for condition `name`, restricted to `clips`."""

    cell = evaluation["conditions"][name]
    prediction = np.asarray(cell["prediction"])[list(clips)]
    target = np.asarray(cell["target"])[list(clips)]
    return float(np.mean(prediction == target))


def summarise(rows: Sequence[dict], comparisons: Dict[str, dict]) -> Dict:
    """Per retention: who is least and most accurate, who MESR ranks first, and RED - control.

    Ties are kept: `least_accurate` lists every procedure at the lowest accuracy, and the
    counts at the end distinguish RED among the least accurate from RED alone there.
    """

    from scipy.stats import spearmanr

    per_retention: Dict[str, dict] = {}
    for retention in sorted({row["retention"] for row in rows}):
        cell = {row["method"]: row for row in rows if row["retention"] == retention}
        lowest = min(c["accuracy"] for c in cell.values())
        highest = max(c["accuracy"] for c in cell.values())
        top = max(c["mesr"] for c in cell.values())
        methods = sorted(cell)
        rho = spearmanr([cell[m]["mesr"] for m in methods],
                        [cell[m]["accuracy"] for m in methods]).statistic
        key = f"{condition('red', retention)}_vs_{condition('raw', retention)}"
        contrast = comparisons.get(key)
        red_minus_control = None
        if contrast is not None:
            by_subject = [s["accuracy_delta"] for s in contrast["by_subject"].values()]
            red_minus_control = {
                "accuracy_delta": contrast["accuracy_delta"],
                "accuracy_delta_95ci": contrast["accuracy_delta_95ci"],
                "subjects": len(by_subject),
                "subjects_red_lower": int(sum(d < 0 for d in by_subject)),
                "subjects_tied": int(sum(d == 0 for d in by_subject)),
            }
        per_retention[f"{retention:g}"] = {
            "clips": cell[methods[0]]["clips"],
            "least_accurate": sorted(m for m, c in cell.items() if c["accuracy"] == lowest),
            "most_accurate": sorted(m for m, c in cell.items() if c["accuracy"] == highest),
            "first_by_mesr": sorted(m for m, c in cell.items() if c["mesr"] == top),
            "rho_mesr_accuracy": float(rho),
            "red_minus_control": red_minus_control,
        }
    least = [r for r, s in per_retention.items() if "red" in s["least_accurate"]]
    alone = [r for r, s in per_retention.items() if s["least_accurate"] == ["red"]]
    rhos = [s["rho_mesr_accuracy"] for s in per_retention.values()]
    return {"per_retention": per_retention,
            "red_least_accurate_at": least, "red_least_accurate_alone_at": alone,
            "mean_rho_mesr_accuracy": float(np.mean(rhos)) if rhos else float("nan")}


def _write_frames(path: Path, events: Sequence[np.ndarray], masks: Sequence[np.ndarray],
                  clips: Sequence[int]) -> None:
    """Frames of `masks[i]` over `events[i]` for each index in `clips`, in that order."""

    stack = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16,
                                      shape=(len(clips), BINS, 2, HEIGHT, WIDTH))
    for row, index in enumerate(clips):
        stack[row] = frames_for(events[index], masks[index], events_to_number_frames)
    stack.flush()
    del stack


def _manifest(path: Path, labels_path: Path, clips: List[dict], n: int,
              conditions: Dict[str, dict]) -> Path:
    manifest = {"format_version": 1,
                "dataset": "IBM DVS Gesture raw AEDAT 3.1, official test split, uncapped",
                "labels": str(labels_path.resolve()), "clips": clips,
                "shape": [n, BINS, 2, HEIGHT, WIDTH], "conditions": conditions}
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retentions", type=float, nargs="+", default=list(RETENTIONS))
    parser.add_argument("--smoke", action="store_true",
                        help="Check the released checkpoint's gate and stop.")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--frames-dir", type=Path, default=FRAMES)
    parser.add_argument("--python", type=Path, default=SNN_PYTHON)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"released checkpoint not found at {args.checkpoint}")
    for source in (FROZEN, FROZEN_EVALUATION):
        if not source.is_file():
            raise SystemExit(f"{source} not found; run `python -m "
                             "dataset_assessment.downstream_gesture_frozen` first")
    if args.frames_dir.exists() and any(args.frames_dir.iterdir()):
        raise SystemExit(f"{args.frames_dir} is not empty; this run deletes what it writes "
                         "there, so it must start with an empty directory")

    started = time.perf_counter()
    written: List[Path] = []

    def owned(path: Path) -> Path:
        written.append(path)
        return path

    def elapsed() -> str:
        return f"{time.perf_counter() - started:.0f}s"

    frozen = json.loads(FROZEN.read_text())
    evaluation = json.loads(FROZEN_EVALUATION.read_text())
    stored_mesr = {(row["method"], float(row["retention"])): float(row["mesr"])
                   for row in frozen["conditions"] if float(row["retention"]) < 1.0}
    stored_clips = {(row["method"], float(row["retention"])): int(row["evaluable_test_samples"])
                    for row in frozen["conditions"] if float(row["retention"]) < 1.0}
    unknown = [r for r in args.retentions if ("raw", float(r)) not in stored_mesr]
    if unknown:
        raise SystemExit(f"the frozen run has no condition at r = {unknown}; every retention "
                         "here is checked against it")
    if REPRODUCED[1] not in args.retentions:
        raise SystemExit(f"r = {REPRODUCED[1]} must be among the retentions: its full-stream "
                         f"{REPRODUCED[0]} condition is the end-to-end reproduction check")

    events, labels, clips, _coverage = load_test_clips()
    print(f"{len(events)} clips loaded ({elapsed()})", flush=True)
    target = np.asarray(evaluation["conditions"]["clean__unfiltered"]["target"])
    if not np.array_equal(target, labels):
        raise SystemExit("the frozen run's clip order differs from this load; its per-clip "
                         "predictions cannot be read against these clips")
    args.frames_dir.mkdir(parents=True, exist_ok=True)

    # The gate first: a broken classifier environment should fail in minutes, not after
    # the filters have been scored.
    labels_all = owned(args.frames_dir / "labels_all.npy")
    np.save(labels_all, labels)
    clean = owned(args.frames_dir / "clean__unfiltered.npy")
    _write_frames(clean, events, [np.ones(len(e), dtype=bool) for e in events],
                  range(len(events)))
    manifest = _manifest(owned(args.frames_dir / "manifest_gate.json"), labels_all, clips,
                         len(events), {"clean__unfiltered": {"frames": str(clean.resolve()),
                                                             "method": "raw",
                                                             "retention": 1.0}})
    evaluated = run_evaluator(manifest, owned(args.frames_dir / "evaluation_gate.json"),
                              args.python, args.checkpoint)
    gate, environment = check_gate(evaluated), evaluated.get("environment")
    if not gate["reproduced"]:
        raise SystemExit("gate failed; no matched-event condition is reported")
    clean.unlink()
    if args.smoke:
        for path in written:
            if path.exists():
                path.unlink()
        if not any(args.frames_dir.iterdir()):
            args.frames_dir.rmdir()
        print(f"gate reproduced ({elapsed()}); --smoke stops here")
        return

    def recording(stream: np.ndarray) -> Recording:
        return Recording(stream, None, WIDTH, HEIGHT, "", False)

    # One scoring per method; the scored prefixes are kept bit-packed (about 50 KB a clip).
    packed: Dict[tuple, List[np.ndarray]] = {}
    clip_mesr: Dict[tuple, List[float]] = {}
    reproduced_keep: List[np.ndarray] = []
    for number, method in enumerate(PAPER_METHODS, start=1):
        scores = [score_events(method, recording(e)) for e in events]
        for retention in args.retentions:
            keeps = [retain_mask(s, retention) for s in scores]
            prefixes = [scored_prefix(k) for k in keeps]
            values = []
            for clip, keep, prefix in zip(events, keeps, prefixes):
                on_prefix = mesr(clip[prefix][:, 1], clip[prefix][:, 2], WIDTH, HEIGHT)
                on_keep = mesr(clip[keep][:, 1], clip[keep][:, 2], WIDTH, HEIGHT)
                if not (on_prefix == on_keep or (np.isnan(on_prefix) and np.isnan(on_keep))):
                    raise AssertionError(f"{method} r={retention}: MESR of the scored prefix "
                                         f"({on_prefix}) differs from the retained set's "
                                         f"({on_keep})")
                values.append(on_prefix)
            finite = [v for v in values if np.isfinite(v)]
            stored, expected = stored_mesr[(method, retention)], stored_clips[(method, retention)]
            if len(finite) != expected or not abs(float(np.mean(finite)) - stored) <= MESR_TOLERANCE:
                raise AssertionError(f"{method} r={retention}: MESR over {len(finite)} clips is "
                                     f"{float(np.mean(finite))!r} against the frozen run's "
                                     f"{stored!r} over {expected}; the retained sets differ")
            packed[(method, retention)] = [np.packbits(p) for p in prefixes]
            clip_mesr[(method, retention)] = values
            if (method, retention) == REPRODUCED:
                reproduced_keep = keeps
        del scores
        print(f"[{number}/{len(PAPER_METHODS)}] {method} scored ({elapsed()})", flush=True)

    def prefixes_of(method: str, retention: float) -> List[np.ndarray]:
        return [np.unpackbits(p, count=len(e)).astype(bool)
                for p, e in zip(packed[(method, retention)], events)]

    rows: List[dict] = []
    comparisons: Dict[str, dict] = {}
    predictions: Dict[str, List[int]] = {}
    targets: Dict[str, List[int]] = {}
    cohorts: Dict[str, dict] = {}
    full_stream: Dict[str, Dict[str, float]] = {}
    reproduction: Optional[dict] = None

    def write_report(complete: bool) -> dict:
        report = {
            "protocol": {
                "retentions": list(args.retentions), "methods": list(PAPER_METHODS),
                "classified_events": "the scored prefix: the first 30,000 * floor(k / 30,000) "
                                     "of the k retained events (native_oracle.scored_prefix)",
                "cohort": "per retention, the clips whose retained stream holds a complete "
                          "slice; the full-stream accuracy beside it is read on the same clips",
                "classifier": "released SEW 7B-Net, frozen, fp32 (evaluator --no-amp)",
                "control": "block-prefix control, the evaluator's `unfiltered` reference",
                "frozen_source_sha256": _sha256(FROZEN),
                "frozen_evaluation_sha256": _sha256(FROZEN_EVALUATION),
                "test_samples": len(events),
            },
            "complete": complete, "gate": gate, "environment": environment,
            "reproduction": reproduction, "cohorts": cohorts, "rows": rows,
            "full_stream_accuracy": full_stream, "comparisons": comparisons,
            "summary": summarise(rows, comparisons) if rows else None,
            "targets": targets, "predictions": predictions,
            "wall_seconds": time.perf_counter() - started,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report

    for retention in args.retentions:
        tag = f"{retention:g}"
        masks = {method: prefixes_of(method, retention) for method in PAPER_METHODS}
        members = cohort(masks["raw"])
        for method, prefixes in masks.items():
            if cohort(prefixes) != members:
                raise AssertionError(f"r={retention}: {method}'s evaluable clips differ from "
                                     "the control's, but the block quota fixes the count alone")
        member_set = set(members)
        excluded = [clips[i]["clip_id"] for i in range(len(events)) if i not in member_set]
        cohorts[tag] = {"clips": len(members), "excluded": excluded}
        labels_path = owned(args.frames_dir / f"labels_{tag}.npy")
        np.save(labels_path, labels[members])
        conditions: Dict[str, dict] = {}
        paths: List[Path] = []
        for method in PAPER_METHODS:
            name = condition(method, retention)
            path = owned(args.frames_dir / f"{name}.npy")
            _write_frames(path, events, masks[method], members)
            paths.append(path)
            conditions[name] = {"frames": str(path.resolve()), "method": method,
                                "retention": retention}
        check_reproduction = (REPRODUCED[1] == retention and len(members) == len(events))
        if check_reproduction:
            name = condition(REPRODUCED[0], retention, scored=False)
            path = owned(args.frames_dir / f"{name}.npy")
            _write_frames(path, events, reproduced_keep, members)
            paths.append(path)
            conditions[name] = {"frames": str(path.resolve()), "method": REPRODUCED[0],
                                "retention": retention}
        manifest = _manifest(owned(args.frames_dir / f"manifest_{tag}.json"), labels_path,
                             [clips[i] for i in members], len(members), conditions)
        evaluated = run_evaluator(manifest, owned(args.frames_dir / f"evaluation_{tag}.json"),
                                  args.python, args.checkpoint)
        if evaluated.get("environment") != environment:
            raise SystemExit(f"the evaluator's environment changed at r={retention}: "
                             f"{evaluated.get('environment')} against {environment}")
        targets[tag] = [int(v) for v in evaluated["conditions"][condition("raw", retention)]
                        ["target"]]
        full_stream[tag] = {}
        for method in PAPER_METHODS:
            name = condition(method, retention)
            cell = evaluated["conditions"][name]
            values = [clip_mesr[(method, retention)][i] for i in members]
            mean_mesr = float(np.mean(values))
            stored = stored_mesr[(method, retention)]
            if not abs(mean_mesr - stored) <= MESR_TOLERANCE:
                raise AssertionError(f"{name}: MESR {mean_mesr!r} over the cohort differs from "
                                     f"the frozen run's {stored!r}; the retained sets differ")
            rows.append({"retention": retention, "method": method, "accuracy": cell["top1"],
                         "macro_f1": cell["macro_f1"], "mesr": mean_mesr,
                         "clips": len(members)})
            predictions[name] = [int(v) for v in cell["prediction"]]
            full_stream[tag][method] = frozen_accuracy(
                evaluation, condition(method, retention, scored=False), members)
            if method != "raw":
                key = f"{name}_vs_{condition('raw', retention)}"
                if key not in evaluated["comparisons"]:
                    raise KeyError(f"the evaluator reported no {key}; its pairing of "
                                   "conditions with the control has changed")
                value = evaluated["comparisons"][key]
                comparisons[key] = {k: value[k] for k in
                                    ("accuracy_delta", "accuracy_delta_95ci", "by_subject",
                                     "cluster_bootstrap_replicates",
                                     "mcnemar_exact_two_sided_p")}
        if check_reproduction:
            name = condition(REPRODUCED[0], retention, scored=False)
            again = [int(v) for v in evaluated["conditions"][name]["prediction"]]
            stored_prediction = [int(v) for v in evaluation["conditions"][name]["prediction"]]
            reproduction = {"condition": name, "top1": evaluated["conditions"][name]["top1"],
                            "stored_top1": evaluation["conditions"][name]["top1"],
                            "identical_predictions": again == stored_prediction}
            if not reproduction["identical_predictions"]:
                raise SystemExit(f"{name} did not reproduce the frozen run's predictions; the "
                                 "pipeline has changed since that run")
        for path in paths:
            path.unlink()
        write_report(complete=False)
        red = next(r for r in rows if r["retention"] == retention and r["method"] == "red")
        print(f"[r={retention}] {len(members)} clips: red accuracy {red['accuracy']:.4f} on "
              f"its scored events, {full_stream[tag]['red']:.4f} on its full stream "
              f"({elapsed()})", flush=True)

    if reproduction is None:
        raise SystemExit("the reproduction check did not run: the r = "
                         f"{REPRODUCED[1]} cohort is not the whole test set")
    report = write_report(complete=True)
    for path in written:
        if path.exists():
            path.unlink()
    if not any(args.frames_dir.iterdir()):
        args.frames_dir.rmdir()
    summary = report["summary"]
    print(f"\nred least accurate at r = {', '.join(summary['red_least_accurate_at']) or 'none'} "
          f"(alone at {', '.join(summary['red_least_accurate_alone_at']) or 'none'}); "
          f"mean within-retention rho {summary['mean_rho_mesr_accuracy']:+.4f}")
    print(f"wrote {args.out} ({report['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
