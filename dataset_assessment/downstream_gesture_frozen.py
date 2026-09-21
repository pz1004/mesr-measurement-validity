"""Criterion validity of MESR against a frozen classifier operating at the task ceiling.

`downstream_gesture` trains three small classifiers from scratch on this corpus and reaches
0.41-0.51 at full retention, where published pipelines reach ~0.95. A correlation computed
against accuracies that far below the ceiling is weak evidence in either direction: the
classifiers may simply be too weak to resolve what the filters did.

This module removes that objection instead of mitigating it. It scores the same conditions
with the *released* DVS Gesture classifier - the SEW 7B-Net of the SEW-ResNet paper, at the
authors' own maximum checkpoint - held frozen. Nothing is trained here, so there is no seed
variance, no schedule, and no capacity argument left to make. The checkpoint reproduces
0.968750 top-1 / 0.966816 macro-F1 on the 288 unfiltered test clips, which is the gate this
module refuses to proceed past (`--smoke`).

Three protocol differences from the trained probe, each forced by the frozen model and each
reported rather than absorbed:

  representation  The released model consumes 16 equal-*event-count* bins with separate
                  ON/OFF planes, not the 8 equal-*time* signed bins `voxelise` builds. The
                  released integrator is reused verbatim (`events_to_number_frames`) so the
                  frames are the ones the checkpoint was trained on.

  event budget    No 80,000-event cap. The cap exists in `downstream_gesture` so that
                  r = 0.4 still leaves one complete 30,000-event MESR slice; the frozen
                  model needs no such budget, and dropping it *improves* the coupling. Mean
                  clip length is ~587k events, so MESR reads ~89-97% of what the classifier
                  sees instead of the capped run's 62.5-93.75%. The two quantities are still
                  read off different event sets - that disclosure stands - but the gap
                  narrows from 31 points to about 8.

  retention grid  At r = 1.0 `retain_mask` keeps everything, so all methods produce the
                  identical stream. The trained probe carries that as eight identical rows;
                  here it is emitted once, as `clean__unfiltered`, so it cannot inflate n in
                  any pooled statistic. Correlations are computed within retention anyway.

The known confound, stated because it is not removable: equal-event-count binning couples
bin duration to retention. At r = 0.4 each of the 16 bins spans about 2.5x more wall-clock
time than the checkpoint was trained on. That hits every method equally and so does not
bias the *ranking* at a fixed retention - which is what the correlations test - but it does
mean accuracy falls with r partly from temporal rescaling and not only from lost content.
Comparisons across retention levels should not be read as content loss alone.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .denoisors import RANDOM_NULL, score_events
from .downstream_gesture import (
    GESTURE_ROOT,
    HEIGHT,
    WIDTH,
    _trials,
    detectable_rho,
    partial_spearman,
    within_retention_correlations,
)
from .esr import SLICE, mesr, retain_mask
from .readers import Recording, read_aedat31

PROJECT_ROOT = Path(__file__).resolve().parents[1]
#: The frozen classifier and its released integrator live in the sibling downstream study,
#: exactly like `native3d` in `denoisors`. Optional: absent checkout, this module declines.
DOWNSTREAM_ROOT = PROJECT_ROOT.parent
CHECKPOINT = (DOWNSTREAM_ROOT / "downstream/assets/snn7b/released"
              / "checkpoint_max_val_acc1.pth")
EVALUATOR = "downstream.evaluate_dvsgesture_variants"
#: The conda environment the released checkpoint was evaluated in. Overridable; the gate
#: below is what actually decides whether an interpreter reproduces the published cell.
SNN_PYTHON = Path("/home/pz1004/anaconda3/envs/snn/bin/python")

OUT = PROJECT_ROOT / "results/downstream_gesture_frozen.json"
FRAMES = PROJECT_ROOT / "results/downstream_gesture_frozen_frames"

BINS = 16
RETENTIONS = (0.4, 0.5, 0.6, 0.8, 1.0)
PAPER_METHODS = ("raw", RANDOM_NULL, "dwf", "evflow", "knoise", "red", "ts", "ynoise")
#: The published cell this module must reproduce before any filtered condition is trusted.
GATE_TOP1 = 0.968750
GATE_MACRO_F1 = 0.9668157414841424
GATE_TOLERANCE = 1e-6


def _condition_name(method: str, retention: float) -> str:
    """`prefix__method`, the scheme the released evaluator pairs and sorts on.

    The evaluator reads `prefix` as the corruption level and `unfiltered` as that level's
    reference row, then computes each method's paired subject-cluster delta against it. Here
    `prefix` is the retention and the reference is `raw` - the row that applies the same
    quota while discriminating nothing - so the deltas it produces are exactly the
    "did choosing *which* events to drop beat dropping the first ones?" contrast.

    At r = 1.0 every method returns the identical stream, so all of them map to the single
    name `clean__unfiltered`. Giving that condition a per-method name instead would write
    eight byte-identical frame arrays and hand eight tied points to any statistic computed
    over the whole grid, inflating n while carrying nothing about the filters.
    """

    if retention == 1.0:
        return "clean__unfiltered"
    return f"r{retention * 100:03.0f}__{'unfiltered' if method == 'raw' else method}"


def load_test_clips(limit_files: int | None = None
                    ) -> Tuple[List[np.ndarray], np.ndarray, List[dict], dict]:
    """Every labelled gesture of the official test split, whole interval, no event cap.

    Returns `(events, labels, clips, coverage)`. `clips` carries the subject per clip, which
    is the unit the evaluator's bootstrap resamples: clips from one subject are dependent,
    so a clip-level interval would be too narrow.
    """

    events: List[np.ndarray] = []
    labels: List[int] = []
    clips: List[dict] = []
    offered: Dict[int, int] = {}
    for name in _trials("trials_to_test.txt")[:limit_files]:
        path = GESTURE_ROOT / name
        label_path = GESTURE_ROOT / (path.stem + "_labels.csv")
        if not path.exists() or not label_path.exists():
            continue
        stream = read_aedat31(path)
        timestamps = stream[:, 0]
        table = pd.read_csv(label_path)
        subject = path.stem.split("_")[0]
        for order, (_, row) in enumerate(table.iterrows()):
            gesture_class = int(row["class"])
            offered[gesture_class] = offered.get(gesture_class, 0) + 1
            lo, hi = np.searchsorted(timestamps,
                                     (int(row["startTime_usec"]), int(row["endTime_usec"])))
            block = stream[lo:hi]
            # The released cache keeps every annotated interval; the only reason to drop one
            # here would be an empty slice, which no annotation produces.
            if len(block) < BINS:
                continue
            block = block.copy()
            block[:, 0] -= block[0, 0]
            events.append(np.ascontiguousarray(block))
            labels.append(gesture_class - 1)
            clips.append({"clip_id": f"{path.stem}:{order:02d}", "subject": subject,
                          "trial": name, "class_index": gesture_class - 1,
                          "events": int(len(block))})
    coverage = {
        "gestures_offered": int(sum(offered.values())),
        "gestures_kept": len(events),
        "mean_events": float(np.mean([len(e) for e in events])) if events else float("nan"),
        "median_events": float(np.median([len(e) for e in events])) if events else float("nan"),
        "capped_at": None,
    }
    return events, np.asarray(labels, dtype=np.int64), clips, coverage


def frames_for(events: np.ndarray, keep: np.ndarray, integrator) -> np.ndarray:
    """`(BINS, 2, H, W)` uint16 ON/OFF count frames of the kept events.

    The released integrator slices on `[start_us, end_us)` and then splits *by count*, so
    the whole retained stream is handed to it and the interval is its own extent.
    """

    kept = events[keep]
    if not len(kept):
        return np.zeros((BINS, 2, HEIGHT, WIDTH), dtype=np.uint16)
    return integrator(kept, int(kept[0, 0]), int(kept[-1, 0]) + 1,
                      bins=BINS, width=WIDTH, height=HEIGHT, dtype=np.uint16)


def scored_share(clip_lengths: Sequence[int], retentions: Sequence[float] = RETENTIONS,
                 slice_size: int = SLICE) -> List[dict]:
    """How much of each classified event set MESR actually reads, uncapped.

    The same quantity `downstream_gesture.scored_share` reports for the 80,000-event budget,
    recomputed against the real clip lengths so the improvement is a measurement rather than
    a claim. `retain_mask` works per block, so the kept count is taken from it directly.
    """

    out: List[dict] = []
    for retention in retentions:
        classified, scored = 0, 0
        for length in clip_lengths:
            kept = int(retain_mask(np.zeros(length), float(retention)).sum())
            classified += kept
            scored += (kept // slice_size) * slice_size
        out.append({
            "retention": float(retention),
            "classified": classified,
            "scored_by_mesr": scored,
            "scored_share": (scored / classified) if classified else float("nan"),
        })
    return out


def build_conditions(events, labels, clips, methods, retentions, frames_dir: Path,
                     integrator) -> Tuple[dict, List[dict]]:
    """Write one frame array per condition and return the evaluator manifest plus MESR rows."""

    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_path = frames_dir / "labels.npy"
    np.save(labels_path, labels)

    conditions: Dict[str, dict] = {}
    rows: List[dict] = []
    for method in methods:
        started = time.perf_counter()
        scores = [score_events(method, Recording(e, None, WIDTH, HEIGHT, "", False))
                  for e in events]
        print(f"[{method}] scored {len(scores)} clips in "
              f"{time.perf_counter() - started:.0f}s", flush=True)

        for retention in retentions:
            name = _condition_name(method, retention)
            # Every method collapses to the same stream at r = 1.0, because `retain_mask`
            # keeps everything regardless of score. Emit that condition once rather than
            # once per method: eight identical rows would inflate n in any pooled statistic
            # while carrying no information about the filters.
            if name in conditions:
                continue
            keeps = [retain_mask(s, retention) for s in scores]
            path = frames_dir / f"{name}.npy"
            stack = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.uint16,
                shape=(len(events), BINS, 2, HEIGHT, WIDTH))
            for index, (clip, keep) in enumerate(zip(events, keeps)):
                stack[index] = frames_for(clip, keep, integrator)
            stack.flush()
            del stack

            values = [mesr(e[k][:, 1], e[k][:, 2], WIDTH, HEIGHT)
                      for e, k in zip(events, keeps)]
            values = [v for v in values if np.isfinite(v)]
            conditions[name] = {"frames": str(path.resolve()), "method": method,
                                "retention": float(retention)}
            rows.append({
                "name": name, "method": method, "retention": float(retention),
                "mesr": float(np.mean(values)) if values else float("nan"),
                "evaluable_test_samples": len(values),
                "actual_kept_fraction": float(np.mean([k.mean() for k in keeps])),
            })
            print(f"  r={retention:.2f}  mesr={rows[-1]['mesr']:.4f}  "
                  f"kept={rows[-1]['actual_kept_fraction']:.3f}  -> {name}", flush=True)

    manifest = {
        "format_version": 1,
        "dataset": "IBM DVS Gesture raw AEDAT 3.1, official test split, uncapped",
        "labels": str(labels_path.resolve()),
        "clips": clips,
        "shape": [len(events), BINS, 2, HEIGHT, WIDTH],
        "conditions": conditions,
    }
    return manifest, rows


def run_evaluator(manifest_path: Path, output_path: Path, python: Path,
                  checkpoint: Path) -> dict:
    """Invoke the released evaluator unchanged, as a module, from the sibling checkout."""

    command = [str(python), "-m", EVALUATOR,
               "--manifest", str(manifest_path),
               "--checkpoint", str(checkpoint),
               "--output", str(output_path)]
    print(f"$ {' '.join(command)}", flush=True)
    # Inherit the environment: the released evaluator runs under a conda interpreter whose
    # CUDA libraries are found through variables a minimal env would drop.
    environment = dict(os.environ, PYTHONPATH=str(DOWNSTREAM_ROOT))
    completed = subprocess.run(command, cwd=str(DOWNSTREAM_ROOT), text=True,
                               env=environment, capture_output=True)
    if completed.returncode != 0:
        raise SystemExit(f"evaluator failed ({completed.returncode}):\n"
                         f"{completed.stdout[-2000:]}\n{completed.stderr[-4000:]}")
    print(completed.stdout.strip()[-1000:], flush=True)
    return json.loads(output_path.read_text())


def check_gate(evaluated: dict) -> dict:
    """Refuse to report filtered conditions unless the unfiltered cell is the published one."""

    cell = evaluated["conditions"].get("clean__unfiltered")
    if cell is None:
        raise SystemExit("no `clean__unfiltered` condition; the gate cannot be checked")
    top1, macro_f1 = cell["top1"], cell["macro_f1"]
    ok = (abs(top1 - GATE_TOP1) < GATE_TOLERANCE
          and abs(macro_f1 - GATE_MACRO_F1) < GATE_TOLERANCE)
    result = {"top1": top1, "macro_f1": macro_f1, "expected_top1": GATE_TOP1,
              "expected_macro_f1": GATE_MACRO_F1, "reproduced": bool(ok),
              "samples": cell["samples"]}
    status = "PASS" if ok else "FAIL"
    print(f"\n[gate {status}] clean__unfiltered  top1={top1:.6f} (expect {GATE_TOP1:.6f})  "
          f"macro_f1={macro_f1:.6f} (expect {GATE_MACRO_F1:.6f})  n={cell['samples']}",
          flush=True)
    return result


def dedupe_rows(rows: Sequence[dict]) -> List[dict]:
    """One row per condition the classifier actually saw.

    Retention 1.0 is a single condition wearing eight method labels: `retain_mask` keeps
    everything, so the streams - and therefore the MESR and the accuracy - are identical.
    Carrying all eight into a statistic computed over the whole grid adds eight tied points
    that no filter produced.
    """

    seen: Dict[str, dict] = {}
    for row in rows:
        name = _condition_name(row["method"], row["retention"])
        if name not in seen:
            seen[name] = dict(row, name=name,
                              method=row["method"] if row["retention"] < 1.0 else "all")
    return list(seen.values())


def summarise(rows: Sequence[dict]) -> dict:
    """The three statistics the criterion question turns on, over deduplicated rows."""

    finite = [r for r in rows if np.isfinite(r["mesr"]) and np.isfinite(r["accuracy"])]
    return {
        "within_retention": within_retention_correlations(finite),
        "partial_spearman": partial_spearman([r["mesr"] for r in finite],
                                             [r["accuracy"] for r in finite],
                                             [r["retention"] for r in finite]),
        "detectable_rho": detectable_rho(len(finite)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-artifact", action="store_true",
                        help=("Recompute the statistics from --out without rebuilding "
                              "frames or re-running the classifier. Only ever changes how "
                              "a summary is computed, never a measured number."))
    parser.add_argument("--limit-test-files", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=list(PAPER_METHODS))
    parser.add_argument("--smoke", action="store_true",
                        help="Build only the unfiltered condition and check the gate.")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--frames-dir", type=Path, default=FRAMES)
    parser.add_argument("--python", type=Path, default=SNN_PYTHON)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()

    if args.from_artifact:
        report = json.loads(args.out.read_text())
        before = len(report["conditions"])
        report["conditions"] = dedupe_rows(report["conditions"])
        report.update(summarise(report["conditions"]))
        report["protocol"]["deduplicated_conditions"] = {
            "before": before, "after": len(report["conditions"]),
            "why": ("retention 1.0 is one condition under eight method labels; the extra "
                    "rows are exact ties and were counted once"),
        }
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"recomputed from {args.out}: {before} -> {len(report['conditions'])} rows")
        within, partial = report["within_retention"], report["partial_spearman"]
        print(f"within-retention mean rho = {within.get('mean_rho', float('nan')):.4f}  "
              f"Fisher p = {within.get('fisher_p', float('nan')):.4g}")
        print(f"partial Spearman = {partial.get('rho', float('nan')):.4f}  "
              f"p = {partial.get('p', float('nan')):.4g}  n = {partial.get('n')}")
        return

    if not args.checkpoint.is_file():
        raise SystemExit(f"released checkpoint not found at {args.checkpoint}")

    sys.path.insert(0, str(DOWNSTREAM_ROOT))
    from downstream.dvsgesture import events_to_number_frames

    started = time.perf_counter()
    methods = ["raw"] if args.smoke else args.methods
    retentions = (1.0,) if args.smoke else RETENTIONS

    print("loading test clips (uncapped) ...", flush=True)
    events, labels, clips, coverage = load_test_clips(args.limit_test_files)
    print(f"  {len(events)} clips, mean {coverage['mean_events']:,.0f} events "
          f"({time.perf_counter() - started:.0f}s)", flush=True)

    manifest, rows = build_conditions(events, labels, clips, methods, retentions,
                                      args.frames_dir, events_to_number_frames)
    manifest_path = args.frames_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    evaluated = run_evaluator(manifest_path, args.frames_dir / "evaluation.json",
                              args.python, args.checkpoint)
    gate = check_gate(evaluated)

    for row in rows:
        cell = evaluated["conditions"].get(row["name"], {})
        row["accuracy"] = cell.get("top1", float("nan"))
        row["macro_f1"] = cell.get("macro_f1", float("nan"))

    if args.smoke:
        args.out.with_suffix(".smoke.json").write_text(
            json.dumps({"gate": gate, "rows": rows, "coverage": coverage},
                       indent=2, sort_keys=True) + "\n")
        raise SystemExit(0 if gate["reproduced"] else 1)

    if not gate["reproduced"]:
        raise SystemExit("gate failed; filtered conditions are not reported")

    rows = dedupe_rows(rows)
    report = {
        "protocol": {
            "classifier": "released SEW 7B-Net, frozen, authors' maximum checkpoint",
            "checkpoint": str(args.checkpoint.resolve()),
            "trained_here": False,
            "seeds": None,
            "representation": f"{BINS} equal-event-count bins, separate ON/OFF planes",
            "event_cap": None,
            "retentions": list(retentions),
            "methods": methods,
            "test_samples": len(events),
            "slice_size": SLICE,
        },
        "gate": gate,
        "coverage": coverage,
        "scored_share": scored_share([c["events"] for c in clips], retentions),
        "conditions": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    report.update(summarise(rows))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    within = report["within_retention"]
    print(f"\nwithin-retention mean rho = {within.get('mean_rho', float('nan')):.4f}  "
          f"Fisher p = {within.get('fisher_p', float('nan')):.4g}")
    print(f"partial Spearman (retention controlled) = "
          f"{report['partial_spearman'].get('rho', float('nan')):.4f}  "
          f"p = {report['partial_spearman'].get('p', float('nan')):.4g}")
    print(f"wrote {args.out}  ({report['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
