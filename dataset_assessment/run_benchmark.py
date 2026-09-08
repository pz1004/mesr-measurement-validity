"""Grid runner: methods x recordings x retentions, under the Task-5 protocol.

Every cell is written to `results/benchmark_<dataset>.json` as soon as it is produced, so a
killed run keeps its partial output.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .denoisors import (available_methods, is_null, is_oracle, native_retention,
                        score_events)
from .esr import mesr, mesr_curve
from .protocol import decompose_esr, protocol_row
from .readers import (iter_dnd21, iter_dvsd22, iter_dvsclean, iter_ed24, iter_emlb,
                      iter_pure_ba_noise)

RETENTIONS = np.round(np.arange(0.05, 1.001, 0.05), 3)
LOADERS = {
    "dnd21": iter_dnd21,
    "dvsclean": iter_dvsclean,
    "emlb": iter_emlb,
    "ed24": iter_ed24,
    "pure_ba": iter_pure_ba_noise,
    "dvsd22": iter_dvsd22,
}
CAPPED_LOADERS = {"emlb", "pure_ba", "dvsd22"}   # accept max_events at read time
OUT_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    parser.add_argument("--methods", nargs="+", default=None,
                        help="default: every method available_methods() finds. "
                             "'label_oracle' is opt-in and requires per-event labels.")
    parser.add_argument("--max-events", type=int, default=1_000_000)
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--scene-stride", type=int, default=1,
                        help="E-MLB only: keep every k-th scene, so a capped run still "
                             "covers all 4 ND levels and both lighting parts.")
    args = parser.parse_args()

    methods = args.methods or available_methods()
    out = args.out or OUT_DIR / f"benchmark_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = []

    loader = LOADERS[args.dataset]
    kwargs = {"max_events": args.max_events} if args.dataset in CAPPED_LOADERS else {}
    if args.dataset == "emlb":
        kwargs["scene_stride"] = args.scene_stride
    iterator = loader(**kwargs)
    for index, rec in enumerate(iterator):
        if args.max_recordings and index >= args.max_recordings:
            break
        events = rec.events[:args.max_events]
        rec = replace(rec, events=events,
                      labels=None if rec.labels is None else rec.labels[:len(events)])
        raw = mesr(rec.x, rec.y, rec.width, rec.height)
        entry = {"recording": rec.name, "synthetic": rec.synthetic,
                 "sensor": [rec.width, rec.height], "events": int(len(events)),
                 "raw_mesr": raw,
                 "raw_decomposition": decompose_esr(rec.x[:30_000], rec.y[:30_000],
                                                    rec.width, rec.height),
                 "methods": {}}
        for method in methods:
            try:
                scores = score_events(method, rec)
            except ValueError as error:          # e.g. oracle on an unlabeled recording
                entry["methods"][method] = {"method": method, "error": str(error)}
                continue
            curve = mesr_curve(scores, rec.x, rec.y, rec.width, rec.height, RETENTIONS)
            try:
                row = protocol_row(method, curve, raw, rec.width, rec.height)
            except ValueError:
                row = {"method": method, "error": "no evaluable retention point"}
            row["native_retention"] = native_retention(scores)
            row["is_oracle"] = is_oracle(method)
            row["is_null"] = is_null(method)
            row["curve"] = curve
            entry["methods"][method] = row
        records.append(entry)
        out.write_text(json.dumps({"dataset": args.dataset,
                                   "retentions": RETENTIONS.tolist(),
                                   "max_events": args.max_events,
                                   "records": records,
                                   "wall_seconds": time.perf_counter() - started},
                                  indent=2, default=float) + "\n")
        print(f"[{index}] {rec.name}: raw={raw:.4f} " +
              " ".join(f"{m}@{entry['methods'][m].get('r_star', float('nan')):.2f}="
                       f"{entry['methods'][m].get('mesr_star', float('nan')):.4f}"
                       for m in methods), flush=True)

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
