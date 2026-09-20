"""EDformer's released model and inference recipe, run on the harness's own E-MLB inputs.

Table IV used to take EDformer's row from an earlier, uncapped run of the released evaluation
script, so it matched the classical rows on recordings but not on the 10^6-event cap, and it
could enter no matched contrast. This module scores EDformer on exactly the capped streams
every other row is scored on.

The recipe is the released one (`EDformer/eval_mesr.py`), with `model.py` imported unmodified:

* blocks of `SEQ_LEN = 4096` events, one block per forward pass (batching shifts logits by
  ~5e-6 and flips about one event per million across the sharp threshold);
* timestamps min-max normalised over the whole input stream before blocking -- here the
  capped stream, which is the input every other method receives;
* raw pixel x and y, polarity in {0, 1};
* an event is kept iff ``sigmoid(f) < 0.005``, the operating point hard-coded in the release.

Events after the last complete block are never scored by the released recipe and never
appear in its output, so they are rejected here. They get the score `TAIL_SCORE`, outside
the sigmoid's range, so a retention sweep of the continuous score ranks them last.

Needs torch and CUDA, so it runs in the `edformer` environment; the rest of the harness stays
torch-free and reads the cached scores through `denoisors`. Two modes:

    # per-recording sigmoid scores on the capped inputs, cached for denoisors
    python -m dataset_assessment.edformer --cap 1000000
    # the released protocol, uncapped: per-cell MESR comparable with the published table
    python -m dataset_assessment.edformer --cap 0 --out results/edformer_emlb_uncapped.json

`--only NAME` restricts either mode to recordings whose name contains NAME.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .esr import SLICE, mesr
from .readers import iter_emlb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDFORMER_ROOT = PROJECT_ROOT / "EDformer"
CACHE = PROJECT_ROOT / "cache" / "edformer"
RESULTS = PROJECT_ROOT / "results"
SEQ_LEN = 4096
THRESHOLD = 0.005
SEED = 230086                     # eval_mesr.setup_seed's default
TAIL_SCORE = 2.0


def cache_dir(cap: int) -> Path:
    return CACHE / f"emlb_cap{cap}"


def cache_path(name: str, cap: int) -> Path:
    return cache_dir(cap) / (name.replace("/", "__") + ".npy")


def keep_mask(scores: np.ndarray) -> np.ndarray:
    """The released decision: keep iff sigmoid < 0.005. Tail events (TAIL_SCORE) are dropped."""

    return scores < THRESHOLD


def _normalise(column: np.ndarray) -> np.ndarray:
    return (column - np.min(column)) / (np.max(column) - np.min(column))


def load_model():
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    if str(EDFORMER_ROOT) not in sys.path:
        sys.path.insert(0, str(EDFORMER_ROOT))
    from model import EDformer

    model = EDformer().cuda()
    model.load_state_dict(torch.load(str(EDFORMER_ROOT / "pretrained_model.pth"),
                                     map_location="cuda:0"))
    model.eval()
    return model


def score_events(model, events: np.ndarray) -> np.ndarray:
    """Per-event sigmoid(f) under the released recipe; TAIL_SCORE past the last full block."""

    import torch

    prepared = np.column_stack([_normalise(events[:, 0].astype(np.float64)),
                                events[:, 1].astype(np.float64),
                                events[:, 2].astype(np.float64),
                                events[:, 3].astype(np.float64)])
    scores = np.full(len(events), TAIL_SCORE, dtype=np.float32)
    n_blocks = len(events) // SEQ_LEN
    for b in range(n_blocks):
        block = prepared[b * SEQ_LEN:(b + 1) * SEQ_LEN][None]
        batch = torch.from_numpy(np.ascontiguousarray(block)).to(dtype=torch.float32).cuda()
        with torch.no_grad():
            logits = model(batch)
        scores[b * SEQ_LEN:(b + 1) * SEQ_LEN] = (
            torch.sigmoid(logits).squeeze(-1).squeeze(0).cpu().numpy())
    return scores


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run_capped(cap: int, only: Optional[str], out: Path) -> Dict:
    """Cache every recording's scores and write a manifest of what was cached."""

    model = load_model()
    cache_dir(cap).mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    started = time.perf_counter()
    for index, rec in enumerate(iter_emlb(max_events=cap)):
        if only and only not in rec.name:
            continue
        events = rec.events[:cap]
        scores = score_events(model, events)
        np.save(cache_path(rec.name, cap), scores)
        kept = keep_mask(scores)
        rows.append({"recording": rec.name, "events": int(len(events)),
                     "scored": int((scores != TAIL_SCORE).sum()),
                     "kept": int(kept.sum()), "sha256": _sha(scores)})
        payload = {"what": "EDformer sigmoid scores on the capped E-MLB inputs",
                   "cap": cap, "seq_len": SEQ_LEN, "threshold": THRESHOLD, "seed": SEED,
                   "tail_score": TAIL_SCORE, "cache": str(cache_dir(cap).relative_to(PROJECT_ROOT)),
                   "records": rows, "wall_seconds": time.perf_counter() - started}
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[{index}] {rec.name}: {len(events)} events, kept {int(kept.sum())} "
              f"({time.perf_counter() - started:.0f}s)", flush=True)
    return payload


def run_uncapped(only: Optional[str], out: Path) -> Dict:
    """The released protocol on whole recordings: MESR of the unfiltered and kept streams."""

    model = load_model()
    rows: List[Dict] = []
    started = time.perf_counter()
    for index, rec in enumerate(iter_emlb(max_events=None)):
        if only and only not in rec.name:
            continue
        if len(rec.events) < SLICE:          # the released script skips these
            continue
        scores = score_events(model, rec.events)
        kept = keep_mask(scores)
        considered = int((scores != TAIL_SCORE).sum())
        rows.append({"recording": rec.name, "events": int(len(rec.events)),
                     "considered": considered, "kept": int(kept.sum()),
                     "retention": float(kept.sum() / considered) if considered else None,
                     "raw_mesr": float(mesr(rec.x, rec.y, rec.width, rec.height)),
                     "edformer_mesr": float(mesr(rec.x[kept], rec.y[kept],
                                                 rec.width, rec.height))})
        out.write_text(json.dumps({"what": "EDformer, released protocol, uncapped E-MLB",
                                   "seq_len": SEQ_LEN, "threshold": THRESHOLD, "seed": SEED,
                                   "records": rows,
                                   "wall_seconds": time.perf_counter() - started},
                                  indent=2) + "\n")
        print(f"[{index}] {rec.name}: kept {int(kept.sum())}/{considered} "
              f"mesr {rows[-1]['edformer_mesr']:.4f} "
              f"({time.perf_counter() - started:.0f}s)", flush=True)
    return {"records": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, default=1_000_000, help="0 = uncapped")
    parser.add_argument("--only", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.cap > 0:
        run_capped(args.cap, args.only,
                   args.out or RESULTS / "edformer_emlb_manifest.json")
    else:
        run_uncapped(args.only, args.out or RESULTS / "edformer_emlb_uncapped.json")


if __name__ == "__main__":
    main()
