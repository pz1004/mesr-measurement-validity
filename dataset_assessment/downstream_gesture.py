"""Does MESR predict downstream task utility? (paper section 9)

The question the whole benchmark rests on: when a denoiser raises MESR, does anything
downstream get better? This runs one fixed classification pipeline over
methods x retentions, holding *everything* constant except the filter, and correlates the
resulting accuracy against the MESR of the same filtered streams.

Design constraints, so the correlation means what it says:
  * identical seed set, architecture, epochs, batch size and optimiser in every condition;
  * identical event budget per sample before filtering;
  * the official DVS Gesture train/test split, never re-drawn;
  * `r = 1.0` is the unfiltered reference and is shared by every method by construction;
  * every condition is trained under EVERY seed in `SEEDS`, so a reported difference
    between conditions can be compared against the spread across seeds.

WHY THE EVENT BUDGET IS 80,000 AND THE GRID STARTS AT r = 0.4
-------------------------------------------------------------
MESR needs one complete 30,000-event slice, so evaluating a sample at retention `r` needs
roughly `30,000 / r` events. Requiring a large budget therefore silently DELETES the
gestures that do not emit that many - and which gestures those are is not random.

An earlier version of this experiment took the first 160,000 events of each gesture's
first 4 s and skipped any gesture that fell short. That kept 771/1176 training and
205/288 test gestures, and the loss was strongly class-dependent: hand clapping (class 1)
kept 21% of its test gestures while the two arm-circle classes kept 100%. The measured
survival rates, on the full gesture rather than its first 4 s, are:

    budget    min r    train kept    test kept    worst class
    160,000    0.20        82.8%        89.9%            33%
    120,000    0.30        89.2%        93.8%            51%
     80,000    0.40        95.7%        99.0%       77% / 92%
     60,000    0.50        98.6%       100.0%            95%

80,000 is the chosen point: the test split - which carries every accuracy and every MESR
reported here - keeps 285 of 288 gestures with 22-24 in every class, so the task is the
official 11-class problem rather than a survivorship-filtered version of it. The price is
that r = 0.2 cannot be evaluated at all, and it is not recoverable: no budget evaluates
MESR at r = 0.2 without re-imposing the same bias. `coverage` in the output artifact
records exactly what was kept, per split and per class, so the cost is auditable and does
not depend on this docstring being read.

Sensor caveat that must survive into the paper: DVS128 has K = 16,384 pixels, fewer than
the 30,000-event ESR slice. This is the one corpus in this project where the occupied-pixel
ceiling in `ln` actually binds (see `esr` module docstring), so absolute MESR here is not
comparable with the 346x260 and 1280x720 corpora. The correlation is computed within this
dataset only, which is unaffected.

Why not DVSD22: its physical reference values were measured by hand. Its release paper
(Micev et al., doi 10.5194/amt-17-335-2024) says so directly - droplets were "manually
selected" and read off with jAER's Speedometer plugin. There is no automatic
estimator to place a denoiser in front of. Two were tried and both fail on the *unfiltered*
stream, which makes them useless as a downstream probe - see `plan/dataset_assessment_notes.md`,
finding F-DROPLETS-TASK.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from .denoisors import RANDOM_NULL, available_methods, score_events
from .esr import mesr, retain_mask
from .readers import Recording, read_aedat31

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GESTURE_ROOT = PROJECT_ROOT / "DVSGesture/DvsGesture"
OUT = PROJECT_ROOT / "results/downstream_gesture.json"

WIDTH = HEIGHT = 128
#: The whole labelled gesture is eligible. An earlier version truncated to the first 4 s,
#: which cost 29% of the test corpus and did so class-dependently - see the module
#: docstring. The event budget below, not a wall-clock window, is what equalises samples.
MAX_EVENTS = 80_000
#: Smallest retention that still yields a complete 30,000-event ESR slice at that budget.
#: `retain_mask` works per 10,240-event block, so 80,000 events at r = 0.4 keeps
#: 7 * 4096 + round(8320 * 0.4) = 32,000 >= 30,000. At r = 0.3 it would keep 24,000 and
#: every cell would be silently unevaluable.
MIN_RETENTION = 0.4
TIME_BINS = 8
N_CLASSES = 11
RETENTIONS = (0.4, 0.5, 0.6, 0.8, 1.0)
#: Every condition is trained under all three. One seed cannot distinguish "this filter is
#: better" from "this run landed better", which is exactly the distinction section 8 needs.
SEEDS = (20260726, 20260727, 20260728)
#: The method set the paper's tables and correlations use: six classical filters plus the
#: two nulls. `available_methods()` additionally returns the sibling project's `native3d`
#: rows whenever that checkout is present next to this one, and those rows are kept in the
#: artifact but excluded from the headline statistics - exactly as the E-MLB benchmark
#: keeps them in a separate file. Mixing a method under evaluation into the statistics of a
#: paper that evaluates the metric would change what those statistics mean.
PAPER_METHODS = ("raw", "random_null", "dwf", "evflow", "knoise", "red", "ts", "ynoise")
EPOCHS = 20
BATCH = 32
LR = 1e-3


def _trials(name: str) -> List[str]:
    return [line.strip() for line in (GESTURE_ROOT / name).read_text().splitlines()
            if line.strip()]


def load_samples(trial_files: List[str], limit_files: int | None = None
                 ) -> Tuple[List[np.ndarray], np.ndarray, dict]:
    """One sample per labelled gesture: its first MAX_EVENTS events, whole gesture eligible.

    Returns `(events, labels, coverage)`. `coverage` records how much of the official split
    survived the event-budget requirement, overall and per class, so the selection this
    experiment applies is reported as data rather than described in prose.
    """

    events, labels = [], []
    offered: dict = {}
    kept_by_class: dict = {}
    for name in trial_files[:limit_files]:
        path = GESTURE_ROOT / name
        label_path = GESTURE_ROOT / (path.stem + "_labels.csv")
        if not path.exists() or not label_path.exists():
            continue
        stream = read_aedat31(path)
        table = pd.read_csv(label_path)
        timestamps = stream[:, 0]
        for _, row in table.iterrows():
            start = int(row["startTime_usec"])
            stop = int(row["endTime_usec"])
            gesture_class = int(row["class"])
            offered[gesture_class] = offered.get(gesture_class, 0) + 1
            lo, hi = np.searchsorted(timestamps, (start, stop))
            block = stream[lo:hi][:MAX_EVENTS]
            # Require the full budget so every retention is evaluable for every sample;
            # a sample measurable at r = 1.0 but not at MIN_RETENTION would silently change
            # the sample set between conditions. Applied identically to both splits, and
            # the resulting loss is reported in `coverage`.
            if len(block) < MAX_EVENTS:
                continue
            kept_by_class[gesture_class] = kept_by_class.get(gesture_class, 0) + 1
            block = block.copy()
            block[:, 0] -= block[0, 0]
            events.append(np.ascontiguousarray(block))
            labels.append(gesture_class - 1)
    coverage = {
        "gestures_offered": int(sum(offered.values())),
        "gestures_kept": int(len(events)),
        "kept_fraction": (len(events) / sum(offered.values())) if offered else float("nan"),
        "per_class": {str(c): {"offered": offered[c], "kept": kept_by_class.get(c, 0),
                               "kept_fraction": kept_by_class.get(c, 0) / offered[c]}
                      for c in sorted(offered)},
    }
    return events, np.asarray(labels, dtype=np.int64), coverage


def voxelise(events: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """(TIME_BINS, H, W) float32 polarity-signed event count grid of the kept events."""

    kept = events[keep]
    grid = np.zeros((TIME_BINS, HEIGHT, WIDTH), dtype=np.float32)
    if not len(kept):
        return grid
    span = max(int(kept[-1, 0] - kept[0, 0]), 1)
    bins = np.minimum(((kept[:, 0] - kept[0, 0]) * TIME_BINS) // span, TIME_BINS - 1)
    flat = (bins * HEIGHT + kept[:, 2]) * WIDTH + kept[:, 1]
    signed = np.where(kept[:, 3] > 0, 1.0, -1.0).astype(np.float32)
    np.add.at(grid.reshape(-1), flat, signed)
    return grid


ARCHS = ("cnn2d", "cnn3d", "mlp")
DEFAULT_ARCH = "cnn2d"


def build_model(arch: str = DEFAULT_ARCH):
    """One classifier per inductive bias, over the identical voxel grid.

    S8 asks whether MESR predicts downstream accuracy, and a null answer is only worth
    reporting if it is not a property of one architecture. The three here differ in what
    they assume about the input, not in their training schedule:

      cnn2d  the shipped model. The 8 time bins enter as 2D convolution *channels*, so
             temporal order is available to the first layer but never modelled as an axis.
      cnn3d  the same depth over Conv3d, with time as a real axis. Tests whether the null
             survives a model that can represent motion.
      mlp    average-pool 4x spatially, then dense layers. No locality and no translation
             equivariance in the classifier. Tests whether the null survives a model with
             no spatial prior at all. It is expected to score lower; that is the point.

    Every architecture sees the same tensors, seeds, epochs, batch size and optimiser, so
    accuracy differences between them are architecture and nothing else.
    """

    import torch.nn as nn

    if arch == "cnn2d":
        return nn.Sequential(
            nn.Conv2d(TIME_BINS, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, N_CLASSES))

    if arch == "cnn3d":
        return nn.Sequential(
            nn.Unflatten(1, (1, TIME_BINS)),
            nn.Conv3d(1, 32, 3, stride=(1, 2, 2), padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(128, N_CLASSES))

    if arch == "mlp":
        pooled = (WIDTH // 4) * (HEIGHT // 4) * TIME_BINS
        return nn.Sequential(
            nn.AvgPool2d(4), nn.Flatten(),
            nn.Linear(pooled, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, N_CLASSES))

    raise ValueError(f"unknown architecture {arch!r}; expected one of {ARCHS}")


def train_and_score(train_x: np.ndarray, train_y: np.ndarray,
                    test_x: np.ndarray, test_y: np.ndarray, seed: int,
                    arch: str = DEFAULT_ARCH) -> float:
    """Test accuracy for one seed. Every condition sees the identical schedule."""

    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(arch).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    xs = torch.from_numpy(train_x)
    ys = torch.from_numpy(train_y)
    generator = torch.Generator().manual_seed(seed)

    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(len(xs), generator=generator)
        for start in range(0, len(order), BATCH):
            batch = order[start:start + BATCH]
            optimiser.zero_grad()
            loss = loss_fn(model(xs[batch].to(device)), ys[batch].to(device))
            loss.backward()
            optimiser.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(test_x), 128):
            block = torch.from_numpy(test_x[start:start + 128]).to(device)
            correct += int((model(block).argmax(1).cpu().numpy()
                            == test_y[start:start + 128]).sum())
    return correct / len(test_y)


def partial_spearman(x: Sequence[float], y: Sequence[float],
                     control: Sequence[float]) -> dict:
    """Spearman correlation of `x` and `y` with `control` partialled out.

    Section 8's question is "does MESR track accuracy?", and retention confounds it: it
    moves accuracy strongly and MESR barely, so a raw correlation over all conditions is
    dominated by retention. Splitting the grid by retention answers the question cleanly
    but leaves only 8 points per level, where Spearman needs |rho| ~ 0.74 to reach
    p < 0.05 - a null there is uninformative. This partials retention out of the ranks and
    keeps all conditions, which is the same question at usable power.
    """

    from scipy.stats import rankdata, t as t_dist

    a, b, c = (rankdata(np.asarray(v, dtype=float)) for v in (x, y, control))
    n = len(a)
    if n < 4:
        return {"rho": float("nan"), "p": float("nan"), "n": int(n)}
    r_ab, r_ac, r_bc = (np.corrcoef(u, v)[0, 1] for u, v in ((a, b), (a, c), (b, c)))
    if not all(np.isfinite(v) for v in (r_ab, r_ac, r_bc)):
        return {"rho": float("nan"), "p": float("nan"), "n": int(n)}
    # Tolerance, not `== 0`: when the control explains either variable almost perfectly the
    # denominator is a rounding residue (~1e-16) and the ratio of two such residues is an
    # arbitrary finite number. Returning it would look like a measurement.
    denominator = np.sqrt(max(1 - r_ac ** 2, 0.0) * max(1 - r_bc ** 2, 0.0))
    if denominator < 1e-12:
        return {"rho": float("nan"), "p": float("nan"), "n": int(n),
                "undefined_because": "the control variable is collinear with x or y"}
    rho = float((r_ab - r_ac * r_bc) / denominator)
    statistic = rho * np.sqrt((n - 3) / max(1 - rho ** 2, np.spacing(1)))
    return {"rho": rho, "p": float(2 * t_dist.sf(abs(statistic), n - 3)), "n": int(n),
            "control": "retention"}


def within_retention_correlations(rows: Sequence[dict]) -> dict:
    """Per-retention MESR-accuracy correlations, plus a Fisher combination of them.

    Each level has one point per method, so these are the "does the metric pick the better
    filter at a fixed operating point?" tests. Individually underpowered; the Fisher
    combination is what should be quoted alongside them.

    `per_seed` repeats the whole calculation against each seed's own accuracies instead of
    the across-seed mean. If a combined result is real it should be visible, if attenuated,
    in the individual seeds; if it appears only after averaging, it is a property of the
    averaging. Reporting both is what lets a reader tell those apart.
    """

    from scipy.stats import combine_pvalues, spearmanr

    levels = sorted({r["retention"] for r in rows if r["retention"] < 1.0})

    def combine(accuracy_of) -> dict:
        per_level = {}
        for level in levels:
            group = [r for r in rows if r["retention"] == level]
            if len(group) < 3:
                continue
            values = [accuracy_of(g) for g in group]
            if any(v is None for v in values):
                return {}
            result = spearmanr([g["mesr"] for g in group], values)
            per_level[f"{level:g}"] = {"rho": float(result.statistic),
                                       "p": float(result.pvalue), "n": len(group)}
        entries = list(per_level.values())
        if not entries:
            return {}
        return {"per_retention": per_level,
                "mean_rho": float(np.mean([v["rho"] for v in entries])),
                "fisher_p": float(combine_pvalues([v["p"] for v in entries],
                                                  method="fisher").pvalue)}

    # Always publish the summary keys, even when nothing was computable: callers print them
    # unguarded, and a KeyError three steps downstream is worse than an explicit NaN.
    out = {"per_retention": {}, "mean_rho": float("nan"), "fisher_p": float("nan")}
    out.update(combine(lambda g: g["accuracy"]))
    seeds = sorted({k for r in rows for k in r.get("accuracy_per_seed", {})})
    per_seed = {}
    for seed in seeds:
        result = combine(lambda g, s=seed: g.get("accuracy_per_seed", {}).get(s))
        if result:
            per_seed[seed] = {"mean_rho": result["mean_rho"],
                              "fisher_p": result["fisher_p"]}
    out["per_seed"] = per_seed
    out["min_detectable_rho_note"] = ("each level has one point per method; Spearman needs "
                                      "|rho| ~ 0.74 at n = 8 to reach p < 0.05")
    return out


def detectable_rho(n: int, alpha: float = 0.05) -> dict:
    """The smallest |rho| this many points can call significant, and the power behind it.

    A failure to reject is only informative next to what the test could have detected. With
    40 matched conditions the paper cannot resolve a weak association at all, and saying so
    is stronger than leaving the reader to assume the null was well powered.
    """

    if n < 5:
        return {"n": int(n), "min_detectable_rho": float("nan")}
    from scipy.stats import norm
    se = 1.0 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2)
    power = {f"rho_{target:.1f}".replace(".", ""): float(
        norm.cdf((abs(np.arctanh(target)) - z_crit * se) / se))
        for target in (0.3, 0.5)}
    return {"n": int(n), "alpha": alpha,
            "min_detectable_rho": float(np.tanh(z_crit * se)),
            "power": power,
            "note": ("a failure to reject bounds the effect only as far as this; the paper "
                     "reports it so the negative result is not read as a well-powered null")}


def summarise_conditions(conditions: Sequence[dict]) -> dict:
    """Every correlation section 8 reports, over whichever conditions are handed in.

    Kept separate from `main` so the same summary can be computed over the paper's method
    set and over the full run without re-training anything, and so `--from-artifact` can
    rebuild it from a finished artifact.
    """

    from scipy.stats import spearmanr

    usable = [c for c in conditions if np.isfinite(c["mesr"])]
    if len(usable) < 4:
        return {"error": f"only {len(usable)} evaluable conditions; nothing to correlate"}

    rho = spearmanr([c["mesr"] for c in usable], [c["accuracy"] for c in usable])
    # Excluding r = 1.0, where every method is the unfiltered stream and the points are
    # duplicates that would inflate any correlation.
    filtered = [c for c in usable if c["retention"] < 1.0]
    rho_filtered = spearmanr([c["mesr"] for c in filtered],
                             [c["accuracy"] for c in filtered])
    retention_accuracy = spearmanr([c["retention"] for c in usable],
                                   [c["accuracy"] for c in usable])
    retention_mesr = spearmanr([c["retention"] for c in usable],
                               [c["mesr"] for c in usable])

    per_seed = {}
    for key in sorted({k for c in usable for k in c.get("accuracy_per_seed", {})}):
        rows = [c for c in usable if key in c.get("accuracy_per_seed", {})]
        result = spearmanr([c["mesr"] for c in rows],
                           [c["accuracy_per_seed"][key] for c in rows])
        per_seed[key] = {"rho": float(result.statistic), "p": float(result.pvalue),
                         "n": len(rows)}

    spreads = [c["accuracy_sd"] for c in usable if "accuracy_sd" in c]
    return {
        "methods": sorted({c["method"] for c in usable}),
        "n_conditions": len(usable),
        "mesr_vs_accuracy": {
            "all": {"rho": float(rho.statistic), "p": float(rho.pvalue), "n": len(usable)},
            "sensitivity": detectable_rho(len(usable)),
            "excluding_r1": {"rho": float(rho_filtered.statistic),
                             "p": float(rho_filtered.pvalue), "n": len(filtered)},
            "partial_given_retention": partial_spearman(
                [c["mesr"] for c in usable], [c["accuracy"] for c in usable],
                [c["retention"] for c in usable]),
            "within_retention": within_retention_correlations(usable),
            "per_seed_all_conditions": per_seed,
        },
        "retention": {
            "vs_accuracy": {"rho": float(retention_accuracy.statistic),
                            "p": float(retention_accuracy.pvalue), "n": len(usable)},
            "vs_mesr": {"rho": float(retention_mesr.statistic),
                        "p": float(retention_mesr.pvalue), "n": len(usable)},
        },
        "accuracy_sd_across_seeds": {
            "median": float(np.median(spreads)) if spreads else float("nan"),
            "max": float(np.max(spreads)) if spreads else float("nan"),
            "note": ("the scale any between-method accuracy difference must beat before it "
                     "means anything"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-train-files", type=int, default=None)
    parser.add_argument("--limit-test-files", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--arch", choices=ARCHS, default=DEFAULT_ARCH,
                        help=("classifier inductive bias; the default is the shipped "
                              "model and the one every paper number comes from"))
    parser.add_argument("--out", type=Path, default=None,
                        help="write elsewhere than results/downstream_gesture.json")
    parser.add_argument("--from-artifact", action="store_true",
                        help=("recompute the statistics from the existing artifact's "
                              "`conditions` without retraining; use after changing how a "
                              "summary is scoped or computed, never to change a number"))
    args = parser.parse_args()
    out_path = args.out or OUT

    started = time.perf_counter()

    if args.from_artifact:
        existing = json.loads(out_path.read_text())
        if "coverage" not in existing:
            raise SystemExit(f"{OUT} has no `coverage` block; it predates this protocol "
                             "and cannot be rescoped. Re-run the experiment.")
        print(f"recomputing statistics from {out_path} "
              f"({len(existing['conditions'])} stored conditions); no training", flush=True)
        write_report(existing["conditions"],
                     existing["coverage"]["train"], existing["coverage"]["test"],
                     existing["protocol"]["train_samples"],
                     existing["protocol"]["test_samples"],
                     existing.get("wall_seconds", float("nan")),
                     existing["protocol"].get("arch", DEFAULT_ARCH), out_path)
        return

    methods = args.methods or (available_methods() + [RANDOM_NULL])

    print("loading samples ...", flush=True)
    train_events, train_y, train_cov = load_samples(_trials("trials_to_train.txt"),
                                                    args.limit_train_files)
    test_events, test_y, test_cov = load_samples(_trials("trials_to_test.txt"),
                                                 args.limit_test_files)
    print(f"  train {len(train_events)}/{train_cov['gestures_offered']} samples "
          f"({train_cov['kept_fraction']:.1%}), "
          f"test {len(test_events)}/{test_cov['gestures_offered']} samples "
          f"({test_cov['kept_fraction']:.1%})  ({time.perf_counter() - started:.0f}s)",
          flush=True)
    worst = min(v["kept_fraction"] for v in test_cov["per_class"].values())
    print(f"  worst per-class test coverage: {worst:.1%}", flush=True)

    conditions = []
    for method in methods:
        t0 = time.perf_counter()
        train_scores = [score_events(method, Recording(e, None, WIDTH, HEIGHT, "", False))
                        for e in train_events]
        test_scores = [score_events(method, Recording(e, None, WIDTH, HEIGHT, "", False))
                       for e in test_events]
        print(f"[{method}] scored in {time.perf_counter() - t0:.0f}s", flush=True)

        for retention in RETENTIONS:
            train_keep = [retain_mask(s, retention) for s in train_scores]
            test_keep = [retain_mask(s, retention) for s in test_scores]
            train_x = np.stack([voxelise(e, k) for e, k in zip(train_events, train_keep)])
            test_x = np.stack([voxelise(e, k) for e, k in zip(test_events, test_keep)])

            values = [mesr(e[k][:, 1], e[k][:, 2], WIDTH, HEIGHT)
                      for e, k in zip(test_events, test_keep)]
            values = [v for v in values if np.isfinite(v)]
            per_seed = [train_and_score(train_x, train_y, test_x, test_y, s, args.arch)
                        for s in SEEDS]
            kept = float(np.mean([k.mean() for k in test_keep]))

            conditions.append({
                "method": method, "retention": float(retention),
                "accuracy": float(np.mean(per_seed)),
                "accuracy_sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
                "accuracy_per_seed": {str(s): float(a) for s, a in zip(SEEDS, per_seed)},
                "mesr": float(np.mean(values)) if values else float("nan"),
                "evaluable_test_samples": len(values),
                "actual_kept_fraction": kept,
            })
            row = conditions[-1]
            print(f"  r={retention:.2f}  acc={row['accuracy']:.4f}+-{row['accuracy_sd']:.4f}  "
                  f"mesr={row['mesr']:.4f}  kept={kept:.3f}", flush=True)
            del train_x, test_x

    write_report(conditions, train_cov, test_cov, len(train_events), len(test_events),
                 time.perf_counter() - started, args.arch, out_path)


def write_report(conditions, train_cov, test_cov, n_train: int, n_test: int,
                 wall_seconds: float, arch: str = DEFAULT_ARCH,
                 out_path: Path | None = None) -> dict:
    """Assemble and write the artifact. Split out so `--from-artifact` can reuse it."""

    paper_rows = [c for c in conditions if c["method"] in PAPER_METHODS]
    extra = sorted({c["method"] for c in conditions} - set(PAPER_METHODS))

    payload = {
        "protocol": {
            "sensor": [WIDTH, HEIGHT],
            "gesture_window": "whole labelled gesture (no wall-clock truncation)",
            "max_events_per_sample": MAX_EVENTS, "min_retention": MIN_RETENTION,
            "retentions": list(RETENTIONS), "time_bins": TIME_BINS,
            "seeds": list(SEEDS), "epochs": EPOCHS, "batch": BATCH, "lr": LR,
            "arch": arch,
            "train_samples": n_train, "test_samples": n_test,
            "paper_methods": list(PAPER_METHODS),
            "methods_outside_the_paper": extra,
            "split": "official trials_to_train.txt / trials_to_test.txt",
            "sensor_caveat": ("K = 16384 < 30000-event slice, so the occupied-pixel ceiling "
                              "in ln binds here; absolute MESR is not comparable with the "
                              "346x260 and 1280x720 corpora."),
            "selection_caveat": ("Gestures emitting fewer than max_events_per_sample events "
                                 "cannot be evaluated at min_retention and are excluded. "
                                 "See `coverage` for exactly what that cost, per class."),
            "scope_note": ("Headline statistics cover `paper_methods` only. Any method in "
                           "`methods_outside_the_paper` was scored and is kept in "
                           "`conditions`, but belongs to a sibling project and is summarised "
                           "separately under `extended_scope`."),
        },
        "coverage": {"train": train_cov, "test": test_cov},
        "conditions": conditions,
        **_flatten_summary(summarise_conditions(paper_rows)),
        "wall_seconds": wall_seconds,
    }
    if extra:
        payload["extended_scope"] = summarise_conditions(conditions)

    out_path = out_path or OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=float) + "\n")

    print("\n=== corpus coverage (official split -> evaluated) ===")
    for name, cov in (("train", train_cov), ("test", test_cov)):
        worst_class = min(cov["per_class"].items(), key=lambda kv: kv[1]["kept_fraction"])
        print(f"  {name:<6} {cov['gestures_kept']}/{cov['gestures_offered']} "
              f"({cov['kept_fraction']:.1%})   worst class {worst_class[0]}: "
              f"{worst_class[1]['kept_fraction']:.0%}")

    summary = summarise_conditions(paper_rows)
    stats, retention = summary["mesr_vs_accuracy"], summary["retention"]
    print(f"\n=== MESR vs downstream accuracy ({len(paper_rows)} paper conditions) ===")
    print(f"  all conditions      rho = {stats['all']['rho']:+.3f}  "
          f"p = {stats['all']['p']:.3e}  n = {stats['all']['n']}")
    print(f"  excluding r = 1.0   rho = {stats['excluding_r1']['rho']:+.3f}  "
          f"p = {stats['excluding_r1']['p']:.3e}  n = {stats['excluding_r1']['n']}")
    partial = stats["partial_given_retention"]
    print(f"  partial | retention rho = {partial['rho']:+.3f}  p = {partial['p']:.3e}  "
          f"n = {partial['n']}")
    within = stats["within_retention"]
    print(f"  within-retention    mean rho = {within['mean_rho']:+.3f}  "
          f"Fisher-combined p = {within['fisher_p']:.3f}")
    for seed, row in stats["per_seed_all_conditions"].items():
        print(f"    seed {seed}        rho = {row['rho']:+.3f}  p = {row['p']:.3f}")
    print(f"\n  retention -> accuracy rho = {retention['vs_accuracy']['rho']:+.3f}  "
          f"p = {retention['vs_accuracy']['p']:.3e}")
    print(f"  retention -> MESR     rho = {retention['vs_mesr']['rho']:+.3f}  "
          f"p = {retention['vs_mesr']['p']:.3e}")
    if extra:
        print(f"\n  excluded from the above (sibling project): {', '.join(extra)}"
              f"  -- see `extended_scope`")
    print(f"\nWrote {out_path}")
    return payload


def _flatten_summary(summary: dict) -> dict:
    """Publish the paper-scope summary under the keys the digest and paper already use."""

    return {"spearman_mesr_vs_accuracy": summary["mesr_vs_accuracy"],
            "spearman_retention": summary["retention"],
            "accuracy_sd_across_seeds": summary["accuracy_sd_across_seeds"],
            "scope": {"methods": summary["methods"],
                      "n_conditions": summary["n_conditions"]}}


if __name__ == "__main__":
    main()
