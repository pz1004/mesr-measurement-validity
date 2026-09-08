"""Do rankings under the protocol differ from rankings at native operating points?

If they agree perfectly, the protocol is hygiene rather than a finding, and the paper must
be framed that way. This script answers it quantitatively.

Gate arithmetic needs enough methods to rank. With fewer than `MIN_METHODS_FOR_GATE` the
correlation is degenerate and the verdict is reported as `undecidable` rather than as
agreement - a Spearman rho over two or three points is not evidence of anything.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .denoisors import has_native_operating_point
from .protocol import rank_agreement, rank_methods

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = Path(__file__).resolve().parents[1] / "results/figures"
OUT = RESULTS / "rank_analysis.json"
MIN_METHODS_FOR_GATE = 4
DATASETS = ("dnd21", "dvsclean", "emlb", "dvsd22", "pure_ba", "ed24")


def _mesr_at(curve: List[Dict], retention: float) -> float:
    """MESR at the evaluable grid point nearest `retention`; NaN if none is evaluable."""

    if not np.isfinite(retention):
        return float("nan")
    usable = [c for c in curve if c.get("evaluable") and np.isfinite(c["mesr"])]
    if not usable:
        return float("nan")
    return float(min(usable, key=lambda c: abs(c["r"] - retention))["mesr"])


def aggregate(dataset: str) -> List[Dict]:
    """Mean protocol row per method across a dataset's recordings."""

    payload = json.loads((RESULTS / f"benchmark_{dataset}.json").read_text())
    per_method: Dict[str, List[Dict]] = {}
    for record in payload["records"]:
        for method, row in record["methods"].items():
            if "error" in row:
                continue
            per_method.setdefault(method, []).append(row)
    rows = []
    for method, entries in per_method.items():
        # A continuous scorer has a whole curve, not an operating point. Older artifacts
        # stored 0.0 for those; the method's own definition is authoritative.
        if has_native_operating_point(method):
            native = [_mesr_at(e["curve"], e["native_retention"]) for e in entries]
            native = [v for v in native if np.isfinite(v)]
        else:
            native = []
        rows.append({
            "method": method,
            "is_oracle": bool(entries[0].get("is_oracle", False)),
            "is_null": bool(entries[0].get("is_null", False)),
            "mesr_star": float(np.mean([e["mesr_star"] for e in entries])),
            "delta_over_raw_at_r_star": float(np.mean(
                [e["delta_over_raw_at_r_star"] for e in entries])),
            "mesr_at_native": float(np.mean(native)) if native else float("nan"),
            "auc_over_r": float(np.mean([e["auc_over_r"] for e in entries])),
            "r_star": float(np.mean([e["r_star"] for e in entries])),
            "native_retention": (float(np.mean([e["native_retention"] for e in entries]))
                                 if has_native_operating_point(method) else float("nan")),
            "recordings": len(entries),
        })
    return rows


def per_recording_native_deltas(dataset: str) -> Dict[str, List[float]]:
    """Delta-over-Raw at each method's native operating point, one value per recording.

    `aggregate` collapses these to a mean, which is what the tables print. Keeping the
    per-recording values is what makes a confidence interval possible, and without one a
    reader cannot tell whether an ordering of means is resolvable at this sample size.
    """

    payload = json.loads((RESULTS / f"benchmark_{dataset}.json").read_text())
    out: Dict[str, List[float]] = {}
    for record in payload["records"]:
        methods = record["methods"]
        raw = methods.get("raw", {})
        if "curve" not in raw or "error" in raw:
            continue
        raw_native = _mesr_at(raw["curve"], raw.get("native_retention", 1.0))
        if not np.isfinite(raw_native):
            continue
        for method, row in methods.items():
            if "error" in row or "curve" not in row:
                continue
            if not has_native_operating_point(method) or row.get("is_oracle"):
                continue
            value = _mesr_at(row["curve"], row["native_retention"])
            if np.isfinite(value):
                out.setdefault(method, []).append(float(value - raw_native))
    return out


def oracle_violations(dataset: str) -> Dict:
    """How often does a real denoiser out-score a PERFECT label-based filter?

    `label_oracle` ranks every event by its ground-truth label, so its MESR@r curve is the
    best a label-driven filter can reach on the same retention grid. A method beating it
    cannot be denoising better - there is nothing better than the labels. Every such cell is
    direct evidence that MESR's optimum is not the ground-truth ranking.
    """

    payload = json.loads((RESULTS / f"benchmark_{dataset}.json").read_text())
    cells, margins, per_method = 0, [], {}
    at_native = 0
    grid_step = (max(payload["retentions"]) - min(payload["retentions"])) / max(
        1, len(payload["retentions"]) - 1)
    for record in payload["records"]:
        oracle = record["methods"].get("label_oracle", {})
        if "mesr_star" not in oracle:
            continue
        for method, row in record["methods"].items():
            if method == "label_oracle" or "mesr_star" not in row or row.get("is_null"):
                continue
            cells += 1
            margin = row["mesr_star"] - oracle["mesr_star"]
            margins.append(margin)
            if margin > 0:
                # Does the violation happen where the method actually operates, or only at
                # a swept point where a binary filter is really "filter + decimation"?
                # If the latter dominated, the violation would be an artifact of the sweep.
                native = row.get("native_retention", float("nan"))
                if np.isfinite(native) and abs(row["r_star"] - native) <= grid_step + 1e-9:
                    at_native += 1
                per_method.setdefault(method, []).append(
                    {"recording": record["recording"], "margin": margin,
                     "method_mesr_star": row["mesr_star"],
                     "oracle_mesr_star": oracle["mesr_star"]})
    if not cells:
        return {"comparable_cells": 0}
    beating = sum(len(v) for v in per_method.values())
    worst = max((c for v in per_method.values() for c in v),
                key=lambda c: c["margin"], default=None)
    return {
        "comparable_cells": cells,
        "cells_beating_oracle": beating,
        "share_beating_oracle": beating / cells,
        "violations_at_native_operating_point": at_native,
        "share_of_violations_at_native": at_native / beating if beating else float("nan"),
        "max_margin_over_oracle": float(np.max(margins)),
        "mean_margin_over_oracle": float(np.mean(margins)),
        "by_method": {m: len(v) for m, v in sorted(per_method.items())},
        "worst_case": worst,
    }


def null_gain_at_fixed_retention(dataset: str,
                                 nulls: Sequence[str] = ("raw", "random_null")) -> Dict:
    """Does a null raise MESR at a retention fixed in advance, rather than at its own argmax?

    This exists because the per-recording argmax is not a safe summary for a null. Retention
    r = 1.0 keeps every event, so Delta-over-Raw is identically 0 there, and therefore
    `max_r Delta >= 0` for every recording *by construction*. The mean of that maximum is
    positive under a true null, and a bootstrap interval around it excludes zero for the same
    structural reason - not because discarding events raised anything. Quoting it would be
    the exact metric-oracle error this paper warns about in the reporting protocol.

    The oracle-free question is whether the null's mean Delta is positive at a retention
    chosen before looking at the recording. We answer it by profiling every grid point:
    `raises_mesr_at_a_fixed_retention` is False when no r < 1 has a positive mean, which is a
    statement about the whole grid and needs no selection at all. `best_fixed` reports the
    strongest grid point, selected once on the mean curve (one choice among 19) rather than
    once per recording (96 x 19), and r = 1.0 is excluded from that argmax so the quantity
    can come out negative - which on E-MLB it does.
    """

    payload = json.loads((RESULTS / f"benchmark_{dataset}.json").read_text())
    grid = [float(r) for r in payload["retentions"]]
    out: Dict[str, Dict] = {}
    for null in nulls:
        per_r: Dict[float, List[float]] = {r: [] for r in grid}
        scenes_r: Dict[float, List[str]] = {r: [] for r in grid}
        max_selected, argmax_at_one = [], 0
        for record in payload["records"]:
            row = record["methods"].get(null)
            if row is None or "error" in row or "raw_mesr" not in row:
                continue
            base = row["raw_mesr"]
            points = [(float(c["r"]), c["mesr"] - base) for c in row["curve"]
                      if c.get("evaluable") and np.isfinite(c["mesr"])]
            if not points:
                continue
            scene = scene_of(record["recording"])
            for r, delta in points:
                if r in per_r:
                    per_r[r].append(float(delta))
                    scenes_r[r].append(scene)
            best_r, best_delta = max(points, key=lambda t: t[1])
            max_selected.append(best_delta)
            argmax_at_one += int(best_r >= 1.0)
        profile = []
        for r in grid:
            values = per_r[r]
            if not values:
                continue
            interval = bootstrap_delta_ci(values)
            # The median and the share of recordings above raw say whether a positive mean is
            # the typical recording or a heavy right tail. On Pure_BA it is the tail.
            profile.append({"r": r, "mean": interval["mean"], "lo": interval["lo"],
                            "hi": interval["hi"], "n": interval["n"],
                            "median": float(np.median(values)),
                            "n_above_raw": int(sum(v > 0 for v in values)),
                            "scenes": scenes_r[r]})
        # r = 1.0 is the identity and contributes a structural zero; excluding it is what
        # lets this argmax report a negative number when the null genuinely never helps.
        candidates = [p for p in profile if p["r"] < 1.0]
        best = max(candidates, key=lambda p: p["mean"]) if candidates else None
        positive = [p for p in candidates if p["mean"] > 0]
        if best is not None:
            # The recording-level interval assumes recordings are independent draws; they are
            # not. Report the interval the corpus can actually carry alongside it.
            best = dict(best)
            best["cluster_ci"] = cluster_bootstrap_delta_ci(per_r[best["r"]],
                                                            scenes_r[best["r"]])
            best.pop("scenes", None)
        for point in profile:
            point.pop("scenes", None)
        out[null] = {
            "profile": profile,
            "best_fixed": best,
            "n_retentions_with_positive_mean": len(positive),
            "n_retentions_below_one": len(candidates),
            "raises_mesr_at_a_fixed_retention": bool(positive),
            "max_selected": {
                "mean": float(np.mean(max_selected)) if max_selected else float("nan"),
                "n": len(max_selected),
                "share_argmax_at_r1": (argmax_at_one / len(max_selected)
                                       if max_selected else float("nan")),
                "note": ("mean of a per-recording max that is bounded below by 0 because "
                         "Delta(r=1) == 0; reported only to show the size of the selection "
                         "effect, never as a gain"),
            },
        }
    return out


def edformer_reference() -> Dict:
    """EDformer's verified E-MLB row, placed on the protocol's Delta-over-Raw axis.

    EDformer cannot be re-run here (it needs its own pinned environment), so its per-cell
    numbers are taken from the parent project's reproduction. Its curve cannot be swept at
    all: the operating point is baked into `EDformer/eval_mesr.py` as `sigmoid >= 0.005`.
    That is not a gap in this benchmark, it is an instance of the problem the paper
    documents.

    TWO MISMATCHES A READER MUST SEE, both reported in the returned dict rather than
    described in prose:

    `reproduction_error` - the reproduction matches the published table to +0.0010 ON
    AVERAGE, but that mean is the residue of per-cell errors that cancel (-0.0195 to
    +0.0402). Since these per-cell values are what this row is built from, and since the
    paper's unit of scale is the 0.0092 EDformer-EDmamba gap, the per-cell spread is the
    honest figure: it is larger than the gap it is being compared against.

    `recording_sets` - both runs now cover the same 384 E-MLB recordings; they differ only
    in the 1,000,000-event cap on the classical rows, and Delta-over-Raw is cap-invariant.
    Earlier revisions ran the classical rows on a 96-recording subset, and this block
    reported that as a mismatch; the counts are read from the artifacts so that a divergence
    reappears here rather than being asserted either way.
    """

    path = (Path(__file__).resolve().parents[1]
            / "reproduction/results/emlb_edformer_mesr.json")
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    cells, deltas, retentions = [], [], []
    for part, levels in payload["results"].items():
        for level, cell in levels.items():
            delta = cell["edformer_mesr"] - cell["raw_mesr"]
            cells.append({"part": part, "nd": level, "raw_mesr": cell["raw_mesr"],
                          "edformer_mesr": cell["edformer_mesr"],
                          "delta_over_raw": delta,
                          "mean_retention": cell["mean_retention"],
                          "paper_edformer": cell["paper_edformer"]})
            deltas.append(delta)
            retentions.append(cell["mean_retention"])
    errors = [c["edformer_mesr"] - c["paper_edformer"] for c in cells]
    return {
        "source": "reproduction/results/emlb_edformer_mesr.json",
        "operating_point": "published protocol, sigmoid >= 0.005 (EDformer/eval_mesr.py:75)",
        "sweepable": False,
        "reason_not_sweepable": ("the threshold is hard-coded in the released evaluation "
                                 "script, so no MESR@r curve can be produced from it"),
        "cells": cells,
        "mean_delta_over_raw": float(np.mean(deltas)),
        # Bootstrapped over the 8 (lighting, ND) cells, not over recordings: the artifact
        # stores cell aggregates only. It is therefore a coarser interval than the
        # classical rows' 96-recording one, and is labelled as such wherever it is quoted.
        "delta_over_raw_ci": {**bootstrap_delta_ci(deltas),
                              "resampling_unit": "(lighting, ND) cell, n=8",
                              "comparable_to_classical_ci": False},
        "retention_range": [float(np.min(retentions)), float(np.max(retentions))],
        "max_events_per_sequence": payload["protocol"]["max_events_per_sequence"],
        "reproduction_error": {
            "mean": float(np.mean(errors)),
            "min": float(np.min(errors)),
            "max": float(np.max(errors)),
            "max_abs": float(np.max(np.abs(errors))),
            "n_cells": len(errors),
            "note": ("mean is the residue of cancelling per-cell errors; quote max_abs "
                     "when claiming agreement, not mean"),
        },
        "recording_sets": _recording_sets(payload),
    }


def _recording_sets(payload: Dict) -> Dict:
    """How the EDformer row and the classical rows are drawn, both counted from artifacts.

    Hardcoding these once left `NUMBERS.md` asserting a 96-recording, 12-scene classical run
    for three revisions after that run was extended to the full 384. A digest that states a
    stale fact is worse than one that omits it, because the paper points readers at it.
    """

    edformer = sum(cell["sequences"] for levels in payload["results"].values()
                   for cell in levels.values())
    classical_path = Path(__file__).resolve().parents[1] / "results/benchmark_emlb.json"
    classical, clusters = 0, 0
    if classical_path.exists():
        records = json.loads(classical_path.read_text())["records"]
        classical = len(records)
        clusters = len({scene_of(r["recording"]) for r in records})
    return {
        "edformer_recordings": edformer,
        "edformer_cap": payload["protocol"]["max_events_per_sequence"],
        "classical_recordings": classical,
        "classical_scene_clusters": clusters,
        "classical_cap": 1_000_000,
        "matched_on_recordings": bool(classical) and classical == edformer,
        "note": ("Delta-over-Raw is cap-invariant, so matched recording sets under "
                 "different caps are comparable"),
    }


def scene_of(recording: str) -> str:
    """The independent unit a recording belongs to; the recording itself when it has none.

    `bootstrap_delta_ci` resamples recordings, which assumes recordings are independent
    draws. Several corpora do not satisfy that, and the release layout says so in the file
    names: DND21's ten recordings are two base scenes at five injected rates, DVSCLEAN's ten
    are five sequences at two noise densities, E-MLB's 384 are 48 scenes at four ND levels
    under two lighting conditions - 96 (scene, lighting) clusters of four - and three DVSD22
    recordings repeat one drop rate.
    Resampling recordings therefore reports a narrower interval than the design supports.
    This recovers the unit so `cluster_bootstrap_delta_ci` can resample it instead.
    """

    for pattern, prefix in ((r"^(E-MLB/[^/]+/.+?)-ND\d+-\d+$", None),
                            (r"^(DVSCLEAN/[^_]+)_\d+$", None),
                            (r"^DND21/\d+hz_(.+)$", "DND21/"),
                            (r"^(DVSD22/.+?)_[a-z]$", None)):
        match = re.match(pattern, recording)
        if match:
            return f"{prefix or ''}{match.group(1)}"
    return recording


def cluster_bootstrap_delta_ci(values: Sequence[float], clusters: Sequence[str],
                               draws: int = 10_000, seed: int = 20260726) -> Dict:
    """Percentile bootstrap that resamples independent scenes rather than recordings.

    This is the interval a corpus can actually carry. Where every recording is its own scene
    it reduces to `bootstrap_delta_ci`; where recordings are siblings of a smaller number of
    scenes it is wider, and on DVSCLEAN - five sequences at two noise densities - it is wide
    enough to change the verdict, which is why the paper does not count that corpus.
    """

    grouped: Dict[str, List[float]] = {}
    for value, cluster in zip(values, clusters):
        if np.isfinite(value):
            grouped.setdefault(cluster, []).append(float(value))
    keys = sorted(grouped)
    flat = np.concatenate([np.asarray(grouped[k]) for k in keys]) if keys else np.empty(0)
    if len(keys) < 2:
        return {"mean": float(flat.mean()) if len(flat) else float("nan"),
                "lo": float("nan"), "hi": float("nan"),
                "n": int(len(flat)), "n_clusters": len(keys)}
    arrays = [np.asarray(grouped[k], dtype=float) for k in keys]
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(keys), size=(draws, len(keys)))
    means = np.array([np.concatenate([arrays[j] for j in row]).mean() for row in picks])
    return {"mean": float(flat.mean()),
            "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)),
            "n": int(len(flat)), "n_clusters": len(keys), "draws": int(draws),
            "excludes_zero": bool(np.percentile(means, 2.5) > 0
                                  or np.percentile(means, 97.5) < 0)}


def bootstrap_delta_ci(values: Sequence[float], draws: int = 10_000,
                       seed: int = 20260726) -> Dict:
    """Percentile bootstrap CI for a mean Delta-over-Raw across recordings.

    Table 5 orders methods by a mean over recordings, and the reader cannot tell from a
    point estimate whether the ordering is resolvable at this sample size. This answers
    that directly, and it also bounds how much re-drawing recordings alone moves a mean
    Delta at this sample size.
    """

    array = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(array) < 2:
        return {"mean": float(array[0]) if len(array) else float("nan"),
                "lo": float("nan"), "hi": float("nan"), "n": int(len(array))}
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return {"mean": float(array.mean()),
            "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)),
            "n": int(len(array)), "draws": int(draws)}


#: Fixed style per method. Matplotlib's default cycle assigns colours by plot order, so a
#: series moved colour between panels whenever `label_oracle` was absent -- and the collision
#: was the worst one available: red meant `label_oracle` on the two labelled corpora and
#: `random_null` on the other three, which are conceptual opposites. Colours are pinned here
#: instead. The two nulls and the oracle are drawn heavier and dashed so the three series the
#: figure exists to show are separable from the six filters at a glance.
CURVE_STYLE: Dict[str, Dict] = {
    "raw":          {"color": "#000000", "linestyle": "--", "linewidth": 2.2, "zorder": 5},
    "random_null":  {"color": "#d62728", "linestyle": "--", "linewidth": 2.2, "zorder": 5},
    "label_oracle": {"color": "#2ca02c", "linestyle": ":",  "linewidth": 2.2, "zorder": 5},
    "dwf":          {"color": "#1f77b4"},
    "evflow":       {"color": "#ff7f0e"},
    "knoise":       {"color": "#8c564b"},
    "red":          {"color": "#9467bd"},
    "ts":           {"color": "#e377c2"},
    "ynoise":       {"color": "#7f7f7f"},
    "native3d":       {"color": "#17becf"},
    "native3d_gated": {"color": "#bcbd22"},
}
#: Nulls and oracle first so the legend leads with them.
CURVE_ORDER = ["raw", "random_null", "label_oracle"]


def _curve_style(method: str) -> Dict:
    """Style for one series; unknown methods fall back to a thin grey line."""

    base = {"marker": "o", "markersize": 3, "linewidth": 1.2}
    return {**base, **CURVE_STYLE.get(method, {"color": "#aaaaaa"})}


def figures(report: Dict) -> List[Path]:
    """MESR@r curves per dataset, plus the null-vs-raw control panel."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    written = []

    datasets = [d for d in report if report[d].get("rows")]
    if datasets:
        fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4),
                                 squeeze=False)
        for axis, dataset in zip(axes[0], datasets):
            payload = json.loads((RESULTS / f"benchmark_{dataset}.json").read_text())
            curves: Dict[str, List[List[float]]] = {}
            for record in payload["records"]:
                for method, row in record["methods"].items():
                    if "curve" not in row:
                        continue
                    for point in row["curve"]:
                        if point["evaluable"]:
                            curves.setdefault(method, []).append([point["r"], point["mesr"]])
            ordered = ([m for m in CURVE_ORDER if m in curves]
                       + sorted(m for m in curves if m not in CURVE_ORDER))
            for method in ordered:
                array = np.array(curves[method])
                grid = np.unique(array[:, 0])
                mean = [array[array[:, 0] == r, 1].mean() for r in grid]
                axis.plot(grid, mean, label=method, **_curve_style(method))
            axis.set_title(dataset)
            axis.set_xlabel("retention r (fraction of events kept)")
            axis.set_ylabel("MESR (mean over recordings)")
            axis.grid(alpha=0.3)
            axis.legend(fontsize=8)
        fig.tight_layout()
        path = FIGURES / "fig_retention_curves.pdf"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)

    regimes = RESULTS / "noise_regimes.json"
    if regimes.exists():
        payload = json.loads(regimes.read_text())
        fig, axis = plt.subplots(figsize=(6, 4))
        for synthetic, colour, label in ((True, "tab:red", "synthetic"),
                                         (False, "tab:blue", "real")):
            xs = [p["spatial_autocorr"] for p in payload["profiles"]
                  if p["synthetic"] is synthetic]
            ys = [p["local_global_coupling"] for p in payload["profiles"]
                  if p["synthetic"] is synthetic]
            axis.scatter(xs, ys, c=colour, label=label, alpha=0.7, s=25)
        axis.set_xlabel("spatial autocorrelation (share of events with an occupied neighbour)")
        axis.set_ylabel("local-global coupling (corr)")
        axis.grid(alpha=0.3)
        axis.legend()
        fig.tight_layout()
        path = FIGURES / "fig_regime_separation.pdf"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)

    return written


def main() -> None:
    report = {}
    for dataset in DATASETS:
        path = RESULTS / f"benchmark_{dataset}.json"
        if not path.exists():
            continue
        rows = aggregate(dataset)
        rankable = [r for r in rows if not r["is_oracle"]]
        entry: Dict = {"rows": rows, "n_rankable_methods": len(rankable),
                       "oracle_violations": oracle_violations(dataset),
                       "null_gain_fixed_retention": null_gain_at_fixed_retention(dataset)}
        # A continuous scorer has no native operating point (native_retention is NaN), so
        # it cannot appear in the native ranking; comparing the two rankings requires the
        # same method set on both sides.
        with_native = [r for r in rankable if np.isfinite(r["mesr_at_native"])]
        if len(with_native) >= MIN_METHODS_FOR_GATE:
            rankable = with_native
            rho_native, tau_native = rank_agreement(rankable, "mesr_star", "mesr_at_native")
            rho_delta, _ = rank_agreement(rankable, "mesr_star",
                                          "delta_over_raw_at_r_star")
            rho_auc, tau_auc = rank_agreement(rankable, "mesr_star", "auc_over_r")
            # THE COMPARISON THE PAPER REPORTS. `mesr_star` is selected at the argmax of the
            # test metric, so ranking on it is the metric-oracle error the protocol warns
            # about; `auc_over_r` is the oracle-free summary the protocol actually
            # recommends. Comparing that against the native ranking is the honest question,
            # and the distance between the two comparisons measures how much apparent rank
            # instability operating-point selection creates on its own.
            rho_auc_native, tau_auc_native = rank_agreement(
                rankable, "auc_over_r", "mesr_at_native")
            entry.update({
                "spearman_auc_vs_native": rho_auc_native,
                "kendall_auc_vs_native": tau_auc_native,
                "rankings_differ_auc_vs_native": (
                    rank_methods(rankable, "auc_over_r")
                    != rank_methods(rankable, "mesr_at_native")),
                "rank_protocol": rank_methods(rankable, "mesr_star"),
                "rank_native": rank_methods(rankable, "mesr_at_native"),
                "rank_delta": rank_methods(rankable, "delta_over_raw_at_r_star"),
                "rank_auc_over_r": rank_methods(rankable, "auc_over_r"),
                "methods_without_native_operating_point": [
                    r["method"] for r in rows
                    if not np.isfinite(r["mesr_at_native"]) and not r["is_oracle"]],
                "delta_over_raw_at_native": {
                    r["method"]: r["mesr_at_native"] - next(
                        x["mesr_at_native"] for x in rows if x["method"] == "raw")
                    for r in rankable},
                # 95% percentile bootstrap over recordings. Two things a point estimate
                # hides: whether adjacent rows in the table are separable at all, and how
                # much of a cross-run gap could be scene sampling rather than method.
                "delta_over_raw_at_native_ci": {
                    method: bootstrap_delta_ci(values)
                    for method, values in sorted(
                        per_recording_native_deltas(dataset).items())},
                "spearman_protocol_vs_auc": rho_auc,
                "kendall_protocol_vs_auc": tau_auc,
                "metric_oracle_warning": (
                    "r_star and mesr_star are selected at the argmax of the TEST metric, so "
                    "they are metric-oracle quantities and bound what a method could reach, "
                    "not what a paper could honestly report. auc_over_r (mean MESR over the "
                    "evaluable retention range) and mesr_at_native are the oracle-free "
                    "summaries; rank on those when the claim must be achievable."),
                "spearman_protocol_vs_native": rho_native,
                "kendall_protocol_vs_native": tau_native,
                "spearman_protocol_vs_delta": rho_delta,
                "rankings_differ": (rank_methods(rankable, "mesr_star")
                                    != rank_methods(rankable, "mesr_at_native")),
                "gate": "decided",
            })
        else:
            entry.update({
                "gate": "undecidable",
                "reason": (f"only {len(with_native)} rankable method(s) with a native "
                           f"operating point; the gate needs "
                           f"{MIN_METHODS_FOR_GATE}. Blocked on the cuke-emlb build "
                           f"(BUILD_CUKE_EMLB.md)."),
            })
        if dataset == "emlb":
            entry["edformer_reference"] = edformer_reference()
        report[dataset] = entry

    OUT.write_text(json.dumps(report, indent=2, default=float) + "\n")
    for dataset, entry in report.items():
        print(f"\n=== {dataset} ===  gate: {entry['gate']}")
        for row in sorted(entry["rows"], key=lambda r: -r["mesr_star"]):
            tag = " [oracle]" if row["is_oracle"] else (" [null]" if row["is_null"] else "")
            print(f"  {row['method']:<14s}{tag:<9s} mesr*={row['mesr_star']:.4f} "
                  f"r*={row['r_star']:.2f}  d-over-raw={row['delta_over_raw_at_r_star']:+.4f} "
                  f"native_r={row['native_retention']:.2f}  n={row['recordings']}")
        if entry["gate"] == "decided":
            print("  protocol (oracle-selected r*):", entry["rank_protocol"])
            print("  native                       :", entry["rank_native"])
            print("  auc-over-r (oracle-free)     :", entry["rank_auc_over_r"])
            print("  spearman = %.3f  kendall = %.3f  differ = %s"
                  % (entry["spearman_protocol_vs_native"],
                     entry["kendall_protocol_vs_native"], entry["rankings_differ"]))
        else:
            print("  ", entry["reason"])
        for null, gain in entry["null_gain_fixed_retention"].items():
            best = gain["best_fixed"]
            if best is None:
                continue
            print("  null %-12s best FIXED r=%.2f  mean=%+.4f [%+.4f,%+.4f]  "
                  "positive at %d/%d retentions  (max-selected would say %+.4f; "
                  "%.0f%% of recordings peak at r=1)"
                  % (null, best["r"], best["mean"], best["lo"], best["hi"],
                     gain["n_retentions_with_positive_mean"], gain["n_retentions_below_one"],
                     gain["max_selected"]["mean"],
                     100 * gain["max_selected"]["share_argmax_at_r1"]))
        violations = entry["oracle_violations"]
        if violations.get("comparable_cells"):
            print("  oracle violations: %d/%d cells (%.0f%%), max margin %+.4f, by method %s"
                  % (violations["cells_beating_oracle"], violations["comparable_cells"],
                     100 * violations["share_beating_oracle"],
                     violations["max_margin_over_oracle"], violations["by_method"]))

    written = figures(report)
    print()
    for path in written:
        print("Wrote", path)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
