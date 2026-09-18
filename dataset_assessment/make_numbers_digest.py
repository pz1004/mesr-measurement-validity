"""Emit `NUMBERS.md`: every number in the paper beside the artifact it came from.

Run after any change to `results/`. Nothing in the digest is typed by hand, so a claim in
the draft that cannot be found there has no artifact behind it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from .esr import SLICE

RESULTS = Path(__file__).resolve().parents[1] / "results"
CORPORA = ("dnd21", "dvsclean", "emlb", "dvsd22", "pure_ba")


def _load(name: str) -> Dict:
    return json.loads((RESULTS / name).read_text())


def _cells(dataset: str) -> List[Dict]:
    return [row for record in _load(f"benchmark_{dataset}.json")["records"]
            for row in record["methods"].values() if "r_star" in row]


def _benchmark_scale(out: Callable[[str], None]) -> None:
    """The scale the abstract and the contributions quote: five corpora, 3,556 cells.

    This was the last headline number with no artifact behind it. It is a *derived* count,
    so no file stores it, and a reader checking the release could not confirm it. Both
    factors are derivable and both matter: `benchmark_pure_ba.json` holds 41 records but 15
    of them carry fewer than one complete 30,000-event slice, so the paper counts 26. The
    sibling-project file `benchmark_emlb_native3d.json` is deliberately excluded -- it is
    not one of the five corpora and its two methods enter no contrast (S IX).
    """

    out("\n## S0 benchmark scale  (results/benchmark_{dnd21,dvsclean,emlb,dvsd22,"
        "pure_ba}.json)\n")
    total_cells = 0
    total_evaluable = 0
    for dataset in CORPORA:
        payload = _load(f"benchmark_{dataset}.json")
        records = payload["records"]
        evaluable = [r for r in records
                     if isinstance(r.get("raw_mesr"), float) and np.isfinite(r["raw_mesr"])]
        scored = [m for r in evaluable for m in r["methods"].values() if "curve" in m]
        names = sorted({k for r in evaluable for k in r["methods"]})
        dropped = len(records) - len(evaluable)
        total_cells += len(scored)
        total_evaluable += len(evaluable)
        out(f"- **{dataset}**: {len(evaluable)} evaluable recordings of {len(records)} in "
            f"the artifact"
            + (f" ({dropped} hold under one complete {SLICE:,}-event slice)"
               if dropped else "")
            + f"; {len(names)} methods {names}; **{len(scored)} (method, recording) "
              f"cells**")
    out(f"- **total: {total_evaluable} recordings, {total_cells:,} (method, recording) "
        f"cells** -- the count S I, S III and S V-A quote")
    out(f"  - excludes results/benchmark_emlb_native3d.json (the sibling-project rows of "
        f"S IX), which adds "
        f"{len([m for r in _load('benchmark_emlb_native3d.json')['records'] for m in r['methods'].values() if 'curve' in m])} more")


def _metric_dependencies(out: Callable[[str], None]) -> None:
    def _spread_over_recordings(out: Callable[[str], None], block: dict, what: str) -> None:
        """The influence-quantity line the paper quotes: mean over recordings, with a CI."""

        summary = block["spread_over_recordings"]
        out(f"- **{what} spread over {summary['n']} recordings: mean {summary['mean']:.4f}** "
            f"[{summary['lo']:.4f}, {summary['hi']:.4f}] "
            f"sd={summary['sd']:.4f} min={summary['min']:.4f} max={summary['max']:.4f}")
        out(f"  - as a multiple of the 0.0092 published gap: **{summary['mean'] / 0.0092:.0f}x** "
            f"[{summary['lo'] / 0.0092:.0f}x, {summary['hi'] / 0.0092:.0f}x]")

    payload = _load("metric_dependencies.json")
    retention = payload["retention_dependence"]
    out("## S3.2 retention dependence  (results/metric_dependencies.json)")
    out(f"- {retention['recording']}, fixed ranking AUC {retention['fixed_ranking_auc']}")
    out(f"- min {retention['min_mesr']:.4f} max {retention['max_mesr']:.4f} "
        f"spread {retention['spread']:.4f} argmax r={retention['argmax_r']} "
        f"(single recording, kept for continuity with the stored curve)")
    _spread_over_recordings(out, retention, "retention")

    resolution = payload["resolution_dependence"]
    out("\n## S3.1 declared-resolution invariance")
    out(f"- spread over 4 declared sensor sizes "
        f"**{resolution['declared_sensor_spread']:.3e}** "
        f"(invariant={resolution['declared_sensor_invariant']})")
    for row in resolution["declared_sensor_rows"]:
        out(f"  - {row['width']}x{row['height']} K={row['K']} MESR={row['mesr']:.12f}")
    for row in resolution["occupancy_rows"]:
        out(f"  - {row['stream']}: K={row['sensor'][0] * row['sensor'][1]} "
            f"occupied={row['occupied_px']} ({row['occupied_px_share_of_K']:.3f} of K) "
            f"ntss={row['ntss']:.6f} ln={row['ln']:.1f} ESR={row['esr']:.4f}")

    slices = payload["slice_dependence"]
    out("\n## S3.3 slice-size dependence")
    out("- " + ", ".join(f"{r['slice']}:{r['mesr']:.4f}" for r in slices["rows"])
        + f"  spread {slices['spread']:.4f} (single recording)")
    _spread_over_recordings(out, slices, "slice-size")
    out(f"- monotone increasing on {slices['monotone_increasing_share']:.0%} of recordings")
    # Same-corpus scale for the verdict: the same ten DND21 recordings, each filter on its own
    # output (first 10^6 events), paired where both filters are scored.
    from .native_oracle import widest_filter_gap

    gap = widest_filter_gap(_load("native_oracle_dnd21.json")["cells"])
    mean = slices["spread_over_recordings"]["mean"]
    out(f"- same-corpus scale (`results/native_oracle_dnd21.json`): widest paired gap between "
        f"two filters' own outputs **{gap['gap']:.4f}** ({gap['a']} over {gap['b']}, "
        f"n={gap['n_recordings']}); slice-size spread = **{mean / gap['gap']:.2f}x** that")
    # Does the convention reorder filters, or shift them together? Each filter's own output
    # at every slice size, on the same first 10^6 events.
    s = _load("slice_ranking_dnd21.json")["summary"]
    out(f"- filters' own outputs across slice sizes (`results/slice_ranking_dnd21.json`): "
        f"**{s['recordings_reordered']}/{s['ranked_recordings']}** recordings reordered against "
        f"30k, top filter changes on {s['recordings_top_changed']}; Kendall tau min "
        f"{s['min_tau']:.3f}, mean {s['mean_tau']:.3f}; cohort sizes {s['cohort_sizes']}")
    out(f"  - corpus-mean paired differences: {len(s['pairs_flipping'])}/{s['pairs']} pairs "
        f"change sign ({', '.join(s['pairs_flipping']) or 'none'}); widest pair gap at 30k "
        f"{s['widest_pair_gap_at_reference']:.4f}")
    out(f"  - unfiltered spread on the same 10^6-event support: "
        f"**{s['unfiltered_spread_mean']:.4f}** (mean over recordings)")


def _controlled_addition(out: Callable[[str], None]) -> None:
    """The one published calibration of ESR, reproduced on DND21's injected series.

    E-MLB validate ESR by adding known noise to a fixed stream and showing the score falls.
    DND21 injects uniform noise into two real recordings at five rates, so the unfiltered
    row of the benchmark is a controlled-addition series of the same kind. Reported because a
    validation result that holds belongs in the record beside the four that do not.
    """

    payload = _load("benchmark_dnd21.json")
    series: Dict[str, List] = {}
    for rec in payload["records"]:
        rate, _, scene = rec["recording"].split("/")[-1].partition("_")
        series.setdefault(scene, []).append((int(rate.rstrip("hz")), rec["raw_mesr"]))

    out("\n## S3.6 controlled noise addition  (results/benchmark_dnd21.json, raw at r=1)")
    out("- the one published calibration of ESR (E-MLB add 10/20/40% noise and the score "
        "falls); DND21's five injection rates reproduce it")
    for scene, points in sorted(series.items()):
        points.sort()
        values = ", ".join(f"{v:.4f}" for _, v in points)
        rates = ", ".join(str(r) for r, _ in points)
        falling = all(b < a for (_, a), (_, b) in zip(points, points[1:]))
        out(f"  - {scene}: at {rates} Hz/px, unfiltered MESR {values} "
            f"-> **monotone decreasing: {falling}**")
    out("  - unit: two independent base scenes, not ten recordings; the five rates within a "
        "scene are one capture at five injection levels")


def _nulls(out: Callable[[str], None]) -> None:
    out("\n## S3.4 nulls and oracle  (results/benchmark_*.json, results/rank_analysis.json)")
    ranks = _load("rank_analysis.json")
    for dataset in ("dnd21", "emlb", "dvsclean", "pure_ba", "dvsd22"):
        rows = {r["method"]: r for r in ranks[dataset]["rows"]}
        records = _load(f"benchmark_{dataset}.json")["records"]
        worst = max((rec["methods"]["random_null"] for rec in records
                     if "delta_over_raw_at_r_star" in rec["methods"].get("random_null", {})),
                    key=lambda r: r["delta_over_raw_at_r_star"])
        unfiltered = [rec["raw_mesr"] for rec in records if np.isfinite(rec["raw_mesr"])]
        out(f"- {dataset}: raw {rows['raw']['delta_over_raw_at_r_star']:+.4f}, "
            f"random_null {rows['random_null']['delta_over_raw_at_r_star']:+.4f}, "
            f"worst single {worst['delta_over_raw_at_r_star']:+.4f} "
            f"at r={worst['r_star']:.2f}; "
            f"unfiltered MESR mean {np.mean(unfiltered):.4f}")
        # Uncertainty on the two headline null gains. A measurement reported without one is
        # not a measurement, and this is the table the paper's central claim rests on.
        from .analyze import bootstrap_delta_ci
        for method in ("raw", "random_null"):
            values = [rec["methods"][method]["delta_over_raw_at_r_star"] for rec in records
                      if "delta_over_raw_at_r_star" in rec["methods"].get(method, {})]
            interval = bootstrap_delta_ci(values)
            out(f"  - {method} 95% CI over {interval['n']} recordings: "
                f"[{interval['lo']:+.4f}, {interval['hi']:+.4f}]")
        # THE NUMBERS THE PAPER QUOTES. The per-recording-argmax values above are retained
        # only to show the size of the selection effect: max_r Delta >= 0 by construction
        # because Delta(r=1) == 0, so they are positive under a true null. Table 2 prints
        # the fixed-retention column below.
        for method in ("raw", "random_null"):
            gain = ranks[dataset]["null_gain_fixed_retention"][method]
            best = gain["best_fixed"]
            out(f"  - **{method} at FIXED r={best['r']:.2f}: {best['mean']:+.4f}** "
                f"[{best['lo']:+.4f}, {best['hi']:+.4f}] over {best['n']} recordings; "
                f"positive mean at {gain['n_retentions_with_positive_mean']}/"
                f"{gain['n_retentions_below_one']} retentions below 1; "
                f"raises_mesr={gain['raises_mesr_at_a_fixed_retention']}; "
                f"{100 * gain['max_selected']['share_argmax_at_r1']:.0f}% of recordings "
                f"peak at r=1")
            # Recordings are not independent draws (see `scene_of`), and the mean is not the
            # median on every corpus. Both decide how many corpora the paper counts.
            cluster = best["cluster_ci"]
            out(f"    - median {best['median']:+.4f}; "
                f"{best['n_above_raw']}/{best['n']} recordings above raw; "
                f"**cluster CI over {cluster['n_clusters']} independent scenes: "
                f"[{cluster['lo']:+.4f}, {cluster['hi']:+.4f}]** "
                f"excludes_zero={cluster['excludes_zero']}")
    out("- EDformer vs EDmamba as published on E-MLB: 1.00588 vs 1.01513 -> gap 0.0092")

    out("\n## S3.5 floor-pinned optima")
    total = 0
    for dataset in CORPORA:
        cells = _cells(dataset)
        pinned = sum(bool(c.get("optimum_at_evaluable_floor")) for c in cells)
        total += len(cells)
        out(f"- {dataset}: {pinned}/{len(cells)} ({100 * pinned / len(cells):.0f}%)")
    out(f"- total cells across corpora: {total}")


def _benchmark(out: Callable[[str], None]) -> None:
    out("\n## S5 benchmark and ranking  (results/rank_analysis.json)")
    for dataset, entry in _load("rank_analysis.json").items():
        out(f"\n### {dataset} (n={max(r['recordings'] for r in entry['rows'])})")
        out(f"- **AUC_r vs native (oracle-free, the paper's Table): "
            f"spearman {entry['spearman_auc_vs_native']:.3f} "
            f"kendall {entry['kendall_auc_vs_native']:.3f} "
            f"differ={entry['rankings_differ_auc_vs_native']}**")
        out(f"- r* vs native (metric-oracle, shown beside it): "
            f"spearman {entry['spearman_protocol_vs_native']:.3f} "
            f"kendall {entry['kendall_protocol_vs_native']:.3f} "
            f"differ={entry['rankings_differ']}")
        out(f"- protocol  : {entry['rank_protocol']}")
        out(f"- native    : {entry['rank_native']}")
        out(f"- auc-over-r: {entry['rank_auc_over_r']}")
        out("- delta-over-raw at native: " + ", ".join(
            f"{k} {v:+.4f}" for k, v in sorted(entry["delta_over_raw_at_native"].items(),
                                               key=lambda kv: -kv[1])))
        retentions = [r for r in entry["rows"] if r.get("native_retention") is not None]
        if retentions:
            out("- native retention (paper quotes the 2 dp form): " + ", ".join(
                f"{r['method']} {r['native_retention']:.4f}={r['native_retention']:.2f}"
                for r in sorted(retentions, key=lambda r: -r["native_retention"])))
        # The matched-retention result comes first: it is what the paper quotes. The
        # `mesr_star`-vs-`mesr_star` count below is unmatched and metric-oracle-selected,
        # and is kept only so the two can be compared.
        lq = RESULTS / f"label_quality_{dataset}.json"
        if lq.is_file():
            s = json.loads(lq.read_text())["summary"]
            n = s["at_native"]
            out(f"- **label quality at MATCHED retention, at each method's own operating "
                f"point: {n['mesr_wins']}/{n['eligible_pairs']} cells score above the label "
                f"oracle** ({n['mesr_wins'] / n['eligible_pairs']:.0%}), of "
                f"{n['total_pairs']} (recording, method) pairs")
            out(f"  - of those wins: **{n['strict_reversals']} strict reversals** "
                f"(strictly less signal AND more noise retained at the identical count), "
                f"{n['ties']} ties, {n['impossible']} impossible")
            out(f"  - all-retention view: {s['all_matched']['mesr_wins']}/"
                f"{s['all_matched']['cells']} matched cells, "
                f"{s['all_matched']['strict_reversals']} strict")
            # Every label rate is quoted on the events MESR scored, because `esr.mesr`
            # drops the incomplete tail; the full-mask tally is printed beside it so the
            # two domains can be compared rather than conflated.
            for cohort, label in (("at_native", "at native"),
                                  ("all_matched", "over all matched cells")):
                frac = s[cohort].get("scored_fraction")
                if frac:
                    out(f"  - scored fraction {label} (share of the retained output MESR "
                        f"reads): min {frac['min']:.3f}, median {frac['median']:.3f}, "
                        f"max {frac['max']:.3f}")
            full = n.get("full_mask")
            if full:
                out(f"  - same wins classified on the FULL retained mask: "
                    f"{full['strict_reversals']} strict, {full['ties']} ties, "
                    f"{full['impossible']} impossible "
                    f"(identical to the scored domain: "
                    f"{full['strict_reversals'] == n['strict_reversals']})")
            w = s.get("largest_signal_deficit")
            if w:
                out(f"  - largest deficit: {w['method']} on {w['recording']} at r={w['r']}: "
                    f"signal {w['tp_scored'] / w['n_signal']:.3f} vs oracle "
                    f"{w['oracle_tp_scored'] / w['n_signal']:.3f}, noise "
                    f"{w['fp_scored'] / w['n_noise']:.3f} vs "
                    f"{w['oracle_fp_scored'] / w['n_noise']:.3f}, MESR {w['mesr']:.4f} vs "
                    f"{w['oracle_mesr']:.4f} at the same {w['kept']} retained, "
                    f"{w['scored']} scored (rates on the scored events; on the full mask "
                    f"{w['signal_retention']:.3f} vs {w['oracle_signal_retention']:.3f})")
            out(f"  - per method (wins/cells at native): " + ", ".join(
                f"{m} {v['mesr_wins']}/{v['cells']}" for m, v in s["by_method"].items()))

        violations = entry["oracle_violations"]
        if violations.get("comparable_cells"):
            out(f"- superseded, unmatched (each side at its own argmax): "
                f"oracle violations {violations['cells_beating_oracle']}/"
                f"{violations['comparable_cells']} "
                f"({100 * violations['share_beating_oracle']:.0f}%) "
                f"max margin {violations['max_margin_over_oracle']:+.4f} "
                f"by {violations['by_method']}")
            out(f"  - **{violations['violations_at_native_operating_point']} of "
                f"{violations['cells_beating_oracle']} "
                f"({violations['share_of_violations_at_native']:.0%}) occur within one grid "
                f"step of the method's own native operating point** -- not an artifact of "
                f"sweeping a binary filter")
        intervals = entry.get("delta_over_raw_at_native_ci")
        if intervals:
            ordered = sorted(intervals.items(), key=lambda kv: -kv[1]["mean"])
            out("- delta-over-raw at native, ALL cells snapped to the nearest grid point "
                "(not what the paper prints), 95% CI over recordings: "
                + ", ".join(f"{m} {v['mean']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]"
                            for m, v in ordered))
        elig = entry.get("native_eligibility")
        if elig and elig["ineligible"]:
            out(f"- **native eligibility: {elig['ineligible']} of {elig['cells']} cells "
                f"({elig['share_ineligible']:.1%}) cannot be scored at the method's own "
                f"operating point**, all of them because the native retention is below the "
                f"recording's measurable floor "
                f"({elig['ineligible_because_native_is_below_the_floor']} of "
                f"{elig['ineligible']})")
            agg = elig.get("delta_vs_native_retention_over_eligible_rows", {})
            if "spearman" in agg:
                out(f"  - **rank correlation between delta-over-raw and native retention "
                    f"over the {agg['methods']} eligible classical rows: spearman "
                    f"{agg['spearman']:+.2f}, kendall {agg['kendall']:+.2f}** -- positive "
                    f"means the rows that keep MORE events score higher; on emlb this is "
                    f"the value S VI-D item (iii) quotes")
            for m, v in sorted(elig["by_method"].items(),
                               key=lambda kv: -kv[1]["ineligible"]):
                if not v["ineligible"]:
                    continue
                out(f"  - {m}: {v['ineligible']} ineligible cells averaging "
                    f"{v['ineligible_mean_delta']:+.4f} against "
                    f"{v.get('eligible_mean_delta', float('nan')):+.4f} on its "
                    f"{v['eligible']} eligible ones; mean native r "
                    f"{v['mean_native_r_all_cells']:.4f} but mean r actually scored at "
                    f"{v['mean_scored_at_r_all_cells']:.4f}")
        floor = entry.get("delta_at_common_floor")
        if floor:
            # S VII reads the blank-sample response here: one retention, every recording,
            # nothing selected, and the nonselective controls at the same retained count.
            ordered = sorted(floor["by_method"].items(), key=lambda kv: -kv[1]["mean"])
            out(f"- **delta-over-raw at the common-support floor r = {floor['retention']:.2f}"
                f"** (one r, every recording, nothing selected; Pure_BA's row is where "
                f"S VII's blank-sample numbers come from): "
                + ", ".join(f"{m} {v['mean']:+.4f} [{v['lo']:+.3f},{v['hi']:+.3f}] "
                            f"(n={v['n']})" for m, v in ordered))
        eligible = entry.get("delta_over_raw_at_native_eligible")
        if eligible:
            # This is the cohort tab:emlb prints. A cell counts only where the grid can
            # stand in for the filter's own operating point; the scene-clustered intervals
            # the table shows are in results/paired_contrasts.json under "marginals".
            ordered = sorted(eligible.items(), key=lambda kv: -kv[1]["mean"])
            out("- **delta-over-raw at native, ELIGIBLE cells only -- the cohort tab:emlb "
                "prints**: " + ", ".join(f"{m} {v['mean']:+.4f} (n={v['n']})"
                                         for m, v in ordered))
        reference = entry.get("edformer_reference")
        if reference:
            out(f"- EDformer: mean delta-over-raw "
                f"**{reference['mean_delta_over_raw']:.4f}**, "
                f"retention {reference['retention_range'][0]:.4f}-"
                f"{reference['retention_range'][1]:.4f}, "
                f"sweepable={reference['sweepable']}")
            cells = reference.get("cells") or []
            if cells:
                mean_r = sum(c["mean_retention"] for c in cells) / len(cells)
                out(f"  - mean native retention over those {len(cells)} cells: "
                    f"{mean_r:.4f}={mean_r:.2f} (the Table's fourth column)")
            interval = reference.get("delta_over_raw_ci")
            if interval:
                out(f"  - EDformer CI [{interval['lo']:+.4f},{interval['hi']:+.4f}] "
                    f"({interval['resampling_unit']}; coarser than the classical rows)")
            error = reference.get("reproduction_error")
            if error:
                # The paper quotes the SPAN, not the bounds, in the abstract, S5.1 and
                # Appendix A. Printing only the bounds left the quoted number derived at
                # writing time rather than generated -- which the S1 traceability claim
                # says never happens.
                out(f"  - reproduction vs published table: mean {error['mean']:+.4f}, "
                    f"per-cell range {error['min']:+.4f} to {error['max']:+.4f} "
                    f"over {error['n_cells']} cells -- quote the range, not the mean")
                out(f"  - **per-cell error span {error['max'] - error['min']:.4f}** "
                    f"(the figure quoted in the abstract, S5.1 and Appendix A; "
                    f"{error['max'] - error['min']:.4f} > 0.0092, the gap the table claims)")
            sets = reference.get("recording_sets")
            if sets:
                # Matched on recordings is not matched on the cap. Delta-over-Raw moves
                # with the cap (S5.3), so the EDformer row stays descriptive either way.
                verdict = ("matched on recordings but NOT on the event cap"
                           if sets["matched_on_recordings"] else "NOT matched")
                out(f"  - recording sets: EDformer "
                    f"{sets['edformer_recordings']} recordings uncapped; classical rows "
                    f"{sets['classical_recordings']} over "
                    f"{sets['classical_scene_clusters']} (scene, lighting) clusters "
                    f"capped at {sets['classical_cap']:,} -- {verdict}")


def _scale(out: Callable[[str], None]) -> None:
    """Every headline ratio under both denominators.

    The paper quotes its effects against 0.0092, the published EDformer-EDmamba gap, which
    S6 shows is not reproducible from released code. A ratio whose denominator cannot be
    measured invites the obvious objection, so each effect is also quoted against the
    reproduction floor of S5.1 -- the per-cell span of the best available reproduction of
    the table that reports 0.0092, which is a quantity we measured ourselves.
    """

    ranks = _load("rank_analysis.json")
    deps = _load("metric_dependencies.json")
    published_gap = 0.0092
    error = ranks["emlb"]["edformer_reference"]["reproduction_error"]
    floor = error["max"] - error["min"]

    effects = [
        ("blank sample, best filter on Pure_BA",
         ranks["pure_ba"]["delta_over_raw_at_native"]["red"]),
        ("random_null on DVSD22, common r",
         ranks["dvsd22"]["null_gain_fixed_retention"]["random_null"]["best_fixed"]["mean"]),
        ("slice-size convention",
         deps["slice_dependence"]["spread_over_recordings"]["mean"]),
        ("operating point, fixed denoiser",
         deps["retention_dependence"]["spread_over_recordings"]["mean"]),
        ("random_null on Pure_BA, common r",
         ranks["pure_ba"]["null_gain_fixed_retention"]["random_null"]["best_fixed"]["mean"]),
    ]

    out("\n## S5.2 the unit of scale, under both denominators  "
        "(results/rank_analysis.json, results/metric_dependencies.json)")
    out(f"- published gap (EDformer vs EDmamba on E-MLB): {published_gap:.4f} "
        "-- a claimed advance, not a measured one (S6)")
    out(f"- **reproduction floor: {floor:.4f}** -- the per-cell span of the best available "
        "reproduction")
    out("  of the table reporting that gap, and the smallest difference that reproduction "
        "can resolve")
    survive = 0
    for label, value in effects:
        ratio_gap = value / published_gap
        ratio_floor = value / floor
        survive += ratio_floor >= 2
        out(f"- {label}: {value:+.4f} = **{ratio_gap:.0f}x** the published gap, "
            f"**{ratio_floor:.1f}x** the reproduction floor")
    out(f"- **{survive} of {len(effects)} effects stay above 2x under the conservative "
        f"denominator**; the exception is")
    out("  the Pure_BA null, whose specificity gain does not survive it (the corpus carries "
        "the blank-sample")
    out("  result instead). Both denominators are E-MLB quantities, so neither is "
        "like-for-like across corpora (S3.3).")


def _blank_signature(out: Callable[[str], None]) -> None:
    """What can and cannot be measured about Pure_BA being a blank.

    S6 calls Pure_BA signal-free, and the release does not document how that was achieved.
    An uncertified blank does not support a blank-sample claim, so this block reports the
    measured signature that stands in for the missing documentation -- and, equally, which
    of the four regime statistics this corpus is too short to support.
    """

    regimes = _load("noise_regimes.json")
    profiles = [p for p in regimes["profiles"] if p["corpus"] == "Pure_BA"]
    shares = sorted(p["hot_pixel_share"] for p in profiles)

    def nan_of(stat: str) -> int:
        """Recordings where the statistic is undefined (NaN != NaN)."""
        return sum(1 for profile in profiles if profile[stat] != profile[stat])

    out("\n## S6.1 the Pure_BA blank: measured signature  (results/noise_regimes.json)")
    # The paper quotes these as percentages, so the digest emits that form too: a value
    # converted at writing time is a value with no artifact behind it.
    out(f"- hot-pixel share over {len(profiles)} profiled recordings: "
        f"**median {np.median(shares):.4f} = {100 * np.median(shares):.1f}%**, range "
        f"{100 * shares[0]:.1f}--{100 * shares[-1]:.1f}%")
    for corpus in ("E-MLB", "DND21", "DVSCLEAN", "DVSD22"):
        block = regimes["per_corpus"].get(corpus)
        if block:
            ratio = np.median(shares) / block["hot_pixel_share"]
            out(f"  - vs {corpus}: {block['hot_pixel_share']:.4f} = "
                f"{100 * block['hot_pixel_share']:.2f}% "
                f"-> Pure_BA is {ratio:.0f}x higher")
    out("- **not measurable on this corpus** (recordings too short to yield 1 ms windows):")
    out(f"  - spatial autocorrelation: {nan_of('spatial_autocorr')}/{len(profiles)} NaN")
    out(f"  - local-global coupling:   {nan_of('local_global_coupling')}/{len(profiles)} NaN")
    real_n = {stat: regimes["verdict"][stat]["real_n"]
              for stat in ("hot_pixel_share", "event_rate_per_px",
                           "spatial_autocorr", "local_global_coupling")}
    out(f"  so S7's real-group n is {real_n['hot_pixel_share']} for hot-pixel share and "
        f"{real_n['event_rate_per_px']} for event rate, but "
        f"{real_n['spatial_autocorr']} and {real_n['local_global_coupling']} for the two")
    out("  window statistics; the blank cannot be certified from spatial coherence.")


def _architectures(out: Callable[[str], None]) -> None:
    """S8 under three classifier inductive biases.

    A null downstream result is only worth reporting if it is not a property of one
    architecture, so the identical grid is re-run under a 3D CNN and an MLP. The answer is
    not a flat null: MESR predicts accuracy for the one architecture that also prefers
    concentrated input, and fails for the two that do not. The row that explains it is
    retention -> accuracy, whose SIGN separates the three.
    """

    from scipy.stats import spearmanr

    out("\n## S8.1 the same grid under three architectures  "
        "(results/downstream_gesture{,_cnn3d,_mlp}.json)")
    for suffix in ("", "_cnn3d", "_mlp"):
        payload = _load(f"downstream_gesture{suffix}.json")
        arch = payload["protocol"]["arch"]
        paper = set(payload["protocol"]["paper_methods"])
        rows = [c for c in payload["conditions"]
                if c["method"] in paper and np.isfinite(c["mesr"])]
        stats = payload["spearman_mesr_vs_accuracy"]
        unfiltered = next(c["accuracy"] for c in rows if c["retention"] == 1.0)
        ret_acc = spearmanr([c["retention"] for c in rows],
                            [c["accuracy"] for c in rows])
        best = max(rows, key=lambda c: c["accuracy"])
        worst = min(rows, key=lambda c: c["accuracy"])
        out(f"- **{arch}** ({payload['wall_seconds'] / 60:.0f} min, "
            f"unfiltered accuracy {unfiltered:.4f}):")
        out(f"  - MESR vs accuracy, all {stats['all']['n']}: "
            f"**rho={stats['all']['rho']:+.3f} p={stats['all']['p']:.3f}**")
        out(f"  - partial | retention: rho={stats['partial_given_retention']['rho']:+.3f} "
            f"p={stats['partial_given_retention']['p']:.3f}")
        out(f"  - within-retention Fisher: mean rho="
            f"{stats['within_retention']['mean_rho']:+.3f} "
            f"p={stats['within_retention']['fisher_p']:.3f}")
        out(f"  - **retention -> accuracy: rho={ret_acc.statistic:+.3f} "
            f"p={ret_acc.pvalue:.3f}** <- the sign that separates the three")
        out(f"  - across-seed accuracy sd: median "
            f"{payload['accuracy_sd_across_seeds']['median']:.4f}, "
            f"max {payload['accuracy_sd_across_seeds']['max']:.4f}")
        out(f"  - best {best['method']}@r={best['retention']:.1f} {best['accuracy']:.4f}; "
            f"worst {worst['method']}@r={worst['retention']:.1f} {worst['accuracy']:.4f}")
    out("- retention -> MESR is -0.163 in all three (MESR is computed on the same filtered")
    out("  test streams whatever the classifier), so the differing MESR-accuracy "
        "correlations are")
    out("  differing accuracy-retention relationships and nothing else.")


def _cap_sensitivity(out: Callable[[str], None]) -> None:
    """The measurement that replaced the unmeasured cap-invariance assertion."""

    path = RESULTS / "cap_sensitivity.json"
    if not path.is_file():
        return
    r = json.loads(path.read_text())
    s = r["spread"]
    caps = ", ".join(f"{c/1e6:g}M" for c in r["caps"])
    out(f"\n## S5.3 event-cap sensitivity of Delta-over-Raw  (results/cap_sensitivity.json)")
    out(f"- E-MLB, {r['cells_scored']} scored cells over caps {caps} "
        f"(all 384 recordings: 48 scenes x 4 ND levels x 2 lighting conditions)")
    out(f"- **Delta-over-Raw is NOT cap-invariant.** Spread across caps: median "
        f"{s['median']:.4f} ({s['median']/r['unit_of_scale']:.1f}x the 0.0092 unit of "
        f"scale), max {s['max']:.4f} ({s['max_in_units_of_scale']:.0f}x); "
        f"**{s['share_above_unit_of_scale']:.0%} of cells exceed the unit of scale**")
    er = r["evaluable_range"]
    out(f"- evaluable range identical across these caps: {er['identical_range']}/"
        f"{er['cells']} cells -- the cap moves the measurable floor as well as the value")
    out("- with retention held FIXED (floor effect removed):")
    for label, v in r["at_fixed_retention"].items():
        above = round(v["share_above_unit_of_scale"] * v["cells"])
        out(f"  - {label}: median {v['median']:.5f}, max {v['max']:.4f}, "
            f"{above}/{v['cells']} cells above the unit of scale "
            f"({v['share_above_unit_of_scale']:.1%})")
    out("- per-method mean by cap, over the cells the cap can bind: " + "; ".join(
        f"{m} " + "/".join(f"{d:+.4f}" for d in v["mean_by_cap"].values())
        for m, v in r["per_method"].items() if m != "raw"))
    ov = r["means_over_all_recordings"]
    out(f"- per-method mean by cap over ALL {ov['recordings']} recordings -- the *unrestricted* "
        f"native cohort, which tab:emlb no longer prints (it prints the eligible cells only, "
        f"see the E-MLB block above); the 1M column here reproduces the unrestricted means: "
        + "; ".join(
            f"{m} " + "/".join(f"{d:+.4f}" for d in v["mean_by_cap"].values())
            + f" (moves {v['spread_across_caps']:.4f}, {v['spread_in_units_of_scale']:.1f}x)"
            for m, v in sorted(ov["by_method"].items(),
                               key=lambda kv: -kv[1]["spread_across_caps"]) if m != "raw"))

    pair = RESULTS / "cap_sensitivity" / "pair_1M_vs_2M.json"
    if pair.is_file():
        p2 = json.loads(pair.read_text())
        s2, e2 = p2["spread"], p2["evaluable_range"]
        out(f"- **the clean pair (1M vs 2M): evaluable range identical for "
            f"{e2['identical_range']}/{e2['cells']} cells, so this is the cap alone** -- "
            f"median {s2['median']:.4f}, max {s2['max']:.4f} "
            f"({s2['max_in_units_of_scale']:.0f}x), "
            f"{round(s2['share_above_unit_of_scale'] * p2['cells_scored'])}/"
            f"{p2['cells_scored']} bindable cells above the unit of scale "
            f"({s2['share_above_unit_of_scale']:.0%})")
        worst = max(v["mean_spread_across_caps"] for m, v in p2["per_method"].items()
                    if m != "raw")
        ov2 = p2["means_over_all_recordings"]
        moves = sorted(((v["spread_across_caps"], m) for m, v in ov2["by_method"].items()
                        if m != "raw"), reverse=True)
        out(f"  - on the printed cohort (all {ov2['recordings']} recordings) the corpus "
            f"means move too, but only one passes the gap: {moves[0][1]} by "
            f"{moves[0][0]:.4f} ({moves[0][0] / p2['unit_of_scale']:.1f}x); the other five "
            f"move at most {moves[1][0]:.4f}. No classical row changes position")
    out("- 2M is still not uncapped: 223 of the 384 recordings remain truncated at it (354 "
        "at 0.5M, 308 at 1M), so the residual against an uncapped run is not bounded by "
        "these data")


def _esr_properties(out: Callable[[str], None]) -> None:
    """S4-A's mechanism. The first two lines replace claims the paper made and could not
    support: that ESR rises with concentration, and that hot-pixel share fixes the null's
    sign. Both are now computed against the released `esr.esr`."""

    payload = _load("esr_properties.json")
    out("\n## S4-A properties of ESR (`results/esr_properties.json`)\n")

    m = payload["monotonicity"]
    c, s = m["concentrated"], m["spread"]
    fmt = lambda v: "ESR(" + ", ".join(f"{n:,}" for n in v["counts"]) + ")"
    out(f"- **ESR is NOT monotone in concentration**: {fmt(c)} = {c['esr']:.5f} < "
        f"{fmt(s)} = {s['esr']:.5f}, and the first majorises the second "
        f"({m['concentrated_majorises_spread']})")
    out(f"  - ntss {c['ntss']:.4f} -> {s['ntss']:.4f} (Schur-convex, favours the "
        f"concentrated vector); ell_n {c['ell_n']:.4f} -> {s['ell_n']:.4f} "
        f"(Schur-concave, favours the spread one)")

    pi = payload["permutation_invariance"]
    out(f"- **ESR is invariant to a bijection of the pixel grid**: max |dESR| over "
        f"{pi['trials']} random slices = **{pi['max_abs_difference']:.3g}**")

    st = payload["stationarity"]
    out(f"- **the null's sign needs a nonstationary scene**, not hot-pixel share: "
        f"{st['length']:,}-event synthetic streams, {len(st['seeds'])} seeds, "
        f"{st['n_hot_pixels']} hot pixels, scene span {st['scene_span_pixels']} px")
    for cell in st["cells"]:
        moves = "  ".join(f"{k} {v:+.4f}"
                          for k, v in cell["delta_over_unthinned"].items() if v)
        out(f"  - scene {cell['scene']}, hot share {cell['hot_share']:.0%}: {moves} "
            f"-> **{cell['sign']}**")


def _paired_contrasts(out: Callable[[str], None]) -> None:
    """S5-C's ranking claims. These replaced marginal-interval overlap reasoning, which is
    invalid in both directions the table used it and which had discarded a real ordering."""

    payload = _load("paired_contrasts.json")
    out("\n## S5-C paired contrasts on E-MLB (`results/paired_contrasts.json`)\n")
    out(f"- ranking by mean Delta-over-Raw: {' > '.join(payload['ranked_by_mean'])}")
    out(f"- **{payload['pairs_resolved']}/{payload['pairs_total']} pairs separate** under a "
        f"paired difference over {payload['n_recordings']} recordings, bootstrapped over "
        f"{payload['n_scenes']} scene clusters")
    for pair in payload["pairs"]:
        flag = "separates" if pair["excludes_zero"] else "**NOT resolved**"
        out(f"  - {pair['a']} - {pair['b']}: {pair['mean_difference']:+.4f} "
            f"[{pair['lo']:+.4f}, {pair['hi']:+.4f}], holds on "
            f"{pair['share_a_above_b']:.0%} of recordings -> {flag}")
    # Dependence never made the correction unavailable -- Bonferroni and Holm hold under
    # arbitrary dependence -- so the corrected count is reported beside the pointwise one.
    familywise = payload.get("bonferroni")
    if familywise:
        dropped = [f"{p['a']}-{p['b']}" for p, q in zip(payload["pairs"], familywise["pairs"])
                   if p["excludes_zero"] and not q["excludes_zero"]]
        note = f" (lost: {', '.join(dropped)})" if dropped else ""
        out(f"- **{familywise['pairs_resolved']}/{payload['pairs_total']} survive a "
            f"Bonferroni level alpha={familywise['alpha']:.5f}**{note}")
    # The intervals tab:emlb PRINTS. They live here, not beside the means in
    # `rank_analysis.json`, because they resample scene clusters rather than recordings --
    # which is what the caption promises. `rank_analysis.delta_over_raw_at_native_eligible`
    # carries a recording-level interval for the identical row, and it is narrower (RED
    # [+0.3739, +0.5429] against the printed [+0.3500, +0.5793]). Printing only one of the
    # two left a reader who followed the mean to its file finding an interval that
    # disagrees with the table.
    out("- **marginals -- the mean and interval `tab:emlb` prints**, eligible cells only, "
        "95% percentile bootstrap over scene clusters (NOT the narrower recording-level "
        "interval stored beside the means in `rank_analysis.json`):")
    for method in payload["ranked_by_mean"]:
        m = payload["marginals"][method]
        out(f"  - {method}: {m['mean']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}] "
            f"(n={m['n_recordings']} recordings over {m['n_scenes']} scene clusters), "
            f"excludes zero={m['excludes_zero']}")


def _common_support(out: Callable[[str], None]) -> None:
    """S4 item 1 and S5-B. AUC_r is a span-normalised trapezoid, not a mean, and it was
    aggregated across recordings with different evaluable ranges. Both are repaired here;
    the own-range column reproduces the published table, which is what licenses the rest."""

    payload = _load("common_support.json")
    out("\n## S4/S5-B common support for AUC_r (`results/common_support.json`)\n")
    for dataset, r in payload.items():
        er = r["evaluable_range_per_recording"]
        v = r["vs_native"]
        out(f"- **{dataset}** ({r['n_recordings']} recordings, ranges "
            f"{'RAGGED' if er['ragged'] else 'uniform'}): common grid "
            f"[{er['common'][0]:.2f}, {er['common'][1]:.2f}], common cohort "
            f"{r['common_cohort']['n_recordings']} recordings")
        for name in ("own_range", "common_grid", "common_cohort"):
            a = v[name]
            out(f"  - AUC_r ({name}) vs native: rho={a['spearman']:+.3f} "
                f"tau={a['kendall']:+.3f} rankings differ={a['differ']}")
        cov = r["coverage"]["recordings_at_r"]
        lo = min(cov, key=float)
        out(f"  - n(r): {cov[lo]} recordings at r={lo}, {cov['1.00']} at r=1.00")


def _nd_levels(out: Callable[[str], None]) -> None:
    """S VIII's replacement for the ND mechanism, and every figure the appendix prints.

    Added because these were the newest numbers in the paper and the only ones with no
    digest line -- the exact defect the data-availability statement promises against.
    """

    payload = _load("nd_levels.json")
    nd, design, paired = payload["nd_levels"], None, None
    design, paired = nd["design"], nd["paired"]

    out("\n## S7.1 E-MLB neutral-density levels  (results/nd_levels.json)")
    out(f"- design: {design['scenes']} scenes x {len(design['levels'])} levels x 1 "
        f"repetition = {design['recordings']} recordings, fully crossed "
        f"(so the level contrast is PAIRED)")
    mapping = payload["level_mapping"]
    out("- level mapping (release directory -> the condition the source paper names): "
        + ", ".join(f"{k}={v['paper_name']} (T={v['transmittance']:.4f}"
                    + (", unattenuated)" if not v["attenuated"] else ")")
                    for k, v in mapping.items()))
    out("- busiest-pixel share per level: " + ", ".join(
        f"{level} median {nd['per_level'][level]['median']:.5f} "
        f"(mean {nd['per_level'][level]['mean']:.5f})" for level in design["levels"]))
    out(f"- **paired: Friedman chi2={paired['friedman_chi2']:.2f} "
        f"p={paired['friedman_p']:.2e}; Wilcoxon unattenuated vs most attenuated "
        f"p={paired['wilcoxon_first_vs_last_p']:.2e}; rising in "
        f"{paired['scenes_rising_first_to_last']}/{design['scenes']} scenes** "
        f"-- the unattenuated level is the LOWEST, so attenuation does not explain "
        f"E-MLB's low share")
    unpaired = nd["unpaired_secondary"]
    out(f"  - unpaired on the same values, kept as a secondary line only: "
        f"p={unpaired['p_value']:.2e} rb={unpaired['rank_biserial']:+.3f}")

    contrast = payload["unattenuated_vs_synthetic"]
    out(f"- unattenuated E-MLB vs the synthetic corpora: p={contrast['p_value']:.3f} "
        f"rb={contrast['rank_biserial']:+.3f} separates={contrast['separates']}, "
        f"{contrast['n_first']} against {contrast['n_second']} recordings from "
        f"{contrast['synthetic_base_scenes']} base scenes; "
        f"{contrast['unattenuated_share_inside_synthetic_range']:.1%} of the unattenuated "
        f"recordings lie inside the synthetic range "
        f"-- NON-SEPARATION under the registered bar, not equivalence")

    out("\n### S7.1 which split separates (S VIII's exception is corpus identity)")
    for key in ("provenance_all_corpora", "provenance_without_emlb", "corpus_identity"):
        row = payload["groupings"][key]
        out(f"- {key}: {row['groups'][0]} (n={row['n_first']}) vs {row['groups'][1]} "
            f"(n={row['n_second']}): p={row['p_value']:.2e} "
            f"rb={row['rank_biserial']:+.3f} separates={row['separates']}")
    rate = payload["rate_association"]
    out(f"- event rate vs busiest-pixel share, all recordings: spearman "
        f"{rate['overall_spearman']:+.3f} p={rate['overall_p']:.2e} n={rate['n']}; "
        "within E-MLB's matched levels the rate medians run "
        + ", ".join(f"{v:.2f}" for v in rate["emlb_rate_median_per_level"].values())
        + f" (Friedman p={rate['emlb_rate_friedman_p']:.1e})")
    out(f"  - post-hoc and not a mechanism: {payload['groupings']['basis'].split('.')[0]}.")


def _regime_and_audit(out: Callable[[str], None]) -> None:
    regimes = _load("noise_regimes.json")
    out("\n## S7 regime separation  (results/noise_regimes.json)")
    bar = regimes.get("separation_bar")
    if bar:
        out(f"- pre-registered bar: p<{bar['p_value']} AND "
            f"|rank-biserial|>{bar['abs_rank_biserial']}")
    for name, row in regimes["verdict"].items():
        out(f"- {name}: synth {row['synthetic_mean']:.4f} (n={row['synthetic_n']}) "
            f"real {row['real_mean']:.4f} (n={row['real_n']}) p={row['p_value']:.2e} "
            f"rb={row['rank_biserial']:+.3f} separates={row['separates']}")

    # Appendix C quotes per-corpus means to explain the asymmetry, so they belong here.
    # Two of them (DVSD22, Pure_BA) were quoted in the paper with no digest line at all.
    out("\n### S7 per-corpus means")
    for corpus, row in sorted(regimes["per_corpus"].items()):
        out(f"- {corpus} (n={row['n']}, synthetic={row['synthetic']}): "
            + ", ".join(f"{stat} {row[stat]:.4f}"
                        for stat in ("spatial_autocorr", "local_global_coupling",
                                     "hot_pixel_share", "event_rate_per_px")))

    sensitivity = regimes.get("leave_one_corpus_out")
    if sensitivity:
        out("\n### S7 sensitivity: each corpus withheld in turn (every cell of "
            "Table tab:loco)")
        clears = []
        for dropped, rows in sensitivity.items():
            best_name, best = max(rows.items(),
                                  key=lambda kv: abs(kv[1]["rank_biserial"]))
            out(f"- without {dropped}: strongest {best_name} p={best['p_value']:.2e} "
                f"rb={best['rank_biserial']:+.3f} separates={best['separates']}")
            for stat, row in rows.items():
                mark = " **clears the bar**" if row["separates"] else ""
                # Sizes are quoted in Table tab:loco: withholding a synthetic corpus halves
                # the synthetic group, and withholding E-MLB cuts the real group to 53.
                out(f"  - {stat}: p={row['p_value']:.2e} "
                    f"rb={row['rank_biserial']:+.3f} "
                    f"n={row['synthetic_n']}/{row['real_n']}{mark}")
                if row["separates"]:
                    clears.append(f"{stat} without {dropped}")
        out(f"- **cells clearing the pre-registered bar: {len(clears)}** "
            + (", ".join(clears) if clears else "none"))

    checks = _load("reproducibility_audit.json")["checks"]
    out(f"\n## S6 audit: {sum(c['verified'] for c in checks)}/{len(checks)} verified  "
        f"(results/reproducibility_audit.json)")
    for check in checks:
        evidence = check["evidence"]
        where = (f"{evidence['file']}:{evidence['line']}"
                 if isinstance(evidence, dict) and "file" in evidence else "-")
        out(f"- [{'OK' if check['verified'] else 'ERR'}] {check['paper']}: "
            f"{check['claim']}  ({where})")


def _downstream(out: Callable[[str], None]) -> None:
    payload = _load("downstream_gesture.json")
    protocol, conditions = payload["protocol"], payload["conditions"]
    missing = [k for k in ("seeds", "retentions") if k not in protocol]
    if missing or "coverage" not in payload:
        raise SystemExit(
            "results/downstream_gesture.json predates the multi-seed protocol "
            f"(missing {missing + ([] if 'coverage' in payload else ['coverage'])}). "
            "Re-run `python -m dataset_assessment.downstream_gesture` before regenerating "
            "the digest; emitting the old fields here would put unlabelled single-seed "
            "numbers into the paper.")
    out("\n## S8 downstream validity  (results/downstream_gesture.json)")
    out(f"- {protocol['train_samples']} train / {protocol['test_samples']} test, "
        f"seeds {protocol['seeds']}, {protocol['epochs']} epochs, "
        f"{protocol['max_events_per_sample']} events/sample, "
        f"retentions {protocol['retentions']}, sensor {protocol['sensor']}")

    # The two quantities this experiment correlates are read off different event sets: the
    # classifier gets every retained event, `esr.mesr` only complete slices. Emitted per
    # retention because the share is non-monotone in r, which is what makes it survive the
    # partial correlation that removes retention.
    shares = protocol.get("scored_share")
    if shares:
        out("- **MESR and the classifier do not read the same events**: the classifier gets "
            "every retained event, MESR only complete slices")
        for row in shares:
            out(f"  - r={row['retention']:.1f}: {row['classified']:,} classified, "
                f"{row['complete_slices']} complete slice(s), "
                f"{row['scored_by_mesr']:,} scored by MESR "
                f"= **{100 * row['scored_share']:.1f}%**")
        worst = min(shares, key=lambda r: r["scored_share"])
        out(f"  - **not monotone in r**: minimum {100 * worst['scored_share']:.1f}% at "
            f"r={worst['retention']:.1f}, back up at r=0.8 where a second slice fits, so "
            f"partialling retention out does not remove it")

    coverage = payload.get("coverage")
    if coverage:
        for split in ("train", "test"):
            row = coverage[split]
            worst = min(row["per_class"].items(),
                        key=lambda kv: kv[1]["kept_fraction"])
            out(f"- coverage {split}: {row['gestures_kept']}/{row['gestures_offered']} "
                f"({100 * row['kept_fraction']:.1f}%) of the official split; "
                f"worst class {worst[0]} at {100 * worst[1]['kept_fraction']:.0f}%")

    stats = payload["spearman_mesr_vs_accuracy"]
    out(f"- **spearman(MESR, accuracy) all: rho={stats['all']['rho']:+.3f} "
        f"p={stats['all']['p']:.3f} n={stats['all']['n']}**")
    out(f"- excluding r=1: rho={stats['excluding_r1']['rho']:+.3f} "
        f"p={stats['excluding_r1']['p']:.3f} n={stats['excluding_r1']['n']}")
    # The pooled row counts one unfiltered stream once per method. Both are quoted because
    # the deduplicated value is larger, and a reader must be able to see that.
    di = stats["distinct_inputs"]
    out(f"- **distinct inputs (r=1 endpoint kept once, not once per method): "
        f"rho={di['rho']:+.3f} p={di['p']:.3f} n={di['n']}** "
        f"({di['duplicate_cells_collapsed']} duplicate cells collapsed)")

    # What the design could have detected. Without it a failure to reject reads as a
    # well-powered null, which at n=40 it is not.
    sensitivity = stats.get("sensitivity")
    if sensitivity:
        out(f"- **sensitivity at n={sensitivity['n']}: significant only if |rho| > "
            f"{sensitivity['min_detectable_rho']:.3f}**; power "
            f"{sensitivity['power']['rho_03']:.2f} at rho=0.3, "
            f"{sensitivity['power']['rho_05']:.2f} at rho=0.5")

    partial = stats.get("partial_given_retention")
    if partial:
        out(f"- **partial rho(MESR, accuracy | retention) = {partial['rho']:+.3f} "
            f"p={partial['p']:.3f} n={partial['n']}**  <- primary within-condition test")
    distinct = stats.get("partial_given_retention_distinct")
    if distinct:
        # Same test on the distinct-input sample: the r = 1 endpoint enters the pooled grid
        # once per method, so the pooled value is computed over as many copies of one
        # deterministic condition as there are methods. Section 9 prints both.
        out(f"- **the same, distinct inputs = {distinct['rho']:+.3f} "
            f"p={distinct['p']:.3f} n={distinct['n']}**  <- endpoint counted once")

    retention_stats = payload.get("spearman_retention", {})
    if retention_stats:
        out(f"- retention -> accuracy rho={retention_stats['vs_accuracy']['rho']:+.3f} "
            f"p={retention_stats['vs_accuracy']['p']:.2e}")
        out(f"- retention -> MESR     rho={retention_stats['vs_mesr']['rho']:+.3f} "
            f"p={retention_stats['vs_mesr']['p']:.3f}")

    within = stats.get("within_retention", {})
    for level, row in within.get("per_retention", {}).items():
        out(f"  - within r={level}: rho={row['rho']:+.3f} p={row['p']:.3f} n={row['n']}")
    if within:
        out(f"  - mean within-r rho={within['mean_rho']:+.3f}, "
            f"Fisher-combined p={within['fisher_p']:.3f} "
            f"(each level is underpowered on its own)")
        for seed, row in within.get("per_seed", {}).items():
            out(f"    - same combination on seed {seed} alone: "
                f"mean rho={row['mean_rho']:+.3f} Fisher p={row['fisher_p']:.3f}")

    for seed, row in stats.get("per_seed_all_conditions", {}).items():
        out(f"  - seed {seed}: rho={row['rho']:+.3f} p={row['p']:.3f} n={row['n']}")

    spread = payload.get("accuracy_sd_across_seeds")
    if spread:
        # Name the architecture. This whole function reads `downstream_gesture.json`, which
        # is the 2D CNN alone; the 3D CNN and MLP are digested by `_architectures`. An
        # unlabelled median/max here reads as if it covered all three, and the main text
        # once paired this median with the 3D CNN's max as though both were one grid.
        out(f"- accuracy sd across seeds (2D CNN): median {spread['median']:.4f} "
            f"max {spread['max']:.4f} (the scale any accuracy difference must beat; "
            f"the 3D CNN and MLP are in S8.1)")

    # Every line above is scoped to the paper's method set; the extremes must be too, or a
    # sibling project's row would be quoted as if it were in the paper's tables.
    scoped = [c for c in conditions if c["method"] in protocol["paper_methods"]]
    best_metric = max(scoped, key=lambda c: c["mesr"])
    best_accuracy = max(scoped, key=lambda c: c["accuracy"])
    worst_accuracy = min(scoped, key=lambda c: c["accuracy"])
    out(f"- highest MESR: {best_metric['method']} r={best_metric['retention']} "
        f"mesr={best_metric['mesr']:.4f} acc={best_metric['accuracy']:.4f}")
    out(f"- highest accuracy: {best_accuracy['method']} r={best_accuracy['retention']} "
        f"mesr={best_accuracy['mesr']:.4f} acc={best_accuracy['accuracy']:.4f}")
    out(f"- lowest accuracy: {worst_accuracy['method']} r={worst_accuracy['retention']} "
        f"mesr={worst_accuracy['mesr']:.4f} acc={worst_accuracy['accuracy']:.4f}")
    # The unfiltered point. Every method is the same stream at r=1, so this is both the
    # determinism check S8 describes and the accuracy S8/S9 quote when conceding how far
    # the probe sits below the task ceiling. It was absent from this digest.
    unfiltered = [c for c in scoped if abs(c["retention"] - 1.0) < 1e-9]
    if unfiltered:
        accuracies = {round(c["accuracy"], 4) for c in unfiltered}
        mesrs = {round(c["mesr"], 4) for c in unfiltered}
        out(f"- **unfiltered (r=1.0): acc={sorted(accuracies)[0]:.4f} "
            f"mesr={sorted(mesrs)[0]:.4f}** over {len(unfiltered)} conditions, "
            f"identical across all of them ({len(accuracies)} distinct accuracy, "
            f"{len(mesrs)} distinct MESR) -- the determinism check, and the figure S8 and "
            f"S9 quote against the task ceiling")

    extended = payload.get("extended_scope")
    if extended:
        ext = extended["mesr_vs_accuracy"]
        out(f"\n### S8 scope sensitivity: the same conditions with "
            f"{', '.join(protocol['methods_outside_the_paper'])} added "
            f"(n={extended['n_conditions']}, excluded from every paper table)")
        out(f"- all: rho={ext['all']['rho']:+.3f} p={ext['all']['p']:.3f}")
        out(f"- partial | retention: rho={ext['partial_given_retention']['rho']:+.3f} "
            f"p={ext['partial_given_retention']['p']:.3f}")
        out(f"- **within-retention mean rho={ext['within_retention']['mean_rho']:+.3f}, "
            f"Fisher p={ext['within_retention']['fisher_p']:.3f}** -- the paper-scope "
            f"Fisher result does not survive adding two filters")


def _paired_seeds(out: Callable[[str], None]) -> None:
    """S5-F's seed claim, formed as a paired contrast rather than two marginal spreads.

    The marginal comparison cannot answer the question it is put to: shared seeds mean a
    large swing in both conditions can leave their difference stable. These are the counts
    the paper quotes, and they do not all favour it -- the MLP is the exception.
    """

    payload = _load("downstream_paired_seeds.json")
    out("\n## S5-F paired seed contrasts (`results/downstream_paired_seeds.json`)\n")
    out(f"- scope: {payload['scope']}")
    for arch, summary in payload["architectures"].items():
        if "error" in summary:
            out(f"- {arch}: {summary['error']}")
            continue
        out(f"- **{arch}: {summary['n_resolved']}/{summary['n_pairs']} pairs whose mean "
            f"difference exceeds the across-seed spread of that difference**, "
            f"{summary['n_sign_consistent']} sign-consistent across all three seeds; "
            f"median paired sd {summary['median_paired_sd']:.4f} against marginal "
            f"{summary['median_marginal_sd']:.4f}")


def _native_oracle(out: Callable[[str], None]) -> None:
    """S5-A's reversals on the filters' own outputs, before and after busiest-pixel removal.

    The `_hotpixel` runs remove the pixels from the *input* and rebuild both sides, so the two
    rows are the same construction on two inputs, not one comparison pruned after the fact."""

    out("\n## S5-A native-output label oracle (`results/native_oracle_<corpus>[_hotpixel].json`)\n")
    for dataset in ("dnd21", "dvsclean"):
        for suffix, label in (("", "as released"), ("_hotpixel", "busiest 0.1% removed first")):
            payload = _load(f"native_oracle_{dataset}{suffix}.json")
            s, cells = payload["summary"], payload["cells"]
            wins = [c for c in cells if c["mesr_win"]]
            margins = sorted(c["mesr"] - c["oracle_mesr"] for c in wins)
            out(f"- {dataset} {label}: **{s['reversals']}/{s['cells']} evaluable cells reverse**, "
                f"strict {s['eligible_strict']}, ties {s['tie']}, impossible {s['impossible']}, "
                f"unevaluable {s['unevaluable']}; largest margin "
                f"{max(c['mesr'] - c['oracle_mesr'] for c in cells):+.4f}, median margin among "
                f"reversals {float(np.median(margins)):+.4f}")
            if payload.get("removals"):
                shares = [r["share_removed"] for r in payload["removals"]]
                out(f"  - median share of events removed {float(np.median(shares)):.2%}")
            for c in cells:
                if c["recording"] == "DND21/10hz_hotel-bar" and c["method"] == "knoise":
                    out(f"  - knoise on 10hz_hotel-bar: {c['tp_scored']} signal / "
                        f"{c['fp_scored']} noise of {c['scored']} scored, MESR "
                        f"{c['mesr']:.4f} vs oracle {c['oracle_mesr']:.4f}")


def _native_emlb(out: Callable[[str], None]) -> None:
    """S6-C's Table IV: E-MLB scored on each filter's own output beside a matched control."""

    from scipy.stats import spearmanr

    payload = _load("native_emlb.json")
    summary = payload["summary"]
    out("\n## S6-C E-MLB on native outputs (`results/native_emlb.json`)\n")
    out(f"- {summary['recordings']} recordings, {summary['cells']} evaluable cells, "
        f"{summary['unevaluable']} unevaluable (native output under one slice)")
    rows = []
    for method, row in sorted(summary["by_method"].items(),
                              key=lambda kv: -kv[1].get("delta_over_raw", {}).get("mean", -9)):
        if "delta_over_raw" not in row:
            out(f"  - {method}: no evaluable cell ({row['unevaluable']} unevaluable)")
            continue
        d, n, s = row["delta_over_raw"], row["null_delta_over_raw"], row["filter_minus_null"]
        rows.append((row["native_retention_mean"], d["mean"]))
        extra = ""
        if "unevaluable_native_retention_mean" in row:
            extra = f", unevaluable native r mean {row['unevaluable_native_retention_mean']:.2f}"
        out(f"  - {method}: n={row['evaluable']} (unevaluable {row['unevaluable']}{extra}), "
            f"native r {row['native_retention_mean']:.2f}, Delta {d['mean']:+.4f} "
            f"[{d['lo']:+.3f}, {d['hi']:+.3f}], matched null Delta {n['mean']:+.4f} "
            f"[{n['lo']:+.3f}, {n['hi']:+.3f}], filter - null {s['mean']:+.4f} "
            f"[{s['lo']:+.3f}, {s['hi']:+.3f}] above null on "
            f"{row['recordings_filter_above_null']}/{row['evaluable']}")
    rho, p = spearmanr([r for r, _ in rows], [d for _, d in rows])
    out(f"- rows: Spearman(native r, Delta) = {rho:+.2f}, p = {p:.2f} over {len(rows)} rows")

    paired = _load("paired_contrasts_native.json")
    out(f"- paired, native outputs: **{paired['pairs_resolved']}/{paired['pairs_total']} "
        f"separate**, **{paired['bonferroni']['pairs_resolved']}** under Bonferroni "
        f"alpha={paired['bonferroni']['alpha']:.5f}; ranking {' > '.join(paired['ranked_by_mean'])}")
    for pair, fw in zip(paired["pairs"], paired["bonferroni"]["pairs"]):
        flag = "separates" if pair["excludes_zero"] else "**NOT resolved**"
        fam = "" if fw["excludes_zero"] or not pair["excludes_zero"] else " (lost under Bonferroni)"
        out(f"  - {pair['a']} - {pair['b']}: {pair['mean_difference']:+.4f} "
            f"[{pair['lo']:+.4f}, {pair['hi']:+.4f}], n={pair['n_recordings']}, holds on "
            f"{pair['share_a_above_b']:.0%} -> {flag}{fam}")


def main() -> None:
    out = print
    out("# Every number in the paper, with its source file\n")
    # The digest lives beside the manuscript it maps, at `paper/NUMBERS.md`. The bare
    # `NUMBERS.md` here dated from the release layout, where `paper/` was stripped; that
    # path does not exist in this tree, so the file told the reader to regenerate it
    # somewhere it has never been.
    out("Regenerate with `python -m dataset_assessment.make_numbers_digest > "
        "paper/NUMBERS.md`.")
    out("Every value is read from `results/*.json`; nothing here is typed by hand, so a "
        "claim in")
    out("the draft that cannot be found below has no artifact behind it.\n")
    _benchmark_scale(out)
    _metric_dependencies(out)
    _controlled_addition(out)
    _nulls(out)
    _benchmark(out)
    _scale(out)
    _blank_signature(out)
    _cap_sensitivity(out)
    _esr_properties(out)
    _native_oracle(out)
    _paired_contrasts(out)
    _native_emlb(out)
    _common_support(out)
    _regime_and_audit(out)
    _nd_levels(out)
    _downstream(out)
    _architectures(out)
    _paired_seeds(out)


if __name__ == "__main__":
    main()
