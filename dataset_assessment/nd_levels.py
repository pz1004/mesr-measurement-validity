"""What E-MLB's neutral-density levels do to busiest-pixel share, and what they do not explain.

Section VIII of the article attributed E-MLB's low busiest-pixel share to its neutral-density
filters: attenuation without sensor-level hot pixels. Two facts refute that attribution, and
both are recoverable from `results/noise_regimes.json` with no new decoding.

1. E-MLB documents the filters as *raising* the noise level ("The noise level gradually
   increases with the amount of light entering the lens reduction"), and a quarter of the
   corpus is captured with no filter at all. Attenuation cannot be why the corpus looks
   clean.
2. Within the corpus the share *rises* with attenuation, so the unattenuated level is the
   lowest of the four -- the opposite of the direction the attribution needs.

The design is fully crossed -- 96 scenes x 4 levels x 1 repetition = 384 -- so the level
contrast is PAIRED. An unpaired test on this design throws away the pairing and invites the
objection that the difference is scene composition; `nd_level_contrast` therefore reports
Friedman across the four levels and Wilcoxon on the matched ND00/ND64 pairs as primary, with
the unpaired figures kept only as a secondary line.

The second half of the module is the consequence for the leave-one-corpus-out result in
`profile_datasets`. Withholding E-MLB makes busiest-pixel share separate synthetic from real
perfectly, which the article reads as evidence that real noise is distinguishable once E-MLB
is set aside. `grouping_contrasts` tests that reading directly: the same 53 recordings
separate from the other 404 far more strongly under a grouping that ignores provenance
entirely and places E-MLB, a real corpus, with the two synthetic ones. The exception is a
corpus-identity effect. It says which two corpora remain, not that provenance is visible.

No mechanism is asserted here. Pure_BA is documented as a background-activity series with no
scene content and DVSD22 as sparse falling drops in a darkroom, and the event rate moves
against the busiest-pixel share both across the corpus set and within E-MLB's matched levels
-- but the rate does not account for DVSCLEAN, which is low on both. That grouping is a
post-hoc description of which corpora these are, not a tested cause, and `basis` records it
as such in the emitted artifact.

The separation bar is `profile_datasets`'s registered one, p < 0.01 and |rank-biserial| > 0.8,
reused so every row here is comparable with `results/noise_regimes.json`. Rank-biserial keeps
that module's sign convention, 1 - 2U/(n1*n2) with the first-named group first, so +1 means
the first group lies entirely below the second.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
PROFILES = RESULTS / "noise_regimes.json"
OUT = RESULTS / "nd_levels.json"

#: Statistic under test throughout. Named `hot_pixel_share` in the artifact for historical
#: reasons; the article calls it busiest-pixel share, because it presumes no defect pixel.
STAT = "hot_pixel_share"

#: Release directory name -> the condition the E-MLB paper names, with its transmittance.
#: ND04/ND16/ND64 map literally onto the paper's ND4/ND16/ND64. ND00 maps onto ND1 by
#: elimination and by the ordering of transmittance: the paper states "ND1 is used to
#: represent the data captured without any ND filters", and ND00 is the one release level
#: with no paper counterpart left. The whole ND argument rests on this mapping, so it is
#: emitted as data rather than left as an unstated inference.
LEVEL_MAPPING = {
    "ND00": {"paper_name": "ND1", "transmittance": 1.0, "attenuated": False,
             "basis": "by elimination; paper: 'ND1 is used to represent the data captured "
                      "without any ND filters'"},
    "ND04": {"paper_name": "ND4", "transmittance": 1 / 4, "attenuated": True,
             "basis": "literal name match"},
    "ND16": {"paper_name": "ND16", "transmittance": 1 / 16, "attenuated": True,
             "basis": "literal name match"},
    "ND64": {"paper_name": "ND64", "transmittance": 1 / 64, "attenuated": True,
             "basis": "literal name match"},
}

#: Ordered least to most attenuated, which is also E-MLB's own nominal noise ordering.
LEVELS: Tuple[str, ...] = ("ND00", "ND04", "ND16", "ND64")

#: The grouping that separates best, named by corpus. Not a hypothesis and not registered.
CONCENTRATED_CORPORA = ("Pure_BA", "DVSD22")

SEPARATION_BAR = {"p_value": 0.01, "abs_rank_biserial": 0.8,
                  "source": "profile_datasets, registered before that run"}

_EMLB_NAME = re.compile(r"^(?P<scene>.+)-(?P<level>ND\d\d)-(?P<repetition>\d+)$")


def corpus_of(name: str) -> str:
    """Leading path component of a profile name, e.g. 'E-MLB/D-END/Architecture-ND00-1'."""

    return name.split("/", 1)[0]


def parse_emlb(name: str) -> Optional[Dict[str, str]]:
    """Session, scene and ND level of one E-MLB profile name, or None if it is not one.

    The scene key carries the session because scene names are only unique within a session.
    """

    parts = name.split("/")
    if len(parts) != 3 or parts[0] != "E-MLB":
        return None
    match = _EMLB_NAME.match(parts[2])
    if match is None:
        return None
    return {"session": parts[1], "scene": f"{parts[1]}/{match['scene']}",
            "level": match["level"], "repetition": match["repetition"]}


def _mannwhitney(first: Sequence[float], second: Sequence[float]) -> Dict:
    """`profile_datasets`'s contrast, sign convention included: +1 means first below second."""

    from scipy.stats import mannwhitneyu

    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"n_first": int(len(a)), "n_second": int(len(b)),
                "p_value": float("nan"), "rank_biserial": float("nan"), "separates": False}
    u = mannwhitneyu(a, b, alternative="two-sided")
    effect = 1 - 2 * u.statistic / (len(a) * len(b))
    return {"n_first": int(len(a)), "n_second": int(len(b)),
            "median_first": float(np.median(a)), "median_second": float(np.median(b)),
            "p_value": float(u.pvalue), "rank_biserial": float(effect),
            "separates": bool(u.pvalue < SEPARATION_BAR["p_value"]
                              and abs(effect) > SEPARATION_BAR["abs_rank_biserial"])}


def matched_levels(profiles: Sequence[Dict], stat: str = STAT
                   ) -> Tuple[List[str], np.ndarray]:
    """Scenes holding all four ND levels, and their values as a (scenes x 4) matrix.

    Returning the matrix rather than four flat vectors is what keeps the contrast paired.
    """

    by_scene: Dict[str, Dict[str, float]] = {}
    for profile in profiles:
        parsed = parse_emlb(profile["name"])
        if parsed is None:
            continue
        by_scene.setdefault(parsed["scene"], {})[parsed["level"]] = float(profile[stat])
    scenes = sorted(s for s, levels in by_scene.items()
                    if all(level in levels for level in LEVELS))
    matrix = np.array([[by_scene[s][level] for level in LEVELS] for s in scenes],
                      dtype=float).reshape(len(scenes), len(LEVELS))
    # A NaN here would pass silently through `friedmanchisquare` and be printed as a NaN
    # p-value beside a claim in the paper. Fail instead, naming the scene.
    if scenes and not np.isfinite(matrix).all():
        bad = [scenes[i] for i in np.flatnonzero(~np.isfinite(matrix).all(axis=1))]
        raise ValueError(f"{stat} is not finite on {len(bad)} scene(s): {bad[:3]}")
    return scenes, matrix


def nd_level_contrast(profiles: Sequence[Dict], stat: str = STAT) -> Dict:
    """Does `stat` track ND level within E-MLB? Paired across scenes; unpaired as a footnote."""

    from scipy.stats import friedmanchisquare, wilcoxon

    scenes, matrix = matched_levels(profiles, stat)
    if len(scenes) < 2:
        raise ValueError("fewer than two fully crossed scenes; the design is not matched")

    friedman = friedmanchisquare(*matrix.T)
    unattenuated, most_attenuated = matrix[:, 0], matrix[:, -1]
    signed = wilcoxon(unattenuated, most_attenuated)
    return {
        "statistic": stat,
        "design": {"scenes": len(scenes), "levels": list(LEVELS),
                   "recordings": int(matrix.size), "fully_crossed": True},
        "per_level": {level: {"n": int(matrix.shape[0]),
                              "median": float(np.median(matrix[:, i])),
                              "mean": float(matrix[:, i].mean()),
                              "iqr": [float(np.percentile(matrix[:, i], 25)),
                                      float(np.percentile(matrix[:, i], 75))]}
                      for i, level in enumerate(LEVELS)},
        "paired": {
            "friedman_chi2": float(friedman.statistic),
            "friedman_p": float(friedman.pvalue),
            "wilcoxon_first_vs_last_p": float(signed.pvalue),
            "scenes_rising_first_to_last": int((most_attenuated > unattenuated).sum()),
            "scenes_strictly_monotone": int(np.all(np.diff(matrix, axis=1) > 0,
                                                   axis=1).sum()),
        },
        "unpaired_secondary": _mannwhitney(unattenuated, most_attenuated),
        "reading": ("The unattenuated level is the lowest of the four, so attenuation does "
                    "not explain E-MLB's low busiest-pixel share."),
    }


def synthetic_base_scene(name: str) -> str:
    """The underlying recording a synthetic profile was built from.

    Both synthetic corpora vary one injection parameter over a small set of base recordings,
    so the number of profiles overstates the number of independent scenes. DND21 names are
    `<rate>hz_<scene>` and DVSCLEAN names are `<clip>_<density>`; the parameter sits on
    opposite sides of the separator, which is why this cannot be one split.
    """

    corpus, _, stem = name.partition("/")
    if corpus == "DND21":
        return f"{corpus}/{stem.split('_', 1)[-1]}"
    if corpus == "DVSCLEAN":
        return f"{corpus}/{stem.rsplit('_', 1)[0]}"
    return name


def unattenuated_vs_synthetic(profiles: Sequence[Dict], stat: str = STAT) -> Dict:
    """E-MLB's unfiltered quarter against the synthetic corpora. Non-separation, not equivalence."""

    unattenuated = [float(p[stat]) for p in profiles
                    if (parsed := parse_emlb(p["name"])) and parsed["level"] == "ND00"]
    synthetic = [float(p[stat]) for p in profiles if p["synthetic"]]
    base_scenes = sorted({synthetic_base_scene(p["name"])
                          for p in profiles if p["synthetic"]})
    low, high = float(np.min(synthetic)), float(np.max(synthetic))
    inside = float(np.mean([low <= v <= high for v in unattenuated]))

    contrast = _mannwhitney(unattenuated, synthetic)
    contrast.update({
        "synthetic_base_scenes": len(base_scenes),
        "synthetic_range": [low, high],
        "unattenuated_range": [float(np.min(unattenuated)), float(np.max(unattenuated))],
        "unattenuated_share_inside_synthetic_range": inside,
        "reading": ("Containment and non-separation under the registered bar. Not "
                    "equivalence: the comparison group is 20 recordings from a handful of "
                    "base scenes and the two spreads differ."),
    })
    return contrast


def grouping_contrasts(profiles: Sequence[Dict], stat: str = STAT) -> Dict:
    """Three ways to cut the same 457 recordings. Only the third one separates strongly.

    The point is the comparison between them: the grouping that separates best ignores
    provenance and puts a real corpus with the synthetic ones, so the leave-one-out exception
    in `profile_datasets` is about which corpora remain, not about real versus synthetic.
    """

    names = [p["name"] for p in profiles]
    if len(set(names)) != len(names):
        raise ValueError("profile names are not unique; the groupings would drop rows")
    value = {p["name"]: float(p[stat]) for p in profiles}
    corpus = {p["name"]: corpus_of(p["name"]) for p in profiles}
    synthetic = {p["name"]: bool(p["synthetic"]) for p in profiles}

    def values(names) -> List[float]:
        return [value[n] for n in names]

    every = list(value)
    synth = [n for n in every if synthetic[n]]
    real = [n for n in every if not synthetic[n]]
    real_no_emlb = [n for n in real if corpus[n] != "E-MLB"]
    concentrated = [n for n in every if corpus[n] in CONCENTRATED_CORPORA]
    rest = [n for n in every if corpus[n] not in CONCENTRATED_CORPORA]

    return {
        "statistic": stat,
        "provenance_all_corpora": dict(
            _mannwhitney(values(synth), values(real)),
            groups=["synthetic", "real"]),
        "provenance_without_emlb": dict(
            _mannwhitney(values(synth), values(real_no_emlb)),
            groups=["synthetic", "real minus E-MLB"]),
        "corpus_identity": dict(
            _mannwhitney(values(rest), values(concentrated)),
            groups=["E-MLB + DND21 + DVSCLEAN", " + ".join(CONCENTRATED_CORPORA)]),
        "reading": ("The corpus-identity split uses the same 53 recordings as the "
                    "withheld-E-MLB split and separates more strongly, while placing the "
                    "real corpus E-MLB with the synthetic ones. The exception is therefore "
                    "corpus identity, not provenance."),
        "basis": ("Post-hoc, not registered, and not a tested cause. Pure_BA is documented "
                  "as a background-activity series with no scene content and DVSD22 as "
                  "sparse falling drops in a darkroom; both are content-sparse static "
                  "captures. Event rate per pixel moves against the statistic across the "
                  "corpus set and within E-MLB's matched levels, but does not account for "
                  "DVSCLEAN, which is low on both."),
    }


def rate_association(profiles: Sequence[Dict], stat: str = STAT) -> Dict:
    """Does activity density move against concentration? Descriptive support for `basis`."""

    from scipy.stats import friedmanchisquare, spearmanr

    rate = np.array([float(p["event_rate_per_px"]) for p in profiles])
    concentration = np.array([float(p[stat]) for p in profiles])
    finite = np.isfinite(rate) & np.isfinite(concentration)
    overall = spearmanr(rate[finite], concentration[finite])

    _, rate_matrix = matched_levels(profiles, "event_rate_per_px")
    return {
        "overall_spearman": float(overall.statistic),
        "overall_p": float(overall.pvalue),
        "n": int(finite.sum()),
        "emlb_rate_median_per_level": {level: float(np.median(rate_matrix[:, i]))
                                       for i, level in enumerate(LEVELS)},
        "emlb_rate_friedman_p": float(friedmanchisquare(*rate_matrix.T).pvalue),
        "caveat": "Does not account for DVSCLEAN, which is low on rate and on concentration.",
    }


def measure(profiles: Sequence[Dict], stat: str = STAT) -> Dict:
    started = time.perf_counter()
    payload = {
        "statistic": stat,
        "level_mapping": LEVEL_MAPPING,
        "separation_bar": SEPARATION_BAR,
        "nd_levels": nd_level_contrast(profiles, stat),
        "unattenuated_vs_synthetic": unattenuated_vs_synthetic(profiles, stat),
        "groupings": grouping_contrasts(profiles, stat),
        "rate_association": rate_association(profiles, stat),
    }
    payload["wall_seconds"] = time.perf_counter() - started
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=PROFILES)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    profiles = json.loads(args.profiles.read_text())["profiles"]
    payload = measure(profiles)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=float) + "\n")

    nd = payload["nd_levels"]
    print(f"=== E-MLB ND levels ({nd['design']['scenes']} fully crossed scenes) ===")
    for level in LEVELS:
        row = nd["per_level"][level]
        mapped = LEVEL_MAPPING[level]
        print(f"  {level} ({mapped['paper_name']:>5s}, transmittance {mapped['transmittance']:.4f}) "
              f" median={row['median']:.5f} mean={row['mean']:.5f}")
    print(f"  paired: Friedman chi2={nd['paired']['friedman_chi2']:.2f} "
          f"p={nd['paired']['friedman_p']:.3e}; Wilcoxon ND00 vs ND64 "
          f"p={nd['paired']['wilcoxon_first_vs_last_p']:.3e}; rising in "
          f"{nd['paired']['scenes_rising_first_to_last']}/{nd['design']['scenes']} scenes")

    contrast = payload["unattenuated_vs_synthetic"]
    print("\n=== unattenuated E-MLB vs synthetic ===")
    print(f"  p={contrast['p_value']:.3f} rb={contrast['rank_biserial']:+.3f} "
          f"separates={contrast['separates']}; "
          f"{contrast['unattenuated_share_inside_synthetic_range']:.0%} inside the "
          f"synthetic range")

    print("\n=== which split separates? ===")
    for key, row in payload["groupings"].items():
        if not isinstance(row, dict):
            continue
        print(f"  {key:<26s} {row['groups'][0]} vs {row['groups'][1]}: "
              f"p={row['p_value']:.2e} rb={row['rank_biserial']:+.3f} "
              f"separates={row['separates']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
