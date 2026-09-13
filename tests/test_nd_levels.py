import json
from pathlib import Path

import numpy as np
import pytest

from dataset_assessment.nd_levels import (
    LEVELS, LEVEL_MAPPING, grouping_contrasts, matched_levels, nd_level_contrast,
    parse_emlb, synthetic_base_scene, unattenuated_vs_synthetic,
)

PROFILES = Path(__file__).resolve().parents[1] / "results" / "noise_regimes.json"


def load_profiles():
    if not PROFILES.is_file():
        pytest.skip("noise_regimes.json not generated")
    return json.loads(PROFILES.read_text())["profiles"]


def crossed_profiles(scene_offsets, level_effects, stat="hot_pixel_share", rate=1.0):
    """A fully crossed E-MLB-shaped corpus: value = scene offset + level effect."""

    out = []
    for i, offset in enumerate(scene_offsets):
        for level, effect in zip(LEVELS, level_effects):
            out.append({"name": f"E-MLB/D-END/Scene{i}-{level}-1", "synthetic": False,
                        stat: offset + effect, "event_rate_per_px": rate})
    return out


def test_level_mapping_covers_the_release_and_marks_only_one_level_unattenuated():
    """ND00 is the whole argument: if it is not the unfiltered level, nothing here holds."""

    assert tuple(LEVEL_MAPPING) == LEVELS
    unattenuated = [k for k, v in LEVEL_MAPPING.items() if not v["attenuated"]]
    assert unattenuated == ["ND00"]
    assert LEVEL_MAPPING["ND00"]["transmittance"] == 1.0
    transmittances = [LEVEL_MAPPING[level]["transmittance"] for level in LEVELS]
    assert transmittances == sorted(transmittances, reverse=True)
    # Every level records how its name was mapped onto the paper's condition.
    assert all(v["basis"] for v in LEVEL_MAPPING.values())


def test_parse_emlb_keys_scenes_by_session_and_rejects_other_corpora():
    parsed = parse_emlb("E-MLB/D-END/Architecture-ND00-1")
    assert parsed["level"] == "ND00"
    assert parsed["scene"] == "D-END/Architecture"
    # The same scene name in the other session must not collide.
    assert parse_emlb("E-MLB/N-END/Architecture-ND00-1")["scene"] == "N-END/Architecture"
    assert parse_emlb("Pure_BA/level3") is None
    assert parse_emlb("DND21/1hz_hotel-bar") is None


def test_synthetic_base_scene_reads_both_naming_patterns():
    """The injection parameter sits on opposite sides of the separator in the two corpora."""

    assert synthetic_base_scene("DND21/1hz_hotel-bar") == "DND21/hotel-bar"
    assert synthetic_base_scene("DND21/10hz_hotel-bar") == "DND21/hotel-bar"
    assert synthetic_base_scene("DVSCLEAN/MAH00446_50") == "DVSCLEAN/MAH00446"
    assert synthetic_base_scene("DVSCLEAN/MAH00446_100") == "DVSCLEAN/MAH00446"


def test_matched_levels_keeps_rows_aligned_by_scene():
    profiles = crossed_profiles([0.0, 10.0], [0.0, 1.0, 2.0, 3.0])
    scenes, matrix = matched_levels(profiles)
    assert len(scenes) == 2 and matrix.shape == (2, 4)
    assert np.allclose(matrix[0], [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(matrix[1], [10.0, 11.0, 12.0, 13.0])


def test_matched_levels_drops_scenes_missing_a_level():
    profiles = crossed_profiles([0.0], [0.0, 1.0, 2.0, 3.0])
    del profiles[-1]
    scenes, matrix = matched_levels(profiles)
    assert scenes == [] and matrix.size == 0
    with pytest.raises(ValueError):
        nd_level_contrast(profiles)


def test_the_paired_contrast_sees_what_the_unpaired_one_misses():
    """Why the level contrast must be paired.

    Scene-to-scene spread is made an order of magnitude larger than the level effect, which
    is the real corpus's situation. The within-scene test finds the effect; the unpaired test
    on the same numbers does not. Running the unpaired test here would have invited the
    objection that the difference is scene composition.
    """

    rng = np.random.default_rng(0)
    offsets = rng.normal(0.0, 1.0, 40)
    profiles = crossed_profiles(offsets, [0.0, 0.03, 0.06, 0.09])

    result = nd_level_contrast(profiles)
    assert result["paired"]["friedman_p"] < 1e-6
    assert result["paired"]["scenes_rising_first_to_last"] == 40
    assert result["paired"]["scenes_strictly_monotone"] == 40
    assert result["unpaired_secondary"]["p_value"] > 0.05


def test_rank_biserial_sign_marks_the_first_group_as_the_lower_one():
    """+1 must mean the first-named group lies entirely below the second."""

    profiles = ([{"name": f"DND21/{i}hz_a", "synthetic": True,
                  "hot_pixel_share": 0.01, "event_rate_per_px": 1.0} for i in range(8)]
                + [{"name": f"Pure_BA/l{i}", "synthetic": False,
                    "hot_pixel_share": 0.9, "event_rate_per_px": 1.0} for i in range(8)])
    contrasts = grouping_contrasts(profiles)
    assert contrasts["provenance_all_corpora"]["rank_biserial"] == pytest.approx(1.0)
    assert contrasts["provenance_all_corpora"]["separates"]


def test_non_separation_is_reported_without_an_equivalence_verdict():
    """A null result must not acquire a field that reads as 'the same'."""

    rng = np.random.default_rng(1)
    profiles = [{"name": f"E-MLB/D-END/S{i}-{level}-1", "synthetic": False,
                 "hot_pixel_share": float(rng.normal(1.0, 0.1)), "event_rate_per_px": 1.0}
                for i in range(24) for level in LEVELS]
    profiles += [{"name": f"DND21/{i}hz_a", "synthetic": True,
                  "hot_pixel_share": float(rng.normal(1.0, 0.1)), "event_rate_per_px": 1.0}
                 for i in range(20)]
    result = unattenuated_vs_synthetic(profiles)
    assert result["separates"] is False
    assert "p_value" in result and "unattenuated_share_inside_synthetic_range" in result
    assert not any("equival" in str(k).lower() for k in result)


def test_published_nd_levels_reproduce_the_section_viii_rewrite():
    """The four numbers the article will print, against the released artifact."""

    profiles = load_profiles()

    levels = nd_level_contrast(profiles)
    assert levels["design"] == {"scenes": 96, "levels": list(LEVELS),
                                "recordings": 384, "fully_crossed": True}
    medians = [levels["per_level"][level]["median"] for level in LEVELS]
    assert medians == sorted(medians), "share must rise with attenuation"
    assert medians[0] == pytest.approx(0.00766, abs=5e-5)
    assert medians[-1] == pytest.approx(0.01146, abs=5e-5)
    assert levels["paired"]["friedman_p"] < 1e-10
    assert levels["paired"]["wilcoxon_first_vs_last_p"] < 1e-6
    assert levels["paired"]["scenes_rising_first_to_last"] == 73

    contrast = unattenuated_vs_synthetic(profiles)
    assert contrast["separates"] is False
    assert contrast["p_value"] == pytest.approx(0.154, abs=5e-4)
    assert contrast["synthetic_base_scenes"] == 7
    assert contrast["unattenuated_share_inside_synthetic_range"] == pytest.approx(0.573,
                                                                                 abs=5e-3)


def test_the_split_that_separates_is_not_the_provenance_split():
    """Section VIII's exception is corpus identity, measured against the same recordings."""

    groupings = grouping_contrasts(load_profiles())
    provenance = groupings["provenance_all_corpora"]
    withheld = groupings["provenance_without_emlb"]
    identity = groupings["corpus_identity"]

    assert provenance["separates"] is False
    assert withheld["separates"] and identity["separates"]
    # Same 53 recordings on the far side of both separating splits.
    assert withheld["n_second"] == identity["n_second"] == 53
    # The provenance-free split separates more strongly, and it holds E-MLB on the low side.
    assert identity["p_value"] < withheld["p_value"]
    assert identity["n_first"] == 404


def test_a_missing_value_fails_loudly_rather_than_reaching_the_paper():
    """A NaN would pass through Friedman and be printed as a NaN p-value beside a claim."""

    profiles = crossed_profiles([0.0, 1.0], [0.0, 1.0, 2.0, 3.0])
    profiles[5]["hot_pixel_share"] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        matched_levels(profiles)


def test_duplicate_profile_names_are_refused():
    """The groupings key on name, so a collision would silently drop a recording."""

    profiles = [{"name": "Pure_BA/l1", "synthetic": False,
                 "hot_pixel_share": 0.5, "event_rate_per_px": 1.0} for _ in range(2)]
    with pytest.raises(ValueError, match="not unique"):
        grouping_contrasts(profiles)
