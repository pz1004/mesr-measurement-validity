"""The null must be scored at a fixed retention, never at its own per-recording argmax.

These tests encode why Table 2 changed. `Delta(r=1) == 0` identically, so a per-recording
`max_r Delta` is bounded below by zero and reports a positive "gain" for a null that does
nothing at all. A regression here would silently restore that artifact.
"""

import json

import numpy as np
import pytest

from dataset_assessment.analyze import bootstrap_delta_ci, null_gain_at_fixed_retention


def _flat_curve(deltas, base=1.0):
    """A benchmark record whose null curve has the given Delta-over-Raw per retention."""

    return {
        "method": "random_null", "is_null": True, "raw_mesr": base,
        "curve": [{"r": r, "mesr": base + d, "kept_events": 10 ** 6, "evaluable": True}
                  for r, d in deltas],
    }


def _payload(records, retentions):
    return {"dataset": "synthetic", "retentions": retentions,
            "records": [{"recording": f"rec{i}", "methods": {"random_null": r}}
                        for i, r in enumerate(records)]}


@pytest.fixture
def written(tmp_path, monkeypatch):
    def write(payload):
        path = tmp_path / "benchmark_synthetic.json"
        path.write_text(json.dumps(payload))
        monkeypatch.setattr("dataset_assessment.analyze.RESULTS", tmp_path)
        return "synthetic"
    return write


def test_zero_mean_noise_yields_a_positive_max_selected_gain_but_no_fixed_gain(written):
    """The artifact, reproduced in miniature.

    Ten recordings whose null is pure zero-mean noise at every r < 1, and exactly 0 at r=1.
    Per-recording argmax reports a gain; the fixed-retention statistic does not.
    """

    rng = np.random.default_rng(0)
    grid = [round(0.1 * i, 2) for i in range(1, 11)]
    records = []
    for _ in range(10):
        noise = rng.normal(0.0, 0.05, len(grid) - 1)
        records.append(_flat_curve(list(zip(grid[:-1], noise)) + [(1.0, 0.0)]))
    gain = null_gain_at_fixed_retention(written(_payload(records, grid)))["random_null"]

    assert gain["max_selected"]["mean"] > 0, "the artifact must reproduce"
    assert gain["best_fixed"]["mean"] < gain["max_selected"]["mean"]
    assert not gain["raises_mesr_at_a_fixed_retention"] or \
        gain["best_fixed"]["lo"] < 0 < gain["best_fixed"]["hi"]


def test_max_selected_is_never_negative_because_delta_at_r_one_is_zero(written):
    """A null that strictly *lowers* MESR at every r still shows a non-negative argmax gain."""

    grid = [0.2, 0.5, 1.0]
    records = [_flat_curve([(0.2, -0.5), (0.5, -0.2), (1.0, 0.0)]) for _ in range(5)]
    gain = null_gain_at_fixed_retention(written(_payload(records, grid)))["random_null"]

    assert gain["max_selected"]["mean"] == pytest.approx(0.0)
    assert gain["max_selected"]["share_argmax_at_r1"] == pytest.approx(1.0)
    assert gain["best_fixed"]["mean"] == pytest.approx(-0.2)
    assert gain["raises_mesr_at_a_fixed_retention"] is False


def test_r_one_is_excluded_from_the_fixed_argmax(written):
    grid = [0.5, 1.0]
    records = [_flat_curve([(0.5, -0.1), (1.0, 0.0)]) for _ in range(4)]
    gain = null_gain_at_fixed_retention(written(_payload(records, grid)))["random_null"]

    assert gain["best_fixed"]["r"] == 0.5
    assert gain["n_retentions_below_one"] == 1


def test_a_genuine_gain_survives_fixed_retention_scoring(written):
    """DVSD22's shape: positive at every retention, so selection is not doing the work."""

    grid = [0.2, 0.5, 1.0]
    records = [_flat_curve([(0.2, 0.8), (0.5, 0.4), (1.0, 0.0)]) for _ in range(6)]
    gain = null_gain_at_fixed_retention(written(_payload(records, grid)))["random_null"]

    assert gain["raises_mesr_at_a_fixed_retention"] is True
    assert gain["n_retentions_with_positive_mean"] == 2
    assert gain["best_fixed"]["mean"] == pytest.approx(0.8)
    assert gain["best_fixed"]["lo"] > 0


def test_shipped_artifacts_still_show_the_published_direction_per_corpus():
    """Locks the five verdicts Table 2 prints, against the real artifacts."""

    expected = {"dnd21": False, "emlb": False,
                "dvsclean": True, "pure_ba": True, "dvsd22": True}
    for dataset, raises in expected.items():
        gain = null_gain_at_fixed_retention(dataset)["random_null"]
        assert gain["raises_mesr_at_a_fixed_retention"] is raises, dataset


def test_bootstrap_ci_brackets_its_own_mean():
    values = [0.1, 0.2, -0.05, 0.3, 0.0]
    interval = bootstrap_delta_ci(values)
    assert interval["lo"] <= interval["mean"] <= interval["hi"]


# --- Recordings are not independent draws --------------------------------------------
# The bootstrap above resamples recordings. Several corpora are the same scene re-recorded
# at a different injected rate, noise density or ND level, so that interval is narrower than
# the design supports. These lock the scene mapping and the wider interval it produces,
# because one corpus changes verdict under it and the paper's headline count depends on that.

def test_scene_of_recovers_the_independent_unit_for_every_corpus_layout():
    from dataset_assessment.analyze import scene_of
    assert scene_of("E-MLB/D-END/Books-ND00-1") == "E-MLB/D-END/Books"
    assert scene_of("E-MLB/N-END/Checker_Highlight-ND64-1") == "E-MLB/N-END/Checker_Highlight"
    assert scene_of("DVSCLEAN/MAH00446_50") == scene_of("DVSCLEAN/MAH00446_100")
    assert scene_of("DND21/1hz_hotel-bar") == scene_of("DND21/10hz_hotel-bar")
    assert scene_of("DND21/1hz_driving") != scene_of("DND21/1hz_hotel-bar")
    assert scene_of("DVSD22/HDD/2x30Hz_a") == scene_of("DVSD22/HDD/2x30Hz_c")
    # A recording with no siblings is its own unit.
    assert scene_of("Pure_BA/0.0") == "Pure_BA/0.0"


def test_cluster_bootstrap_is_wider_than_the_recording_bootstrap_when_scenes_repeat():
    from dataset_assessment.analyze import cluster_bootstrap_delta_ci
    values = [0.10, 0.11, 0.09, 0.10, -0.30, -0.31]
    scenes = ["a", "a", "a", "a", "b", "b"]
    naive = bootstrap_delta_ci(values)
    clustered = cluster_bootstrap_delta_ci(values, scenes)
    assert clustered["n_clusters"] == 2
    assert (clustered["hi"] - clustered["lo"]) > (naive["hi"] - naive["lo"])


def test_cluster_bootstrap_reduces_to_the_plain_one_when_every_recording_is_its_own_scene():
    from dataset_assessment.analyze import cluster_bootstrap_delta_ci
    values = [0.1, 0.2, -0.05, 0.3, 0.0]
    scenes = [f"s{i}" for i in range(len(values))]
    a, b = bootstrap_delta_ci(values), cluster_bootstrap_delta_ci(values, scenes)
    assert b["mean"] == pytest.approx(a["mean"])
    assert b["lo"] == pytest.approx(a["lo"], abs=0.02)


def test_dvsclean_null_gain_does_not_survive_its_own_scene_structure():
    """The paper counts two corpora, not three, and this is why.

    DVSCLEAN's ten recordings are five sequences at two noise densities. Resampling the five
    sequences puts zero inside the interval, so the corpus does not establish the specificity
    failure and the paper does not claim it does.
    """

    best = null_gain_at_fixed_retention("dvsclean")["random_null"]["best_fixed"]
    assert best["lo"] > 0, "recording-level interval excludes zero"
    assert best["cluster_ci"]["n_clusters"] == 5
    assert best["cluster_ci"]["excludes_zero"] is False


def test_the_two_real_unfiltered_corpora_survive_clustering():
    for dataset in ("dvsd22", "pure_ba"):
        cluster = null_gain_at_fixed_retention(dataset)["random_null"]["best_fixed"]["cluster_ci"]
        assert cluster["excludes_zero"] is True, dataset
        assert cluster["lo"] > 0, dataset


def test_pure_ba_positive_mean_is_a_tail_not_the_typical_recording():
    """Reported because the mean and the median disagree in sign; the paper says so."""

    best = null_gain_at_fixed_retention("pure_ba")["random_null"]["best_fixed"]
    assert best["mean"] > 0 and best["median"] < 0
    assert best["n_above_raw"] * 2 < best["n"]
