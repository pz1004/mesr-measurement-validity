"""The E-MLB table's ordering is not carried by how much each filter discards.

Section VI-D item (iii) confines the aggressiveness/selectivity confound to the single
RED-vs-EDformer pair, because across the table as a whole the rank correlation between
Delta-over-Raw and native retention is *positive*: the rows that keep more events score
higher.  These tests pin the number the section quotes and the sign it depends on.
"""

from dataset_assessment.analyze import native_eligibility
from dataset_assessment.denoisors import CLASSICAL


def test_the_emlb_ordering_does_not_follow_aggressiveness():
    agg = native_eligibility("emlb")["delta_vs_native_retention_over_eligible_rows"]
    assert agg["methods"] == len(CLASSICAL)
    # S VI-D item (iii) prints this to two decimals.
    assert round(agg["spearman"], 2) == 0.60
    # The sign is what licenses "runs the other way": more retention, higher Delta.
    assert agg["spearman"] > 0
    assert agg["kendall"] > 0


def test_the_correlation_is_computed_on_the_cohort_the_table_prints():
    elig = native_eligibility("emlb")
    rows = {m: v for m, v in elig["by_method"].items() if m in CLASSICAL}
    # tab:emlb's own values, so the correlation cannot drift from the printed table.
    assert round(rows["red"]["eligible_mean_delta"], 4) == 0.4550
    assert rows["red"]["eligible"] == 311
    assert round(rows["red"]["eligible_mean_native_r"], 2) == 0.26
    assert round(rows["evflow"]["eligible_mean_native_r"], 2) == 0.13
    # RED is the most aggressive of the top two and EvFlow the most aggressive overall,
    # yet EvFlow sits fifth: that pair alone breaks the aggressiveness story.
    assert (rows["evflow"]["eligible_mean_native_r"]
            < rows["red"]["eligible_mean_native_r"])
    assert (rows["evflow"]["eligible_mean_delta"]
            < rows["red"]["eligible_mean_delta"])
