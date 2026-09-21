"""Tests for the frozen-classifier criterion probe.

The probe's one piece of non-obvious bookkeeping is that retention 1.0 is a single
condition wearing eight method labels: `retain_mask` keeps every event regardless of score,
so all eight streams - and therefore all eight MESR values and all eight accuracies - are
identical. The first run of this module got that wrong. `_condition_name` returned
`clean__dwf`, `clean__evflow` and so on, the deduplication guard never fired, eight
byte-identical frame arrays were written, and eight tied points entered the correlation
computed over the whole grid. The partial Spearman moved from -0.0015 at n = 40 to -0.0678
at n = 33 once the ties were removed.

Both halves are pinned below. The paper's own argument is that a statistic which returns a
plausible number from an input it cannot speak about is a silent-error hazard; a correlation
whose n is inflated by duplicate rows is exactly that.
"""

from __future__ import annotations

from dataset_assessment.downstream_gesture_frozen import (PAPER_METHODS, RETENTIONS,
                                                          _condition_name, dedupe_rows)


def _row(method: str, retention: float, mesr: float, accuracy: float) -> dict:
    return {"method": method, "retention": retention, "mesr": mesr, "accuracy": accuracy}


def test_full_retention_collapses_to_one_name_across_methods():
    names = {_condition_name(method, 1.0) for method in PAPER_METHODS}
    assert names == {"clean__unfiltered"}


def test_partial_retention_keeps_methods_distinct():
    names = {_condition_name(method, 0.6) for method in PAPER_METHODS}
    assert len(names) == len(PAPER_METHODS)
    # `raw` is the no-discrimination reference the released evaluator pairs against, so it
    # must carry the name that evaluator looks for at its own retention level.
    assert _condition_name("raw", 0.6) == "r060__unfiltered"
    assert "r060__red" in names


def test_dedupe_removes_only_the_duplicated_full_retention_rows():
    rows = [_row(m, r, 1.4 + 0.01 * i, 0.6 + 0.01 * i)
            for i, m in enumerate(PAPER_METHODS) for r in RETENTIONS]
    # Every method reports the identical pair at r = 1.0, because the stream is identical.
    for row in rows:
        if row["retention"] == 1.0:
            row["mesr"], row["accuracy"] = 1.4327, 0.96875

    deduped = dedupe_rows(rows)

    assert len(rows) == len(PAPER_METHODS) * len(RETENTIONS)
    assert len(deduped) == len(PAPER_METHODS) * (len(RETENTIONS) - 1) + 1
    full = [r for r in deduped if r["retention"] == 1.0]
    assert len(full) == 1
    # The surviving row must not claim to belong to whichever method happened to be first.
    assert full[0]["method"] == "all"
    assert {r["method"] for r in deduped if r["retention"] < 1.0} == set(PAPER_METHODS)


def test_dedupe_is_idempotent():
    rows = [_row(m, r, 1.4, 0.6) for m in PAPER_METHODS for r in RETENTIONS]
    once = dedupe_rows(rows)
    assert len(dedupe_rows(once)) == len(once)
