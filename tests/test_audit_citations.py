"""The audit table's Location column must match the artifact it claims to summarise.

Three rows cited `engines/test.py` and `datasets/denoise_emlb.py`; EDmamba ships both
under `pointcept/`, so a reviewer checking the claim would not have found the files. The
artifact had the full path all along -- only the manuscript was short. A fourth row cited
the `median_filter` helper for a claim about the variant that calls it. These are claims
about third parties' released code, so the citation is the whole evidence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT = ROOT / "paper/supplementary_tim.tex"
AUDIT = json.loads((ROOT / "results/reproducibility_audit.json").read_text())


def _recorded_locations() -> set[str]:
    """`file:line` for every check that cites one. One check evidences a claim with a
    directory listing instead (`cuke-emlb/modules/net` holds only `.gitkeep`), so the
    file key is not universal."""

    return {f"{c['evidence']['file']}:{c['evidence']['line']}"
            for c in AUDIT["checks"]
            if c["evidence"] and "file" in c["evidence"]}


def _cited_locations() -> set[str]:
    """Every `\\texttt{path:line}` inside the audit float, LaTeX escaping removed.

    The label sits above the rows, not below them, so slicing at the label returns the
    caption and none of the table. An empty result must fail rather than pass vacuously --
    it did, and it hid the very defect these tests exist to catch.
    """

    text = SUPPLEMENT.read_text(encoding="utf-8", errors="replace")
    label = text.index(r"\label{tab:audit}")
    opened = text.rindex(r"\begin{widetable}", 0, label)
    closed = text.index(r"\end{widetable}", label)
    found = {m.group(1).replace(r"\_", "_")
             for m in re.finditer(r"\\texttt\{([^}]*\.py:\d+)\}", text[opened:closed])}
    assert found, "no file:line citations found in the audit table; the parser is stale"
    return found


@pytest.mark.skipif(not SUPPLEMENT.exists(), reason="manuscript not in this checkout")
def test_every_cited_line_matches_the_audit_artifact():
    evidence = _recorded_locations()
    assert evidence, "the audit recorded no file evidence at all"
    for cited in _cited_locations():
        assert any(full.endswith(cited) for full in evidence), (
            f"{cited} is cited in the audit table but no audit check records it; "
            f"recorded: {sorted(evidence)}")


# The only prefixes a citation may drop are the repository name and, for cuke-emlb, the
# source root inside it. Dropping an interior package directory -- `pointcept/` -- leaves a
# path that does not exist at the repository root, which is what the reader starts from.
DROPPABLE = ({("EDmamba",), ("EDformer",), ("cuke-emlb",), ("cuke-emlb", "python", "src")})


@pytest.mark.skipif(not SUPPLEMENT.exists(), reason="manuscript not in this checkout")
def test_a_cited_path_is_not_truncated_past_the_repository_root():
    """A suffix match alone accepts `engines/test.py:481` for a file that ships at
    `pointcept/engines/test.py`. That was the actual defect, so the test has to reject it:
    what the paper drops must be a whole repository prefix and nothing deeper."""

    for cited in _cited_locations():
        full = next(f for f in _recorded_locations() if f.endswith(cited))
        remainder = full[:len(full) - len(cited)]
        dropped = tuple(p for p in remainder.strip("/").split("/") if p)
        assert dropped in DROPPABLE, (
            f"`{cited}` drops {list(dropped)} from `{full}`; a reader starting at the "
            f"repository root cannot find that path")


def test_all_nine_claims_are_verified():
    assert len(AUDIT["checks"]) == 9
    assert all(c["verified"] for c in AUDIT["checks"])


def test_the_edformer_row_points_at_the_variant_not_the_helper_it_calls():
    hit = next(c["evidence"] for c in AUDIT["checks"]
               if c["evidence"] and c["evidence"]["file"] == "EDformer/metrics.py")
    assert hit["text"].startswith("class EventStructuralRatioV2")
