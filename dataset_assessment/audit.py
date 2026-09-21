"""Re-verify, from source, every reproducibility claim the paper makes.

Each check is a static or executable fact about a released repository, recorded with the
file and line so a reader can confirm it independently. Nothing here may be cited from the
plan's briefing section: if a check cannot confirm a claim, the claim does not go in the
paper.

Every check reads the file **at the commit pinned below**, through `git show`, never
through the working copy. An earlier version of this module grepped the checked-out trees
reachable by the `EDmamba`, `EDformer` and `cuke-emlb` symlinks. Those trees carry local
edits, so three EDmamba line numbers were off by a uniform three and were published that
way: a reader following `pointcept/engines/test.py:481` would land past the commented-out
ESR call it was supposed to evidence. The claims all survived re-checking, but the
locators did not, and a locator is the whole evidence for a claim about someone else's
released code.

Reading the blob rather than the file also makes the artifact reproducible: it no longer
depends on the state anyone's checkout happens to be in, and it cannot drift again when
those checkouts are updated.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results/reproducibility_audit.json"

#: Each release's default branch at the commit the supplement pins. Changing one of these
#: changes what the paper claims, so they are data, not configuration: update them only
#: alongside the supplement's own pinning paragraph.
PINNED = {"EDmamba": "302f758", "EDformer": "820ff16", "cuke-emlb": "105a1c3"}
ACCESSED = "2026-09-21"


def _git(repo: str, *arguments: str) -> Optional[str]:
    """Run git in the checkout `repo` names, or None if it fails.

    The checkouts are reached through the symlinks beside this project. Only their object
    stores are used - never their working trees - so a dirty checkout cannot reach a check.
    """

    completed = subprocess.run(("git", "-C", str(PROJECT_ROOT / repo)) + arguments,
                               capture_output=True, text=True)
    return completed.stdout if completed.returncode == 0 else None


def blob(repo: str, path: str) -> Optional[str]:
    """`path` as it stands at `repo`'s pinned commit."""

    return _git(repo, "show", f"{PINNED[repo]}:{path}")


def grep(repo: str, path: str, pattern: str, after: str | None = None) -> Optional[dict]:
    """First line of `path` matching `pattern` at the pinned commit, 1-indexed.

    `after` anchors the search past a preceding line, which is what separates two
    identically named methods in one file: `cuke-emlb`'s `evalEventStorePerTime` appears
    both in the unused median-filtering variant and in the official class, and only the
    second evidences the claim the paper makes.
    """

    text = blob(repo, path)
    if text is None:
        return None
    lines = text.splitlines()
    start = 0
    if after is not None:
        anchor = re.compile(after)
        found = next((i for i, line in enumerate(lines) if anchor.search(line)), None)
        if found is None:
            return None
        start = found
    compiled = re.compile(pattern)
    for number, line in enumerate(lines[start:], start + 1):
        if compiled.search(line):
            return {"repo": repo, "commit": PINNED[repo], "file": f"{repo}/{path}",
                    "line": number, "text": line.strip()}
    return None


def grep_repo(repo: str, pattern: str, pathspec: str = "*.py") -> Optional[dict]:
    """First match of `pattern` anywhere in `repo` at the pinned commit."""

    out = _git(repo, "grep", "-nE", pattern, PINNED[repo], "--", pathspec)
    if not out or not out.strip():
        return None
    _commit, path, number, text = out.splitlines()[0].split(":", 3)
    return {"repo": repo, "commit": PINNED[repo], "file": f"{repo}/{path}",
            "line": int(number), "text": text.strip()}


def grep_all(repo: str, pattern: str, pathspec: str = "*.py") -> List[dict]:
    """Every match of `pattern` in `repo` at the pinned commit."""

    out = _git(repo, "grep", "-nE", pattern, PINNED[repo], "--", pathspec)
    if not out or not out.strip():
        return []
    hits = []
    for row in out.splitlines():
        _commit, path, number, text = row.split(":", 3)
        hits.append({"file": f"{repo}/{path}", "line": int(number), "text": text.strip()})
    return hits


def tree(repo: str, path: str) -> Optional[List[str]]:
    """Names directly under `path` at the pinned commit, or None if it is not a tree."""

    out = _git(repo, "ls-tree", "--name-only", f"{PINNED[repo]}:{path}")
    return sorted(Path(name).name for name in out.split()) if out is not None else None


def main() -> None:
    checks = []

    hit = grep("EDmamba", "pointcept/engines/test.py", r"#\s*ESR\s*=")
    checks.append({"paper": "EDmamba", "table": "Table 2 (MESR)",
                   "claim": "the ESR computation is commented out in the released tester",
                   "verified": hit is not None, "evidence": hit})

    esr_defined = grep_repo("EDmamba", r"def compute_average_esr")
    checks.append({"paper": "EDmamba", "table": "Table 2 (MESR)",
                   "claim": "compute_average_esr is not defined anywhere in the repo",
                   "verified": esr_defined is None, "evidence": esr_defined})

    hit = grep("EDmamba", "pointcept/datasets/denoise_emlb.py", r"test\s*=\s*\[\"ND00\"\]")
    checks.append({"paper": "EDmamba", "table": "Table 2 (MESR)",
                   "claim": "the E-MLB reader evaluates ND00 only, not 4 ND levels",
                   "verified": hit is not None, "evidence": hit})

    hit = grep("EDmamba", "pointcept/datasets/denoise_emlb.py",
               r"labels\s*=\s*events_slice\[:,\s*3\]")
    checks.append({"paper": "EDmamba", "table": "Table 2 (MESR)",
                   "claim": "the E-MLB reader substitutes polarity for labels",
                   "verified": hit is not None, "evidence": hit})

    hit = grep("EDformer", "eval_mesr.py", r"0\.005")
    checks.append({"paper": "EDformer", "table": "Table 3 (MESR)",
                   "claim": "the operating point IS stated in the released code (0.005)",
                   "verified": hit is not None, "evidence": hit})

    # Locate the VARIANT, not the helper it calls. `median_filter` first matches its own
    # `def` at :12, which evidences none of the three properties the claim asserts and
    # sends a reader to a generic utility. The class declaration is the location the
    # sibling cuke-emlb row cites, and reading it shows the median filter, the division by
    # K and the 1000x rescale.
    hit = grep("EDformer", "metrics.py", r"class EventStructuralRatioV2")
    checks.append({"paper": "EDformer", "table": "Table 3 (MESR)",
                   "claim": "the repo ships an unused ESR variant that median-filters "
                            "and rescales by 1000 - a silent-error hazard",
                   "verified": hit is not None, "evidence": hit})

    # The same hazard in the benchmark suite both papers measure against, at the path this
    # project actually uses. cuke-emlb/metrics.py does not exist; the file is
    # cuke-emlb/python/src/utils/metric.py, and only V2 is unused there.
    hit = grep("cuke-emlb", "python/src/utils/metric.py", r"class EventStructuralRatioV2")
    checks.append({"paper": "E-MLB (cuke-emlb)", "table": "metric definition",
                   "claim": "cuke-emlb ships a median-filtering V2 variant beside the "
                            "official EventStructuralRatio, in python/src/utils/metric.py",
                   "verified": hit is not None, "evidence": hit})

    # Section VI-C's slice-size argument rests on this: the official class is delimited by
    # count through a caller-supplied argument, and the time-delimited entry point sitting
    # beside it is dead code in the release. Anchor past the official class declaration -
    # the unused V2 variant above defines a method of the same name, and citing that one
    # would evidence a claim about the variant rather than about the official class.
    entry = grep("cuke-emlb", "python/src/utils/metric.py",
                 r"def evalEventStorePerTime", after=r"^class EventStructuralRatio\(")
    callers = [h for h in grep_all("cuke-emlb", r"\.evalEventStorePerTime\(")
               if not h["file"].endswith("python/src/utils/metric.py")]
    checks.append({"paper": "E-MLB (cuke-emlb)", "table": "metric definition",
                   "claim": "a time-delimited entry point (33 ms default) ships beside the "
                            "count-delimited one, and no released driver calls it",
                   "verified": entry is not None and callers == [],
                   "evidence": dict(entry, callers=callers) if entry else None})

    hit = grep("cuke-emlb", "python/src/utils/metric.py",
               r"ln = K - \(\(1 - M / N\) \*\* n\)\.sum\(\)")
    checks.append({"paper": "E-MLB (cuke-emlb)", "table": "metric definition",
                   "claim": "the official ln subtracts from K, so empty pixels contribute "
                            "1 each and K cancels; ESR carries no W*H term",
                   "verified": hit is not None, "evidence": hit})

    # EDnCNN / MLPF cannot be run: the registry points at checkpoints that are not shipped.
    contents = tree("cuke-emlb", "modules/net")
    weights = [name for name in (contents or []) if name.endswith((".pt", ".pth", ".ckpt"))]
    checks.append({"paper": "E-MLB (cuke-emlb)", "table": "learned baselines",
                   "claim": "EDnCNN/MLPF weights are not in the release",
                   "verified": contents is not None and weights == [],
                   "evidence": {"repo": "cuke-emlb", "commit": PINNED["cuke-emlb"],
                                "dir": "cuke-emlb/modules/net", "contents": contents}})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"pinned_commits": PINNED, "accessed": ACCESSED,
                               "read_from": "git blob at the pinned commit, not the "
                                            "working copy",
                               "checks": checks}, indent=2) + "\n")
    passed = sum(1 for c in checks if c["verified"])
    print(f"{passed}/{len(checks)} claims verified from source at the pinned commits")
    for check in checks:
        evidence = check["evidence"] or {}
        where = (f"{evidence['file']}:{evidence['line']}"
                 if "line" in evidence else evidence.get("dir", "-"))
        print(f"  [{'OK ' if check['verified'] else 'ERR'}] {check['paper']}: "
              f"{check['claim']}\n         {where}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
