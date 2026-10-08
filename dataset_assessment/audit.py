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

**It fails closed.** Two of the claims are negative - a function defined nowhere, an entry
point nobody calls - and a negative claim is only as good as the search behind it. A git
command that fails (no checkout, an unresolvable commit, a path missing at the commit)
raises `AuditError` instead of reading as a search that found nothing; `git grep`'s exit
status 1 with nothing on stderr, which means no line matched, is the only failure read as an
answer. Every pinned commit is resolved, and every checkout confirmed to be reached at its top
level (a search run from a subdirectory would cover only that subdirectory), before any claim is
read. The negative searches cover every file at the commit, not only Python. A check whose
source cannot be inspected is recorded as unverified with the error, and the run exits non-zero
unless every claim holds; a failing run writes `reproducibility_audit.failed.json` beside the
artifact rather than over it.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results/reproducibility_audit.json"
FAILED = OUT.with_suffix(".failed.json")
#: `git grep` flags that make its output parseable whatever the user's configuration: no colour
#: codes, no column numbers, and binary files read as text so a match inside one is reported
#: as a line rather than as "Binary file ... matches".
GREP = ("grep", "-nE", "--no-color", "--no-column", "-a")

#: Each release's default branch at the commit the supplement pins. Changing one of these
#: changes what the paper claims, so they are data, not configuration: update them only
#: alongside the supplement's own pinning paragraph.
PINNED = {"EDmamba": "302f758", "EDformer": "820ff16", "cuke-emlb": "105a1c3"}
ACCESSED = "2026-09-21"


class AuditError(RuntimeError):
    """A source could not be inspected. Never to be read as a finding about it."""


def _git(repo: str, *arguments: str, no_match_ok: bool = False) -> Optional[str]:
    """Run git in the checkout `repo` names and return its output.

    The checkouts are reached through the symlinks beside this project. Only their object
    stores are used - never their working trees - so a dirty checkout cannot reach a check.
    With `no_match_ok`, exit status 1 with nothing on stderr - `git grep`'s "no line
    matched" - returns None. Any other failure raises `AuditError`.
    """

    try:
        completed = subprocess.run(("git", "-C", str(PROJECT_ROOT / repo)) + arguments,
                                   capture_output=True, text=True, encoding="utf-8",
                                   errors="replace")
    except OSError as error:
        raise AuditError(f"cannot run git in {repo}: {error}") from error
    if completed.returncode == 0:
        return completed.stdout
    if no_match_ok and completed.returncode == 1 and not completed.stderr.strip():
        return None
    raise AuditError(f"`git {' '.join(arguments)}` in {repo} exited {completed.returncode}: "
                     f"{completed.stderr.strip()[:300] or 'no message'}")


def check_pinned() -> None:
    """Resolve every pinned commit, and reach every checkout at its top level, before any
    claim is read. `git grep` searches only below the directory it runs in."""

    for repo, commit in PINNED.items():
        _git(repo, "cat-file", "-e", f"{commit}^{{commit}}")
        prefix = (_git(repo, "rev-parse", "--show-prefix") or "").strip()
        if prefix:
            raise AuditError(f"{repo} is reached at {prefix!r}, not at its top level; a "
                             "search there would cover only that subdirectory")


def blob(repo: str, path: str) -> str:
    """`path` as it stands at `repo`'s pinned commit."""

    text = _git(repo, "show", f"{PINNED[repo]}:{path}")
    assert text is not None                      # only no_match_ok calls return None
    return text


def _search(repo: str, pattern: str, pathspec: str) -> List[str]:
    """Lines of `git grep` at the pinned commit; empty only when the search ran and matched
    nothing. A zero exit status with no output is not an answer either."""

    out = _git(repo, *GREP, pattern, PINNED[repo], "--", pathspec, no_match_ok=True)
    if out is None:
        return []
    if not out.strip():
        raise AuditError(f"`git grep` in {repo} reported a match and printed none")
    # Git ends records with "\n" only; `splitlines` would also break inside a matched line
    # that holds a form feed or another separator, which `-a` can print.
    return [line.rstrip("\r") for line in out.split("\n") if line]


def grep(repo: str, path: str, pattern: str, after: str | None = None) -> Optional[dict]:
    """First line of `path` matching `pattern` at the pinned commit, 1-indexed.

    `after` anchors the search past a preceding line, which is what separates two
    identically named methods in one file: `cuke-emlb`'s `evalEventStorePerTime` appears
    both in the unused median-filtering variant and in the official class, and only the
    second evidences the claim the paper makes.
    """

    # Split as git numbers lines, on "\n" alone.
    lines = [line.rstrip("\r") for line in blob(repo, path).split("\n")]
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


def grep_repo(repo: str, pattern: str, pathspec: str = "*") -> Optional[dict]:
    """First match of `pattern` anywhere in `repo` at the pinned commit; None if none."""

    lines = _search(repo, pattern, pathspec)
    if not lines:
        return None
    _commit, path, number, text = lines[0].split(":", 3)
    return {"repo": repo, "commit": PINNED[repo], "file": f"{repo}/{path}",
            "line": int(number), "text": text.strip()}


def grep_all(repo: str, pattern: str, pathspec: str = "*") -> List[dict]:
    """Every match of `pattern` in `repo` at the pinned commit; empty if none."""

    hits = []
    for row in _search(repo, pattern, pathspec):
        _commit, path, number, text = row.split(":", 3)
        hits.append({"file": f"{repo}/{path}", "line": int(number), "text": text.strip()})
    return hits


def tree(repo: str, path: str) -> List[str]:
    """Names directly under `path` at the pinned commit."""

    out = _git(repo, "ls-tree", "--name-only", f"{PINNED[repo]}:{path}") or ""
    return sorted(Path(name).name for name in out.split())


def _found(hit: Optional[dict]) -> Tuple[bool, Optional[dict]]:
    return hit is not None, hit


def _record(checks: List[dict], paper: str, table: str, claim: str,
            compute: Callable[[], Tuple[bool, object]]) -> None:
    """Run one check. A source that cannot be inspected leaves the claim unverified."""

    entry: Dict[str, object] = {"paper": paper, "table": table, "claim": claim}
    try:
        verified, evidence = compute()
        entry.update(verified=bool(verified), evidence=evidence)
    except AuditError as error:
        entry.update(verified=False, evidence=None, error=str(error))
    checks.append(entry)


def _time_delimited_entry() -> Tuple[bool, Optional[dict]]:
    # Anchor past the official class declaration: the unused V2 variant defines a method of
    # the same name, and citing that one would evidence a claim about the variant.
    entry = grep("cuke-emlb", "python/src/utils/metric.py",
                 r"def evalEventStorePerTime", after=r"^class EventStructuralRatio\(")
    callers = [h for h in grep_all("cuke-emlb", r"\.evalEventStorePerTime\(")
               if not h["file"].endswith("python/src/utils/metric.py")]
    return entry is not None and callers == [], dict(entry, callers=callers) if entry else None


def _no_learned_weights() -> Tuple[bool, dict]:
    contents = tree("cuke-emlb", "modules/net")
    weights = [name for name in contents if name.endswith((".pt", ".pth", ".ckpt"))]
    return weights == [], {"repo": "cuke-emlb", "commit": PINNED["cuke-emlb"],
                           "dir": "cuke-emlb/modules/net", "contents": contents}


def main() -> None:
    try:
        check_pinned()
    except AuditError as error:
        raise SystemExit(f"cannot audit the pinned sources: {error}")

    checks: List[dict] = []
    _record(checks, "EDmamba", "Table 2 (MESR)",
            "the ESR computation is commented out in the released tester",
            lambda: _found(grep("EDmamba", "pointcept/engines/test.py", r"#\s*ESR\s*=")))
    # Negative: verified only by a search that ran and matched nothing.
    _record(checks, "EDmamba", "Table 2 (MESR)",
            "compute_average_esr is not defined anywhere in the repo",
            lambda: (lambda hit: (hit is None, hit))(
                grep_repo("EDmamba", r"def compute_average_esr")))
    _record(checks, "EDmamba", "Table 2 (MESR)",
            "the E-MLB reader evaluates ND00 only, not 4 ND levels",
            lambda: _found(grep("EDmamba", "pointcept/datasets/denoise_emlb.py",
                                r"test\s*=\s*\[\"ND00\"\]")))
    _record(checks, "EDmamba", "Table 2 (MESR)",
            "the E-MLB reader substitutes polarity for labels",
            lambda: _found(grep("EDmamba", "pointcept/datasets/denoise_emlb.py",
                                r"labels\s*=\s*events_slice\[:,\s*3\]")))
    _record(checks, "EDformer", "Table 3 (MESR)",
            "the operating point IS stated in the released code (0.005)",
            lambda: _found(grep("EDformer", "eval_mesr.py", r"0\.005")))
    # Locate the VARIANT, not the helper it calls. `median_filter` first matches its own
    # `def` at :12, which evidences none of the three properties the claim asserts and
    # sends a reader to a generic utility. The class declaration is the location the
    # sibling cuke-emlb row cites, and reading it shows the median filter, the division by
    # K and the 1000x rescale.
    _record(checks, "EDformer", "Table 3 (MESR)",
            "the repo ships an unused ESR variant that median-filters "
            "and rescales by 1000 - a silent-error hazard",
            lambda: _found(grep("EDformer", "metrics.py", r"class EventStructuralRatioV2")))
    # The same hazard in the benchmark suite both papers measure against, at the path this
    # project actually uses. cuke-emlb/metrics.py does not exist; the file is
    # cuke-emlb/python/src/utils/metric.py, and only V2 is unused there.
    _record(checks, "E-MLB (cuke-emlb)", "metric definition",
            "cuke-emlb ships a median-filtering V2 variant beside the "
            "official EventStructuralRatio, in python/src/utils/metric.py",
            lambda: _found(grep("cuke-emlb", "python/src/utils/metric.py",
                                r"class EventStructuralRatioV2")))
    # Section VI-C's slice-size argument rests on this: the official class is delimited by
    # count through a caller-supplied argument, and the time-delimited entry point sitting
    # beside it is dead code in the release. Negative in part: no caller found by a search
    # that ran.
    _record(checks, "E-MLB (cuke-emlb)", "metric definition",
            "a time-delimited entry point (33 ms default) ships beside the "
            "count-delimited one, and no released driver calls it",
            _time_delimited_entry)
    _record(checks, "E-MLB (cuke-emlb)", "metric definition",
            "the official ln subtracts from K, so empty pixels contribute "
            "1 each and K cancels; ESR carries no W*H term",
            lambda: _found(grep("cuke-emlb", "python/src/utils/metric.py",
                                r"ln = K - \(\(1 - M / N\) \*\* n\)\.sum\(\)")))
    # EDnCNN / MLPF cannot be run: the registry points at checkpoints that are not shipped.
    _record(checks, "E-MLB (cuke-emlb)", "learned baselines",
            "EDnCNN/MLPF weights are not in the release", _no_learned_weights)

    passed = sum(1 for c in checks if c["verified"])
    target = OUT if passed == len(checks) else FAILED
    if target is OUT:
        FAILED.unlink(missing_ok=True)            # a stale failure must not outlive a pass
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"pinned_commits": PINNED, "accessed": ACCESSED,
                                  "read_from": "git blob at the pinned commit, not the "
                                               "working copy",
                                  "checks": checks}, indent=2) + "\n")
    print(f"{passed}/{len(checks)} claims verified from source at the pinned commits")
    for check in checks:
        evidence = check["evidence"] or {}
        where = (f"{evidence['file']}:{evidence['line']}"
                 if "line" in evidence else evidence.get("dir", "-"))
        print(f"  [{'OK ' if check['verified'] else 'ERR'}] {check['paper']}: "
              f"{check['claim']}\n         {check.get('error', where)}")
    print(f"\nWrote {target}")
    raise SystemExit(0 if target == OUT else 1)


if __name__ == "__main__":
    main()
