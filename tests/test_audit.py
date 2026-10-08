"""The source audit must fail closed.

Two of its claims are negative: a function defined nowhere, an entry point nobody calls. A
negative claim is only as good as the search behind it, so a git command that could not run
must never read as a search that found nothing. These tests stand in for git with canned
exit statuses; the real checkouts are exercised only by the last test, and only where they
exist. No test writes the shipped artifact.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from dataset_assessment import audit


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(args=(), returncode=returncode, stdout=stdout,
                                       stderr=stderr)


def test_a_search_that_could_not_run_raises(monkeypatch):
    monkeypatch.setattr(audit.subprocess, "run",
                        lambda *a, **k: _completed(128, stderr="fatal: bad revision"))
    with pytest.raises(audit.AuditError, match="bad revision"):
        audit.grep_repo("EDmamba", r"def compute_average_esr")
    with pytest.raises(audit.AuditError):
        audit.grep_all("cuke-emlb", r"\.evalEventStorePerTime\(")


def test_exit_1_with_an_error_is_not_a_search_result(monkeypatch):
    """An unreadable blob makes `git grep` exit 1, the "no match" status, with an error on
    stderr. Only the stderr check tells the two apart."""

    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **k: _completed(
        1, stderr="error: '302f758:x.py': unable to read 9332a27"))
    with pytest.raises(audit.AuditError):
        audit.grep_repo("EDmamba", r"def compute_average_esr")
    with pytest.raises(audit.AuditError):
        audit.grep_all("cuke-emlb", r"\.evalEventStorePerTime\(")


def test_a_match_reported_without_output_is_an_error(monkeypatch):
    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **k: _completed(0, stdout=""))
    with pytest.raises(audit.AuditError):
        audit.grep_repo("EDmamba", r"def compute_average_esr")


def test_pinned_commit_check_never_accepts_exit_1(monkeypatch):
    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **k: _completed(1))
    with pytest.raises(audit.AuditError):
        audit.check_pinned()


def test_a_checkout_reached_below_its_top_level_is_refused(monkeypatch):
    """`git grep` searches only below where it runs, so a subdirectory checkout would narrow
    a negative search without failing it."""

    def run(command, **kwargs):
        return _completed(0, stdout="sub/\n" if "--show-prefix" in command else "")

    monkeypatch.setattr(audit.subprocess, "run", run)
    with pytest.raises(audit.AuditError, match="top level"):
        audit.check_pinned()


def test_a_search_that_matched_nothing_is_an_answer(monkeypatch):
    """`git grep` exits 1, silently, when no line matches: the one failure that is a result."""

    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **k: _completed(1))
    assert audit.grep_repo("EDmamba", r"def compute_average_esr") is None
    assert audit.grep_all("cuke-emlb", r"\.evalEventStorePerTime\(") == []


def test_a_match_holding_a_form_feed_stays_one_line(monkeypatch):
    """`-a` prints binary matches as text; only git's own newline ends a record."""

    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **k: _completed(
        0, stdout="302f758:x.py:3:a\x0cb\n302f758:y.py:9:c\n"))
    hits = audit.grep_all("cuke-emlb", r"\.evalEventStorePerTime\(")
    assert [(h["file"], h["line"], h["text"]) for h in hits] == [
        ("cuke-emlb/x.py", 3, "a\x0cb"), ("cuke-emlb/y.py", 9, "c")]


def test_a_missing_file_at_the_commit_is_not_a_missing_line(monkeypatch):
    monkeypatch.setattr(audit.subprocess, "run",
                        lambda *a, **k: _completed(128, stderr="fatal: path does not exist"))
    with pytest.raises(audit.AuditError):
        audit.grep("EDformer", "eval_mesr.py", r"0\.005")


def _artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "OUT", tmp_path / "audit.json")
    monkeypatch.setattr(audit, "FAILED", tmp_path / "audit.failed.json")
    return tmp_path / "audit.json", tmp_path / "audit.failed.json"


def test_an_unresolvable_pinned_commit_stops_the_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(audit.subprocess, "run",
                        lambda *a, **k: _completed(128, stderr="fatal: Not a valid object"))
    out, failed = _artifacts(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as stopped:
        audit.main()
    assert stopped.value.code not in (0, None)
    assert not out.exists() and not failed.exists()


def test_an_uninspectable_source_leaves_the_claim_unverified(monkeypatch, tmp_path):
    """Commits resolve, every read fails: the negative claim must not come out verified,
    and the run must exit non-zero."""

    def run(command, **kwargs):
        if "cat-file" in command or "--show-prefix" in command:
            return _completed(0)
        return _completed(128, stderr="fatal: unreadable object store")

    monkeypatch.setattr(audit.subprocess, "run", run)
    out, failed = _artifacts(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as stopped:
        audit.main()
    assert stopped.value.code == 1
    assert not out.exists(), "a failing run must not overwrite the shipped artifact"
    checks = json.loads(failed.read_text())["checks"]
    negative = next(c for c in checks if "not defined anywhere" in c["claim"])
    assert negative["verified"] is False and "unreadable" in negative["error"]
    assert not any(c["verified"] for c in checks)


def test_the_audit_passes_at_the_pinned_commits(monkeypatch, tmp_path):
    if not all((audit.PROJECT_ROOT / repo).exists() for repo in audit.PINNED):
        pytest.skip("needs the EDmamba, EDformer and cuke-emlb checkouts")
    out, failed = _artifacts(monkeypatch, tmp_path)
    failed.write_text("{}\n")                   # left by an earlier failing run
    with pytest.raises(SystemExit) as stopped:
        audit.main()
    assert stopped.value.code == 0
    assert not failed.exists()
    shipped = audit.PROJECT_ROOT / "results/reproducibility_audit.json"
    assert out.read_bytes() == shipped.read_bytes()
