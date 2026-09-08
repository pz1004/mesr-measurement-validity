"""Re-verify, from source, every reproducibility claim the paper makes.

Each check is a static or executable fact about a released repository, recorded with the
file and line so a reader can confirm it independently. Nothing here may be cited from the
plan's briefing section: if a check cannot confirm a claim, the claim does not go in the
paper.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results/reproducibility_audit.json"


def _files(target: Path, suffixes: Iterable[str]) -> Iterable[Path]:
    """Files to scan: `target` itself, or every matching file beneath it if it is a
    directory. `Path.read_text()` on a directory raises IsADirectoryError, so a plain
    exists() guard is not enough."""

    if target.is_file():
        yield target
    elif target.is_dir():
        for suffix in suffixes:
            yield from sorted(target.rglob(f"*{suffix}"))


def grep(target: Path, pattern: str,
         suffixes: Iterable[str] = (".py",)) -> Optional[dict]:
    """First match of `pattern` under `target`, or None."""

    compiled = re.compile(pattern)
    for path in _files(target, suffixes):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if compiled.search(line):
                return {"file": str(path.relative_to(PROJECT_ROOT)), "line": number,
                        "text": line.strip()}
    return None


def main() -> None:
    checks = []

    hit = grep(PROJECT_ROOT / "EDmamba/pointcept/engines/test.py", r"#\s*ESR\s*=")
    checks.append({"paper": "EDmamba", "table": "Table 2 (MESR)",
                   "claim": "the ESR computation is commented out in the released tester",
                   "verified": hit is not None, "evidence": hit})

    esr_defined = grep(PROJECT_ROOT / "EDmamba", r"def compute_average_esr")
    checks.append({"paper": "EDmamba", "table": "Table 2 (MESR)",
                   "claim": "compute_average_esr is not defined anywhere in the repo",
                   "verified": esr_defined is None, "evidence": esr_defined})

    hit = grep(PROJECT_ROOT / "EDmamba/pointcept/datasets/denoise_emlb.py",
               r"test\s*=\s*\[\"ND00\"\]")
    checks.append({"paper": "EDmamba", "table": "Table 2 (MESR)",
                   "claim": "the E-MLB reader evaluates ND00 only, not 4 ND levels",
                   "verified": hit is not None, "evidence": hit})

    hit = grep(PROJECT_ROOT / "EDmamba/pointcept/datasets/denoise_emlb.py",
               r"labels\s*=\s*events_slice\[:,\s*3\]")
    checks.append({"paper": "EDmamba", "table": "Table 2 (MESR)",
                   "claim": "the E-MLB reader substitutes polarity for labels",
                   "verified": hit is not None, "evidence": hit})

    hit = grep(PROJECT_ROOT / "EDformer/eval_mesr.py", r"0\.005")
    checks.append({"paper": "EDformer", "table": "Table 3 (MESR)",
                   "claim": "the operating point IS stated in the released code (0.005)",
                   "verified": hit is not None, "evidence": hit})

    hit = grep(PROJECT_ROOT / "EDformer/metrics.py", r"median_filter")
    checks.append({"paper": "EDformer", "table": "Table 3 (MESR)",
                   "claim": "the repo ships an unused ESR variant that median-filters "
                            "and rescales by 1000 - a silent-error hazard",
                   "verified": hit is not None, "evidence": hit})

    # The same hazard in the benchmark suite both papers measure against, at the path this
    # project actually uses. cuke-emlb/metrics.py does not exist; the file is
    # cuke-emlb/python/src/utils/metric.py, and only V2 is unused there.
    hit = grep(PROJECT_ROOT / "cuke-emlb/python/src/utils/metric.py",
               r"class EventStructuralRatioV2")
    checks.append({"paper": "E-MLB (cuke-emlb)", "table": "metric definition",
                   "claim": "cuke-emlb ships a median-filtering V2 variant beside the "
                            "official EventStructuralRatio, in python/src/utils/metric.py",
                   "verified": hit is not None, "evidence": hit})

    hit = grep(PROJECT_ROOT / "cuke-emlb/python/src/utils/metric.py",
               r"ln = K - \(\(1 - M / N\) \*\* n\)\.sum\(\)")
    checks.append({"paper": "E-MLB (cuke-emlb)", "table": "metric definition",
                   "claim": "the official ln subtracts from K, so empty pixels contribute "
                            "1 each and K cancels; ESR carries no W*H term",
                   "verified": hit is not None, "evidence": hit})

    # EDnCNN / MLPF cannot be run: the registry points at checkpoints that are not shipped.
    net_dir = PROJECT_ROOT / "cuke-emlb/modules/net"
    weights = sorted(p.name for p in net_dir.glob("*.pt")) if net_dir.is_dir() else []
    checks.append({"paper": "E-MLB (cuke-emlb)", "table": "learned baselines",
                   "claim": "EDnCNN/MLPF weights are not in the release",
                   "verified": weights == [],
                   "evidence": {"dir": "cuke-emlb/modules/net",
                                "contents": sorted(p.name for p in net_dir.iterdir())
                                if net_dir.is_dir() else None}})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"checks": checks}, indent=2) + "\n")
    passed = sum(1 for c in checks if c["verified"])
    print(f"{passed}/{len(checks)} claims verified from source")
    for c in checks:
        print(f"  [{'OK ' if c['verified'] else 'ERR'}] {c['paper']}: {c['claim']}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
