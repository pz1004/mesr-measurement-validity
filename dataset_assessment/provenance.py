"""What the real corpora's own file headers record about filtering before release.

Supplementary Appendix N states these facts, so they are read from the files here rather than
asserted:

E-MLB, Pure_BA  AEDAT-4 files written by iniVation's DV. The header's XML describes each
                recorded output: the module it came from (`originalModuleName`), its type, and
                the sensor size, camera serial and time offset. Any other attribute anywhere
                in that XML, a bias or a device filter setting, is listed under `other_keys`.
DVSD22          AEDAT-2 files written by jAER, whose `#` header embeds the preference store the
                recording was written under, including the camera's own DVS settings
                (`Davis346blue.DVS.*`): the on-device background-activity, refractory-period,
                polarity, event-skipping and row-only-event filters. The same store holds
                jAER's host-side filter chain (`FilterFrame.filters`, the class names jAER
                builds the chain from) and each filter's `<Class>.filterEnabled`, which jAER
                writes whenever a filter in the chain is switched on or off. jAER runs only the
                enabled members of the chain, so a filter flagged enabled outside it runs on
                nothing; those are reported apart.

A header records the configuration it was written with; it cannot show a setting changed
during a recording, and an AEDAT-4 header that records no configuration shows nothing either
way about the device's filters.

Writes `results/corpus_provenance.json`. Run:

    python -m dataset_assessment.provenance
"""

from __future__ import annotations

import base64
import json
import re
import struct
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .readers import PROJECT_ROOT

OUT = PROJECT_ROOT / "results/corpus_provenance.json"
EMLB = PROJECT_ROOT / "E-MLB"
PURE_BA = PROJECT_ROOT / "ECCV2024_datasets/Pure_BA_noise"
DVSD22 = PROJECT_ROOT / "DVSD22/Recordings"

#: The keys DV writes to describe a recorded output. Anything else in an AEDAT-4 header
#: would be configuration, and is reported rather than ignored.
OUTPUT_INFO_KEYS = frozenset({
    "compression", "originalModuleName", "originalOutputName", "typeDescription",
    "typeIdentifier", "colorFilter", "sizeX", "sizeY", "source", "tsOffset",
})
#: jAER's preference prefix for the camera these recordings name (`AEChip: ...Davis346blue`).
DVS_PREFIX = "Davis346blue.DVS."
#: The line jAER writes after the last header line. The first event can begin with the byte
#: `#` (it does in one DVSD22 recording), so the header ends here, not at the first line
#: lacking a `#`.
JAER_END = b"#End Of ASCII Header"
ENTRY = re.compile(r'<entry key="([^"]+)" value="([^"]*)"\s*/>')
#: The preference under which jAER stores the class names of a chip's filter chain.
CHAIN_KEY = "FilterFrame.filters"
#: How Java serialises a `java.util.ArrayList`: stream magic and version, a new object of a
#: new class descriptor, and the class name.
JAVA_ARRAYLIST = b"\xac\xed\x00\x05\x73\x72\x00\x13java.util.ArrayList"


def aedat4_header(path: Path, limit: int = 1 << 16) -> str:
    """The XML description at the head of an AEDAT-4 file, `<dv ...> ... </dv>`."""

    with path.open("rb") as handle:
        head = handle.read(limit)
    start, end = head.find(b"<dv"), head.find(b"</dv>")
    if start < 0 or end < 0:
        raise ValueError(f"{path}: no <dv> description in the first {limit} bytes")
    return head[start:end + len(b"</dv>")].decode("utf-8")


def aedat4_keys(xml: str) -> List[str]:
    """Every attribute key in the description, wherever its node sits."""

    return [a.get("key", "") for a in ET.fromstring(xml).iter("attr")]


def aedat4_outputs(xml: str) -> List[Dict[str, str]]:
    """One dict per recorded output: its attributes and those of its `info` node."""

    root = ET.fromstring(xml)
    outputs = []
    for info in root.iter("node"):
        if info.get("name") != "outInfo":
            continue
        for output in info.findall("node"):
            attributes = {a.get("key", ""): a.text or "" for a in output.findall("attr")}
            for child in output.findall("node"):
                attributes.update({a.get("key", ""): a.text or "" for a in child.findall("attr")})
            outputs.append(attributes)
    return outputs


def jaer_header(path: Path) -> List[str]:
    """The header lines of an AEDAT-2 file, decoded, through jAER's end-of-header line."""

    lines: List[str] = []
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.startswith(b"#"):
                raise ValueError(f"{path}: the header ends without {JAER_END.decode()!r}")
            lines.append(raw.decode("latin-1").rstrip("\r\n"))
            if raw.startswith(JAER_END):
                return lines
    raise ValueError(f"{path}: no {JAER_END.decode()!r} line")


def jaer_settings(lines: Iterable[str], prefix: str = DVS_PREFIX) -> Dict[str, str]:
    """Preference entries whose key starts with `prefix`, with the prefix removed."""

    settings: Dict[str, str] = {}
    for line in lines:
        match = ENTRY.search(line)
        if match and match.group(1).startswith(prefix):
            settings[match.group(1)[len(prefix):]] = match.group(2)
    return settings


def jaer_entries(lines: Iterable[str]) -> Dict[str, str]:
    """Every preference entry in the header, key to value."""

    return {m.group(1): m.group(2) for m in map(ENTRY.search, lines) if m}


def java_string_list(blob: bytes) -> List[str]:
    """The strings of a Java-serialised `java.util.ArrayList<String>`.

    Read field by field rather than searched, so a list in any other form, one that ends
    before its stated size or one holding anything but plain strings, is refused.
    """

    try:
        if not blob.startswith(JAVA_ARRAYLIST):
            raise ValueError("not a serialised java.util.ArrayList")
        i = len(JAVA_ARRAYLIST) + 8 + 1                    # serialVersionUID, flags
        (fields,) = struct.unpack_from(">H", blob, i)
        i += 2
        for _ in range(fields):                            # primitive field descriptors
            if blob[i:i + 1] not in (b"I", b"J", b"S", b"B", b"Z", b"C", b"F", b"D"):
                raise ValueError("unexpected field descriptor")
            (length,) = struct.unpack_from(">H", blob, i + 1)
            i += 3 + length
        if blob[i:i + 2] != b"\x78\x70":                     # end of class, no superclass
            raise ValueError("unexpected class annotation or superclass")
        (size,) = struct.unpack_from(">i", blob, i + 2)
        if blob[i + 6] != 0x77:                            # block data: the capacity
            raise ValueError("missing block data")
        i += 8 + blob[i + 7]
        names = []
        for _ in range(size):
            if blob[i] != 0x74:
                raise ValueError(f"element tag {blob[i]:#04x} is not a string")
            (length,) = struct.unpack_from(">H", blob, i + 1)
            if i + 3 + length > len(blob):
                raise ValueError("string runs past the end of the list")
            names.append(blob[i + 3:i + 3 + length].decode("utf-8"))
            i += 3 + length
        if blob[i:] != b"\x78":
            raise ValueError("the list does not end where its size says")
    except (IndexError, struct.error) as exc:
        raise ValueError(f"truncated serialised list: {exc}") from exc
    return names


def jaer_chain(lines: Iterable[str]) -> Optional[List[str]]:
    """The class names of the stored filter chain, or None when the header stores none."""

    stored = jaer_entries(lines).get(CHAIN_KEY)
    return None if stored is None else java_string_list(base64.b64decode(stored))


def jaer_field(lines: Iterable[str], name: str) -> str:
    """The value of a `# <name>: <value>` header line, or '' when absent."""

    marker = f"# {name}:"
    return next((line[len(marker):].strip() for line in lines if line.startswith(marker)), "")


def summarise_aedat4(paths: Sequence[Path]) -> Dict:
    modules: Counter = Counter()
    other: Counter = Counter()
    cameras: Counter = Counter()
    files_capture_only = 0
    for path in paths:
        xml = aedat4_header(path)
        outputs = aedat4_outputs(xml)
        names = {o.get("originalModuleName", "") for o in outputs}
        modules.update(o.get("originalModuleName", "") for o in outputs)
        other.update(k for k in aedat4_keys(xml) if k not in OUTPUT_INFO_KEYS)
        cameras.update({o["source"] for o in outputs if "source" in o})
        files_capture_only += names == {"capture"}
    return {"files": len(paths), "format": "AEDAT-4 (DV)",
            "files_whose_outputs_all_come_from_capture": files_capture_only,
            "original_module_names": dict(modules), "other_keys": dict(other),
            "cameras": dict(cameras)}


def summarise_host_chain(headers: Sequence[List[str]], sessions: Sequence[str]) -> Dict:
    """jAER's stored filter chain per file, and which filters were flagged enabled.

    `sessions[i]` names the capture session of `headers[i]` (its folder in the release).
    """

    chains: Counter = Counter()
    chain_sessions: Dict[tuple, Counter] = {}
    in_chain: Counter = Counter()
    outside: Counter = Counter()
    outside_per_file: Counter = Counter()
    unrecorded = idle = 0
    if len(headers) != len(sessions):
        raise ValueError("one session name per header")
    for lines, session in zip(headers, sessions):
        chain = jaer_chain(lines)
        if chain is None:
            unrecorded += 1
            continue
        members = [name.rsplit(".", 1)[-1] for name in chain]
        enabled = {key[:-len(".filterEnabled")] for key, value in jaer_entries(lines).items()
                   if key.endswith(".filterEnabled") and value == "true"}
        chains[tuple(members)] += 1
        chain_sessions.setdefault(tuple(members), Counter())[session] += 1
        running = [name for name in members if name in enabled]
        in_chain.update(running)
        idle += not running
        rest = enabled.difference(members)
        outside.update(rest)
        outside_per_file[len(rest)] += 1
    return {"chains": [{"filters": list(c), "files": n, "sessions": dict(chain_sessions[c])}
                       for c, n in chains.most_common()],
            "enabled_in_chain": dict(in_chain),
            "files_with_no_enabled_filter_in_chain": idle,
            "enabled_outside_chain": dict(sorted(outside.items())),
            "enabled_outside_chain_per_file": {str(k): v for k, v in sorted(outside_per_file.items())},
            "files_without_stored_chain": unrecorded}


def summarise_dvsd22(paths: Sequence[Path]) -> Dict:
    settings: Dict[str, Counter] = {}
    interfaces: Counter = Counter()
    chips: Counter = Counter()
    headers = [jaer_header(path) for path in paths]
    for lines in headers:
        interfaces[jaer_field(lines, "HardwareInterface")] += 1
        chips[jaer_field(lines, "AEChip")] += 1
        for key, value in jaer_settings(lines).items():
            settings.setdefault(key, Counter())[value] += 1
    filters = {key: dict(values) for key, values in sorted(settings.items())
               if key.startswith("Filter")}
    return {"files": len(paths), "format": "AEDAT-2 (jAER)",
            "used_here": sum("calibration" not in p.name for p in paths),
            "hardware_interfaces": dict(interfaces), "chips": dict(chips),
            "device_filters": filters,
            "host_filter_chain": summarise_host_chain(headers, [p.parent.name for p in paths])}


def main() -> None:
    emlb = sorted(EMLB.glob("*/*/*.aedat4"))
    pure_ba = sorted(PURE_BA.glob("*.aedat4"))
    dvsd22 = sorted(DVSD22.glob("*/*.aedat"))
    if not (emlb and pure_ba and dvsd22):
        raise SystemExit("a corpus is missing; see DATA.md for the expected layout")
    report = {"E-MLB": summarise_aedat4(emlb), "Pure_BA": summarise_aedat4(pure_ba),
              "DVSD22": summarise_dvsd22(dvsd22)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for corpus, block in report.items():
        print(f"{corpus}: {block['files']} files")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
