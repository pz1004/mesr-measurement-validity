"""The header parsers report what a file records, and an unexpected key is not ignored."""

from __future__ import annotations

import base64
import json
import struct

import pytest

from dataset_assessment.provenance import (JAER_END, OUT, aedat4_header, aedat4_outputs,
                                           jaer_chain, jaer_field, jaer_header, jaer_settings,
                                           java_string_list, summarise_aedat4, summarise_dvsd22)

XML = ('<dv version="2.0"><node name="outInfo" path="/mainloop/Recorder/outInfo/">'
       '<node name="0" path="/mainloop/Recorder/outInfo/0/">'
       '<attr key="compression" type="string">LZ4</attr>'
       '<attr key="originalModuleName" type="string">capture</attr>'
       '<attr key="originalOutputName" type="string">events</attr>'
       '<node name="info" path="/mainloop/Recorder/outInfo/0/info/">'
       '<attr key="sizeX" type="int">346</attr>'
       '<attr key="source" type="string">DAVIS346_1</attr>'
       '</node></node></node></dv>')


def _aedat4(tmp_path, xml: str = XML, name: str = "a.aedat4"):
    path = tmp_path / name
    path.write_bytes(b"#!AER-DAT4.0\r\n\x10\x00\x00\x00" + xml.encode() + b"\x00\xff" * 64)
    return path


def test_an_aedat4_output_names_its_module_and_description_only(tmp_path):
    path = _aedat4(tmp_path)
    assert aedat4_outputs(aedat4_header(path)) == [{
        "compression": "LZ4", "originalModuleName": "capture", "originalOutputName": "events",
        "sizeX": "346", "source": "DAVIS346_1"}]
    summary = summarise_aedat4([path])
    assert summary["files_whose_outputs_all_come_from_capture"] == 1
    assert summary["other_keys"] == {}


def test_an_unexpected_aedat4_key_is_reported(tmp_path):
    xml = XML.replace('<attr key="sizeX"',
                      '<attr key="BackgroundActivityFilter" type="bool">true</attr>'
                      '<attr key="sizeX"')
    assert summarise_aedat4([_aedat4(tmp_path, xml)])["other_keys"] == {
        "BackgroundActivityFilter": 1}


HEADER = ("#!AER-DAT2.0\r\n"
          "# HardwareInterface: DAVIS BLUE 346 1\r\n"
          "# AEChip: eu.seebetter.ini.chips.davis.Davis346blue\r\n"
          '#   <entry key="Davis346blue.DVS.FilterBackgroundActivity" value="false"/>\r\n'
          '#   <entry key="DAVIS240C.DVS.FilterBackgroundActivity" value="true"/>\r\n'
          "#End Of ASCII Header\r\n")


def test_an_aedat4_key_outside_the_output_description_is_reported(tmp_path):
    xml = XML.replace("</dv>", '<node name="device" path="/mainloop/capture/">'
                               '<attr key="BackgroundActivityFilter" type="bool">true</attr>'
                               "</node></dv>")
    assert summarise_aedat4([_aedat4(tmp_path, xml)])["other_keys"] == {
        "BackgroundActivityFilter": 1}


def test_a_jaer_header_ends_at_its_marker_even_when_the_first_event_starts_with_a_hash(tmp_path):
    path = tmp_path / "Davis346blue_2021-07-19_2x30Hz_a.aedat"
    path.write_bytes(HEADER.encode() + b"#\x00\x01\x02 binary\n" + b"\x00" * 16)
    lines = jaer_header(path)
    assert len(lines) == 6 and lines[-1].startswith(JAER_END.decode())


def test_a_jaer_header_without_its_marker_is_refused(tmp_path):
    path = tmp_path / "broken.aedat"
    path.write_bytes(HEADER.replace("#End Of ASCII Header\r\n", "").encode() + b"\x00\x01\n")
    with pytest.raises(ValueError):
        jaer_header(path)


def test_a_jaer_header_reads_this_cameras_settings_only(tmp_path):
    path = tmp_path / "Davis346blue_2021-07-19_2x45Hz.aedat"
    path.write_bytes(HEADER.encode() + b"\x00\x01\x02#not a header line\n")
    lines = jaer_header(path)
    assert len(lines) == 6
    assert jaer_field(lines, "HardwareInterface") == "DAVIS BLUE 346 1"
    # Another chip's stored preference is not this camera's setting.
    assert jaer_settings(lines) == {"FilterBackgroundActivity": "false"}
    summary = summarise_dvsd22([path])
    assert summary["device_filters"] == {"FilterBackgroundActivity": {"false": 1}}
    assert summary["used_here"] == 1


def _java_list(names) -> bytes:
    """`names` as Java serialises a `java.util.ArrayList<String>`, the form jAER stores."""

    body = bytearray(b"\xac\xed\x00\x05\x73\x72\x00\x13java.util.ArrayList")
    body += bytes.fromhex("7881d21d99c7619d") + b"\x03\x00\x01I\x00\x04size\x78\x70"
    body += struct.pack(">i", len(names)) + b"\x77\x04" + struct.pack(">i", len(names))
    for name in names:
        body += b"\x74" + struct.pack(">H", len(name)) + name.encode()
    return bytes(body + b"\x78")


CHAIN = ["net.sf.jaer.eventprocessing.filter.BackgroundActivityFilter",
         "ch.unizh.ini.jaer.projects.minliu.Speedometer"]


def _chain_header(chain=CHAIN, enabled=("Speedometer", "AerCorrFilter")) -> str:
    stored = base64.b64encode(_java_list(chain)).decode()
    entries = [f'#   <entry key="FilterFrame.filters" value="{stored}"/>']
    entries += [f'#   <entry key="{name}.filterEnabled" value="{str(name in enabled).lower()}"/>'
                for name in ("BackgroundActivityFilter", "Speedometer", "AerCorrFilter")]
    return HEADER.replace("#End Of ASCII Header", "\r\n".join(entries) + "\r\n#End Of ASCII Header")


def test_a_serialised_string_list_round_trips():
    assert java_string_list(_java_list(CHAIN)) == CHAIN
    assert java_string_list(_java_list([])) == []


def test_a_malformed_string_list_is_refused():
    blob = _java_list(CHAIN)
    with pytest.raises(ValueError):
        java_string_list(blob.replace(b"java.util.ArrayList", b"java.util.HashSetXX"))
    for cut in (1, 10):                      # ends before its stated size
        with pytest.raises(ValueError):
            java_string_list(blob[:-cut])
    first = blob.index(b"\x77\x04") + 6      # the first element's tag
    with pytest.raises(ValueError):
        java_string_list(blob[:first] + b"\x71" + blob[first + 1:])   # a back-reference


def test_a_jaer_header_records_its_filter_chain_and_which_members_run(tmp_path):
    path = tmp_path / "Davis346blue_2021-07-19_2x45Hz.aedat"
    path.write_bytes(_chain_header().encode() + b"\x00\x01")
    assert jaer_chain(jaer_header(path)) == CHAIN
    host = summarise_dvsd22([path])["host_filter_chain"]
    assert host["chains"] == [{"filters": ["BackgroundActivityFilter", "Speedometer"], "files": 1,
                               "sessions": {tmp_path.name: 1}}]
    assert host["enabled_in_chain"] == {"Speedometer": 1}
    # Flagged enabled but outside the chain, so jAER never runs it: counted apart.
    assert host["enabled_outside_chain"] == {"AerCorrFilter": 1}
    assert host["files_with_no_enabled_filter_in_chain"] == 0
    assert host["files_without_stored_chain"] == 0


def test_a_header_without_a_stored_chain_is_counted_not_guessed(tmp_path):
    path = tmp_path / "Davis346blue_2021-07-19_2x45Hz.aedat"
    path.write_bytes(HEADER.encode() + b"\x00\x01")
    assert jaer_chain(jaer_header(path)) is None
    host = summarise_dvsd22([path])["host_filter_chain"]
    assert host["files_without_stored_chain"] == 1 and host["chains"] == []


def test_the_artifact_covers_every_file_of_each_corpus():
    if not OUT.exists():
        pytest.skip("corpus_provenance.json not present")
    report = json.loads(OUT.read_text())
    for corpus in ("E-MLB", "Pure_BA"):
        block = report[corpus]
        assert block["files_whose_outputs_all_come_from_capture"] == block["files"]
    dvsd22 = report["DVSD22"]
    for values in dvsd22["device_filters"].values():
        assert sum(values.values()) == dvsd22["files"]
    host = dvsd22["host_filter_chain"]
    assert sum(c["files"] for c in host["chains"]) + host["files_without_stored_chain"] \
        == dvsd22["files"]
