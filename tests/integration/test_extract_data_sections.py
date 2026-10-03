import json

import pytest

from scripts.extract import extract_data_sections as eds
from tests.conftest import requires_gcc, requires_arm32_gcc, requires_arm64_gcc, requires_x86_32

pytestmark = pytest.mark.slow

FUNC_DEF = (
    "int touch_data(int flag) {\n"
    "    static int counter = 7;\n"
    "    const char *msg = \"hi\";\n"
    "    counter += flag;\n"
    "    return counter + (int)msg[0];\n"
    "}\n"
)


@requires_gcc
def test_process_record_x86_64_finds_data_and_rodata(record_factory):
    rec = record_factory(fname="touch_data", func_def=FUNC_DEF, func_head="int touch_data(int flag)")
    data_section, errors = eds.process_record(rec, archs=["x86-64"])
    assert errors["x86-64"] is None
    assert ".data" in data_section["x86-64"]
    assert ".rodata" in data_section["x86-64"]


@requires_gcc
def test_process_record_main_named_function_no_driver_collision(record_factory):
    rec = record_factory(fname="main", func_def="int main(void) {\n    static int x = 3;\n    return x;\n}",
                          func_head="int main(void)")
    data_section, errors = eds.process_record(rec, archs=["x86-64"])
    assert errors["x86-64"] is None


@requires_gcc
def test_process_record_link_failure_reports_error_not_empty_success(record_factory):
    rec = record_factory(fname="calls_missing", func_def="int calls_missing(void) { return undefined_symbol(); }",
                          func_head="int calls_missing(void)")
    data_section, errors = eds.process_record(rec, archs=["x86-64"])
    assert data_section["x86-64"] == ""
    assert errors["x86-64"] is not None
    assert "link_failed" in errors["x86-64"]


@requires_x86_32
def test_process_record_x86_32(record_factory):
    rec = record_factory(fname="touch_data", func_def=FUNC_DEF, func_head="int touch_data(int flag)")
    data_section, errors = eds.process_record(rec, archs=["x86"])
    assert errors["x86"] is None
    assert data_section["x86"]


@requires_arm32_gcc
def test_process_record_arm32(record_factory):
    rec = record_factory(fname="touch_data", func_def=FUNC_DEF, func_head="int touch_data(int flag)")
    data_section, errors = eds.process_record(rec, archs=["ARM32"])
    assert errors["ARM32"] is None
    assert data_section["ARM32"]


@requires_arm64_gcc
def test_process_record_arm64(record_factory):
    rec = record_factory(fname="touch_data", func_def=FUNC_DEF, func_head="int touch_data(int flag)")
    data_section, errors = eds.process_record(rec, archs=["ARM64"])
    assert errors["ARM64"] is None
    assert data_section["ARM64"]


@requires_gcc
def test_cli_writes_jsonl_with_data_section_field(monkeypatch, tmp_path, patch_iter_split, run_main, record_factory):
    recs = [record_factory(fname="touch_data", func_def=FUNC_DEF, func_head="int touch_data(int flag)")]
    patch_iter_split(eds, recs)
    out_path = tmp_path / "out.jsonl"
    summary_path = tmp_path / "summary.json"

    run_main(eds, ["--out", str(out_path), "--summary-out", str(summary_path),
                   "--archs", "x86-64", "--n", "1"])

    lines = out_path.read_text().splitlines()
    assert len(lines) == 1
    rec_out = json.loads(lines[0])
    assert rec_out["fname"] == "touch_data"
    assert rec_out["data_section_errors"]["x86-64"] is None
    assert ".rodata" in rec_out["data_section"]["x86-64"]

    summary = json.loads(summary_path.read_text())
    assert summary["written"] == 1
    assert summary["nonempty"]["x86-64"] == 1
