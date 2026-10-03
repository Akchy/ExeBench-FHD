import json

import pytest

from scripts.extract import extract_unused_param as eup

_WRAPPER = """\
extern "C" {
#include "placeholder.c"
}
int main(int argc, char** argv) {
    int a = input_json["a"];
    int b = input_json["b"];
    int result = add(a, b);
    output_json["result"] = result;
}
"""


@pytest.mark.parametrize("split", ["train_real_compilable", "test_real"])
def test_process_record_adds_adapted_wrapper_on_any_split(record_factory, split):
    rec = record_factory(fname="add", func_head_types="int add(int a, int b)",
                          func_def="int add(int a, int b) { return a; }",
                          real_exe_wrapper=_WRAPPER)
    out = eup.process_record(rec, split)
    assert out["unused_param"] is True
    assert "b" not in out["opt_real_exe_wrapper"].split("add(")[1].split(")")[0]
    assert out["opt_real_exe_wrapper_error"] is None


def test_process_record_eval_split_no_unused_param_adds_nothing(record_factory):
    rec = record_factory(fname="add", func_head_types="int add(int a, int b)",
                          func_def="int add(int a, int b) { return a + b; }",
                          real_exe_wrapper=_WRAPPER)
    out = eup.process_record(rec, "test_real")
    assert out["unused_param"] is False
    assert "opt_real_exe_wrapper" not in out


def test_process_record_original_wrapper_field_untouched(record_factory):
    rec = record_factory(fname="add", func_head_types="int add(int a, int b)",
                          func_def="int add(int a, int b) { return a; }",
                          real_exe_wrapper=_WRAPPER)
    out = eup.process_record(rec, "test_real")
    assert out["real_exe_wrapper"] == _WRAPPER


def test_cli_writes_only_unused_param_records_by_default(tmp_path, patch_iter_split, run_main, record_factory):
    matching = record_factory(fname="add", func_head_types="int add(int a, int b)",
                               func_def="int add(int a, int b) { return a; }", path="o/r/a.c")
    non_matching = record_factory(fname="sub", func_head_types="int sub(int a, int b)",
                                   func_def="int sub(int a, int b) { return a - b; }", path="o/r/b.c")
    patch_iter_split(eup, [matching, non_matching])

    out_path = tmp_path / "out.jsonl"
    run_main(eup, ["--out", str(out_path), "--split", "train_real_compilable"])

    lines = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert [r["fname"] for r in lines] == ["add"]
    assert lines[0]["unused_param"] is True


def test_cli_all_flag_writes_every_record(tmp_path, patch_iter_split, run_main, record_factory):
    matching = record_factory(fname="add", func_head_types="int add(int a, int b)",
                               func_def="int add(int a, int b) { return a; }", path="o/r/a.c")
    non_matching = record_factory(fname="sub", func_head_types="int sub(int a, int b)",
                                   func_def="int sub(int a, int b) { return a - b; }", path="o/r/b.c")
    patch_iter_split(eup, [matching, non_matching])

    out_path = tmp_path / "out.jsonl"
    run_main(eup, ["--out", str(out_path), "--split", "train_real_compilable", "--all"])

    lines = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert {r["fname"] for r in lines} == {"add", "sub"}


def test_cli_analyze_only_writes_nothing_and_prints_report(capsys, tmp_path, patch_iter_split, run_main, record_factory):
    rec = record_factory(fname="add", func_head_types="int add(int a, int b)",
                          func_def="int add(int a, int b) { return a; }", signature=["int", "int", "int"])
    patch_iter_split(eup, [rec])
    out_path = tmp_path / "out.jsonl"

    run_main(eup, ["--out", str(out_path), "--analyze-only"])

    assert not out_path.exists()
    report = json.loads(capsys.readouterr().out)
    assert report["scanned"] == 1
    assert report["signature_param_count_match"] == 1
