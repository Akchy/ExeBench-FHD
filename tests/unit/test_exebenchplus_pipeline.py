"""Tests for scripts.exebenchplus_pipeline.core.process_record.

These compile with real gcc (x86-64 only, tiny functions); CLI and
multi-arch coverage is in tests/integration/test_exebenchplus_pipeline.py.
"""
from tests.conftest import requires_gcc

from scripts.exebenchplus_pipeline import core

pytestmark = [requires_gcc]

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

# Produced only by extract_asm_regen.py, never by the pipeline.
_OUT_OF_SCOPE_FIELDS = (
    "asm", "asm_was_static", "objdump_asm", "objdump_asm_errors",
    "disassembled_asm", "disassembled_asm_errors",
)


def test_process_record_normal_record_has_every_documented_field(tmp_path, record_factory):
    rec = record_factory(fname="add_one", func_head_types="int add_one(int x)",
                          func_def="int add_one(int x) { return x + 1; }",
                          signature=["int", "int"])
    out = core.process_record(rec, "train_real_compilable", str(tmp_path), "t1", archs=["x86-64"])

    assert isinstance(out["no_return"], bool) and out["no_return"] is False
    assert isinstance(out["dummy_udt"], bool) and out["dummy_udt"] is False
    assert isinstance(out["trivial_function"], bool)
    assert isinstance(out["truncated_funcs"], bool) and out["truncated_funcs"] is False
    assert isinstance(out["unused_param"], bool) and out["unused_param"] is False
    assert "opt_real_exe_wrapper" not in out
    assert "opt_angha_exe_wrapper" not in out

    for field in _OUT_OF_SCOPE_FIELDS:
        assert field not in out, f"{field!r} should not be in the dataset-pipeline output"

    assert out["func_call_skip_reason"] is None
    assert set(out["func_call"]) == {"x86-64"}
    assert set(out["func_call_errors"]) == {"x86-64"}

    assert set(out["data_section"]) == {"x86-64"}
    assert set(out["data_section_errors"]) == {"x86-64"}

    assert out["fname"] == "add_one"
    assert out["path"] == rec["path"]


def test_process_record_main_named_function_skips_func_call_but_populates_everything_else(
        tmp_path, record_factory):
    rec = record_factory(fname="main", func_head_types="int main(void)",
                          func_def="int main(void) { return 0; }",
                          signature=["int"])
    out = core.process_record(rec, "train_real_compilable", str(tmp_path), "t2", archs=["x86-64"])

    assert out["func_call_skip_reason"] == "fname_is_main"
    assert out["func_call"] is None
    assert out["func_call_errors"] is None

    assert isinstance(out["no_return"], bool)
    assert isinstance(out["unused_param"], bool) and out["unused_param"] is False
    assert set(out["data_section"]) == {"x86-64"}
    for field in _OUT_OF_SCOPE_FIELDS:
        assert field not in out


def test_process_record_eval_split_unused_param_produces_opt_wrapper(tmp_path, record_factory):
    rec = record_factory(fname="add", func_head_types="int add(int a, int b)",
                          func_def="int add(int a, int b) { return a; }",
                          signature=["int", "int", "int"],
                          real_exe_wrapper=_WRAPPER)
    out = core.process_record(rec, "test_real", str(tmp_path), "t3", archs=["x86-64"])

    assert out["unused_param"] is True
    assert out["opt_real_exe_wrapper"]
    assert out["opt_real_exe_wrapper_error"] is None
    assert "b" not in out["opt_real_exe_wrapper"].split("add(")[1].split(")")[0]
    assert out["real_exe_wrapper"] == _WRAPPER

    assert out["func_call_skip_reason"] is None


def test_process_record_preserves_exebench_native_asm_field_untouched(tmp_path, record_factory):
    """ExeBench's own `asm` field (different shape) must not be overwritten."""
    native_asm = {"real_gcc_x86_O0": {"func_asm": ".globl add_one\nadd_one:\n\tret\n"}}
    rec = record_factory(fname="add_one", func_head_types="int add_one(int x)",
                          func_def="int add_one(int x) { return x + 1; }",
                          signature=["int", "int"], asm=native_asm)
    out = core.process_record(rec, "train_real_compilable", str(tmp_path), "t4", archs=["x86-64"])

    assert out["asm"] == native_asm
    assert "asm_was_static" not in out


def test_select_exe_wrappers_offers_reduced_and_full_form_above_o0():
    from scripts.exebenchplus_eval import select_exe_wrappers
    row = {"real_exe_wrapper": "full", "opt_real_exe_wrapper": "reduced", "unused_param": True}
    assert select_exe_wrappers(row, "O2") == ["reduced", "full"]
    assert select_exe_wrappers(row, "O0") == ["full"]
    assert select_exe_wrappers(dict(row, unused_param=False), "O2") == ["full"]
