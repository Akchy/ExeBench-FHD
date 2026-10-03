import json

import pytest

from scripts.extract import extract_func_call as efc
from tests.conftest import requires_gcc, requires_arm32_gcc, requires_arm64_gcc, requires_x86_32

pytestmark = pytest.mark.slow

FUNC_DEF = "int add_scaled(int base, float scale) {\n    return base + (int)(base * scale);\n}\n"


@requires_gcc
def test_process_record_x86_64_finds_call_site(tmp_path, record_factory):
    rec = record_factory(fname="add_scaled", func_def=FUNC_DEF, func_head="int add_scaled(int base, float scale)",
                          signature=["int", "int", "float"])
    rec_out, skip_reason = efc.process_record(rec, str(tmp_path), "t1", archs=["x86-64"])
    assert skip_reason is None
    assert rec_out["func_call_errors"]["x86-64"] is None
    assert rec_out["func_call"]["x86-64"]
    assert "func0" in rec_out["func_call"]["x86-64"]


@requires_gcc
def test_process_record_main_named_is_skipped(tmp_path, record_factory):
    rec = record_factory(fname="main", func_def="int main(void){return 0;}",
                          func_head="int main(void)", signature=["int"])
    rec_out, skip_reason = efc.process_record(rec, str(tmp_path), "t2", archs=["x86-64"])
    assert skip_reason == "fname_is_main"
    assert rec_out["func_call"] is None


@requires_gcc
def test_process_record_struct_by_value_param_is_skipped(tmp_path, record_factory):
    rec = record_factory(fname="takes_struct", func_def="int takes_struct(struct foo f){return 0;}",
                          func_head="int takes_struct(struct foo f)", signature=["int", "struct foo"])
    rec_out, skip_reason = efc.process_record(rec, str(tmp_path), "t3", archs=["x86-64"])
    assert skip_reason == "unsupported_param_type:struct_or_union_by_value"
    assert rec_out["func_call"] is None


@requires_x86_32
def test_process_record_x86_32(tmp_path, record_factory):
    rec = record_factory(fname="add_scaled", func_def=FUNC_DEF, func_head="int add_scaled(int base, float scale)",
                          signature=["int", "int", "float"])
    rec_out, skip_reason = efc.process_record(rec, str(tmp_path), "t4", archs=["x86"])
    assert skip_reason is None
    assert rec_out["func_call"]["x86"]


@requires_arm32_gcc
def test_process_record_arm32(tmp_path, record_factory):
    rec = record_factory(fname="add_scaled", func_def=FUNC_DEF, func_head="int add_scaled(int base, float scale)",
                          signature=["int", "int", "float"])
    rec_out, skip_reason = efc.process_record(rec, str(tmp_path), "t5", archs=["ARM32"])
    assert skip_reason is None
    assert rec_out["func_call"]["ARM32"]


@requires_arm64_gcc
def test_process_record_arm64(tmp_path, record_factory):
    rec = record_factory(fname="add_scaled", func_def=FUNC_DEF, func_head="int add_scaled(int base, float scale)",
                          signature=["int", "int", "float"])
    rec_out, skip_reason = efc.process_record(rec, str(tmp_path), "t6", archs=["ARM64"])
    assert skip_reason is None
    assert rec_out["func_call"]["ARM64"]


@requires_gcc
def test_cli_writes_matched_and_skipped_jsonl(tmp_path, patch_iter_split, run_main, record_factory):
    good = record_factory(fname="add_scaled", func_def=FUNC_DEF, func_head="int add_scaled(int base, float scale)",
                           signature=["int", "int", "float"], path="o/r/good.c")
    skipped = record_factory(fname="main", func_def="int main(void){return 0;}",
                              func_head="int main(void)", signature=["int"], path="o/r/main.c")
    patch_iter_split(efc, [good, skipped])

    out_path = tmp_path / "matched.jsonl"
    skip_path = tmp_path / "skipped.jsonl"
    work_dir = tmp_path / "work"

    run_main(efc, ["--out", str(out_path), "--skipped-out", str(skip_path),
                   "--work-dir", str(work_dir), "--archs", "x86-64", "--n", "5"])

    matched_lines = out_path.read_text().splitlines()
    skipped_lines = skip_path.read_text().splitlines()
    assert len(matched_lines) == 1
    assert len(skipped_lines) == 1
    assert json.loads(matched_lines[0])["fname"] == "add_scaled"
    assert json.loads(skipped_lines[0])["func_call_skip_reason"] == "fname_is_main"


# Ignores every parameter: with only noinline, -O3 clones it to
# func0.constprop.0 and drops the arguments from the call.
UNUSED_PARAMS_DEF = ("int ignore_all(unsigned char *name, unsigned int namelen, unsigned char *data, "
                     "unsigned int datalen)\n{\n return -1;\n}\n")
UNUSED_PARAMS_SIG = ["int", "unsigned char *", "unsigned int", "unsigned char *", "unsigned int"]


def _driver_src(record_factory):
    rec = record_factory(fname="ignore_all", func_def=UNUSED_PARAMS_DEF, signature=UNUSED_PARAMS_SIG,
                         real_deps="", angha_deps="")
    params, _ = efc.analyze_params(UNUSED_PARAMS_SIG)
    return efc.build_driver_source(rec, "ignore_all", "int", params)


def _compile_and_window(tmp_path, src, extra_flags=()):
    import subprocess
    c_file = tmp_path / "d.c"
    c_file.write_text(src)
    out = tmp_path / "d.out"
    subprocess.run(["gcc", "-m64", "-O3", *extra_flags, str(c_file), "-o", str(out)], check=True,
                   capture_output=True)
    asm = subprocess.run(["objdump", "-d", str(out)], capture_output=True, text=True, check=True).stdout
    return efc.extract_call_window(asm)


@requires_gcc
def test_noipa_prevents_clone_and_keeps_every_argument(tmp_path, record_factory):
    window = _compile_and_window(tmp_path, _driver_src(record_factory))
    assert window.splitlines()[-1].rstrip().endswith("<func0>")
    for reg in ("%rdi", "%rsi", "%rdx", "%rcx"):  # 4 args -> first 4 SysV integer arg registers
        assert reg in window or reg.replace("%r", "%e") in window


@requires_gcc
def test_plain_noinline_would_clone_this_function(tmp_path, record_factory):
    # Guards the test above: proves this input really does get cloned without noipa.
    src = _driver_src(record_factory).replace(
        "#    define __FUNC_CALL_NOIPA __attribute__((noipa))",
        "#    define __FUNC_CALL_NOIPA __attribute__((noinline))")
    window = _compile_and_window(tmp_path, src)
    assert "<func0." in window.splitlines()[-1]


@requires_gcc
@pytest.mark.parametrize("simulate", ["-D__has_attribute(x)=0", "-U__has_attribute"],
                         ids=["noipa_unsupported", "no_has_attribute"])
def test_driver_falls_back_to_noinline_without_noipa(tmp_path, record_factory, simulate):
    import subprocess
    src = _driver_src(record_factory)
    c_file = tmp_path / "d.c"
    c_file.write_text(src)
    pre = subprocess.run(["gcc", "-E", "-P", simulate, str(c_file)], capture_output=True, text=True, check=True).stdout
    assert "__attribute__((noinline))\nint func0(" in pre
    assert "noipa" not in pre
    # and it still builds and yields a call window (via the fallback attribute)
    assert _compile_and_window(tmp_path, src, extra_flags=(simulate,))


@requires_gcc
def test_cleanup_only_removes_own_files_not_prefix_matches(tmp_path, record_factory):
    # task_id "t_1" must not delete "t_13"'s files.
    c_dir = tmp_path / "C_files"
    c_dir.mkdir()
    bystanders = [c_dir / "t_13.c", c_dir / "t_13_x86-64.out", c_dir / "t_1x.c"]
    for p in bystanders:
        p.write_text("other record")
    rec = record_factory(fname="add_scaled", func_def=FUNC_DEF, signature=["int", "int", "float"])
    efc.process_record(rec, str(tmp_path), "t_1", archs=["x86-64"])
    assert all(p.exists() for p in bystanders)
    assert not (c_dir / "t_1.c").exists()
    assert not (c_dir / "t_1_x86-64.out").exists()


@requires_gcc
def test_per_level_mode_returns_arch_level_dicts(tmp_path, record_factory):
    rec = record_factory(fname="add_scaled", func_def=FUNC_DEF, signature=["int", "int", "float"])
    rec_out, skip_reason = efc.process_record(rec, str(tmp_path), "t_lv", archs=["x86-64"],
                                              levels=efc.FUNC_CALL_LEVELS)
    assert skip_reason is None
    assert list(rec_out["func_call"]["x86-64"]) == efc.FUNC_CALL_LEVELS
    for level in efc.FUNC_CALL_LEVELS:
        assert rec_out["func_call_errors"]["x86-64"] is None  # collapsed: no error at any level
        assert rec_out["func_call"]["x86-64"][level].rstrip().endswith("<func0>")
    assert not list((tmp_path / "C_files").iterdir())  # every per-level .out cleaned up


@requires_gcc
def test_default_mode_is_still_single_level_string(tmp_path, record_factory):
    rec = record_factory(fname="add_scaled", func_def=FUNC_DEF, signature=["int", "int", "float"])
    rec_out, _ = efc.process_record(rec, str(tmp_path), "t_def", archs=["x86-64"])
    assert isinstance(rec_out["func_call"]["x86-64"], str) and rec_out["func_call"]["x86-64"]


def test_has_snippet_handles_both_shapes():
    assert efc._has_snippet("x") and not efc._has_snippet("")
    assert efc._has_snippet({"O0": "", "O2": "x"}) and not efc._has_snippet({"O0": "", "O2": ""})
    assert not efc._has_snippet(None)


@requires_gcc
def test_per_level_errors_collapse_when_identical(tmp_path, record_factory):
    # A missing callee fails the link identically at every level -> stored once.
    rec = record_factory(fname="calls_missing", func_def="int calls_missing(int x) {\n    return nowhere(x);\n}\n",
                         signature=["int", "int"], real_deps="int nowhere(int);\n", angha_deps="")
    rec_out, skip = efc.process_record(rec, str(tmp_path), "t_err", archs=["x86-64"], levels=efc.FUNC_CALL_LEVELS)
    err = rec_out["func_call_errors"]["x86-64"]
    assert isinstance(err, str) and err.startswith("compile_failed:")
    assert rec_out["func_call"]["x86-64"] == {lv: "" for lv in efc.FUNC_CALL_LEVELS}
    assert efc.func_call_by_arch(rec_out["func_call_errors"], "O2") == {"x86-64": err}
