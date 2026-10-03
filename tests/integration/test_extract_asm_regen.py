import json
import re

import pytest

from scripts.extract import extract_asm_regen as ear
from scripts.extract.extract_func_call import ARCH_TOOLCHAIN
from tests.conftest import (requires_arm32_gcc, requires_arm64_gcc, requires_gcc,
                             requires_x86_32)

pytestmark = [pytest.mark.slow, requires_gcc]

STATIC_FUNC_DEF = "static int helper(int x) {\n    return x + 1;\n}\n"
PLAIN_FUNC_DEF = "int helper(int x) {\n    return x + 1;\n}\n"
PLAIN_FUNC_DEF2 = "int helper2(int x) {\n    return x + 1;\n}\n"

# "   4:\te8 00 00 00 00 \tcall ..." - an address-prefixed disassembly line.
_ADDR_LINE_RE = re.compile(r"^\s*[0-9a-fA-F]+:", re.MULTILINE)


def test_process_record_asm_field_unchanged_by_the_arch_extension(tmp_path, record_factory):
    rec = record_factory(fname="helper", func_def=STATIC_FUNC_DEF, signature=["int", "int"])
    out = ear.process_record(rec, work_dir=str(tmp_path), task_id="t1", archs=["x86-64"])

    assert out["asm_was_static"] is True
    for level in ("O0", "O1", "O2", "O3"):
        res = out["asm"][level]
        assert "error" not in res, res.get("error")
        assert res["n_instr"] > 0


def test_process_record_preserves_every_original_field(tmp_path, record_factory):
    rec = record_factory(fname="helper", func_def=STATIC_FUNC_DEF, signature=["int", "int"])
    out = ear.process_record(rec, work_dir=str(tmp_path), task_id="t2", archs=["x86-64"])
    for key, value in rec.items():
        assert out[key] == value


def test_process_record_default_is_production_safe_and_skips_objdump_asm(tmp_path, record_factory):
    rec = record_factory(fname="helper", func_def=STATIC_FUNC_DEF, signature=["int", "int"])
    out = ear.process_record(rec, work_dir=str(tmp_path), task_id="tprod")

    assert out["asm_was_static"] is True
    assert out["objdump_asm"] is None
    assert out["objdump_asm_errors"] is None
    assert not (tmp_path / "C_files").exists()


def test_process_record_objdump_asm_x86_64_has_all_five_levels_with_address(tmp_path, record_factory):
    rec = record_factory(fname="helper", func_def=STATIC_FUNC_DEF, signature=["int", "int"])
    out = ear.process_record(rec, work_dir=str(tmp_path), task_id="t3", archs=["x86-64"],
                              include_objdump_asm=True)

    levels = out["objdump_asm"]["x86-64"]
    assert set(levels) == {"O0", "O1", "O2", "O3", "Os"}
    for level, snippet in levels.items():
        assert out["objdump_asm_errors"]["x86-64"][level] is None, out["objdump_asm_errors"]["x86-64"][level]
        assert snippet, f"{level} produced no snippet"
        assert "<helper>" in snippet
        assert _ADDR_LINE_RE.search(snippet), f"{level} snippet is missing address-prefixed lines:\n{snippet}"


def test_process_record_objdump_asm_driver_never_calls_target(tmp_path, record_factory):
    # The target is never called, so it survives only through external linkage.
    rec = record_factory(fname="helper", func_def=STATIC_FUNC_DEF, signature=["int", "int"])
    out = ear.process_record(rec, work_dir=str(tmp_path), task_id="t4", archs=["x86-64"],
                              include_objdump_asm=True)
    assert "<helper>" in out["objdump_asm"]["x86-64"]["O2"]


@requires_x86_32
def test_process_record_objdump_asm_x86_32(tmp_path, record_factory):
    rec = record_factory(fname="helper", func_def=STATIC_FUNC_DEF, signature=["int", "int"])
    out = ear.process_record(rec, work_dir=str(tmp_path), task_id="t5", archs=["x86"],
                              include_objdump_asm=True)
    assert out["objdump_asm_errors"]["x86"]["O2"] is None
    assert "<helper>" in out["objdump_asm"]["x86"]["O2"]


@requires_arm32_gcc
def test_process_record_objdump_asm_arm32(tmp_path, record_factory):
    rec = record_factory(fname="helper", func_def=STATIC_FUNC_DEF, signature=["int", "int"])
    out = ear.process_record(rec, work_dir=str(tmp_path), task_id="t6", archs=["ARM32"],
                              include_objdump_asm=True)
    assert out["objdump_asm_errors"]["ARM32"]["O2"] is None
    assert "<helper>" in out["objdump_asm"]["ARM32"]["O2"]


@requires_arm64_gcc
def test_process_record_objdump_asm_arm64(tmp_path, record_factory):
    rec = record_factory(fname="helper", func_def=STATIC_FUNC_DEF, signature=["int", "int"])
    out = ear.process_record(rec, work_dir=str(tmp_path), task_id="t7", archs=["ARM64"],
                              include_objdump_asm=True)
    assert out["objdump_asm_errors"]["ARM64"]["O2"] is None
    assert "<helper>" in out["objdump_asm"]["ARM64"]["O2"]


@requires_x86_32
@requires_arm32_gcc
@requires_arm64_gcc
def test_process_record_objdump_asm_default_archs_covers_all_four(tmp_path, record_factory):
    rec = record_factory(fname="helper", func_def=STATIC_FUNC_DEF, signature=["int", "int"])
    out = ear.process_record(rec, work_dir=str(tmp_path), task_id="t8",
                              include_objdump_asm=True)  # archs=None -> all four

    assert set(out["objdump_asm"]) == set(ARCH_TOOLCHAIN)
    for arch in ARCH_TOOLCHAIN:
        assert out["objdump_asm_errors"][arch]["O2"] is None
        assert "<helper>" in out["objdump_asm"][arch]["O2"]


def test_process_record_defaults_to_a_private_temp_dir_and_cleans_up(record_factory):
    rec = record_factory(fname="helper", func_def=STATIC_FUNC_DEF, signature=["int", "int"])
    out = ear.process_record(rec, archs=["x86-64"], include_objdump_asm=True)  # no work_dir/task_id given
    assert "<helper>" in out["objdump_asm"]["x86-64"]["O2"]


def test_process_record_keep_artifacts_leaves_files_on_disk(tmp_path, record_factory):
    rec = record_factory(fname="helper", func_def=STATIC_FUNC_DEF, signature=["int", "int"])
    ear.process_record(rec, work_dir=str(tmp_path), task_id="keepme", archs=["x86-64"],
                        cleanup=False, include_objdump_asm=True)
    assert list((tmp_path / "C_files").glob("keepme*"))


def test_cli_writes_only_records_that_needed_stripping(tmp_path, patch_iter_split, run_main, record_factory):
    matching = record_factory(fname="helper", func_def=STATIC_FUNC_DEF, signature=["int", "int"], path="o/r/a.c")
    non_matching = record_factory(fname="helper2", func_def=PLAIN_FUNC_DEF2, signature=["int", "int"], path="o/r/b.c")
    patch_iter_split(ear, [matching, non_matching])

    out_path = tmp_path / "out.jsonl"
    run_main(ear, ["--out", str(out_path), "--n", "10", "--work-dir", str(tmp_path / "work")])

    lines = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["fname"] == "helper"
    assert lines[0]["asm_was_static"] is True


def test_cli_default_production_run_never_computes_objdump_asm(tmp_path, patch_iter_split, run_main,
                                                                 record_factory):
    rec = record_factory(fname="helper", func_def=STATIC_FUNC_DEF, signature=["int", "int"], path="o/r/a.c")
    patch_iter_split(ear, [rec])

    out_path = tmp_path / "out.jsonl"
    run_main(ear, ["--out", str(out_path), "--all", "--work-dir", str(tmp_path / "work")])

    line = json.loads(out_path.read_text().splitlines()[0])
    assert line["asm_was_static"] is True
    assert line["objdump_asm"] is None
    assert line["objdump_asm_errors"] is None


def test_cli_all_with_objdump_asm_flag_writes_asm_and_objdump_asm_fields(tmp_path, patch_iter_split, run_main,
                                                                          record_factory):
    matching = record_factory(fname="helper", func_def=STATIC_FUNC_DEF, signature=["int", "int"], path="o/r/a.c")
    non_matching = record_factory(fname="helper2", func_def=PLAIN_FUNC_DEF2, signature=["int", "int"], path="o/r/b.c")
    patch_iter_split(ear, [matching, non_matching])

    out_path = tmp_path / "out.jsonl"
    run_main(ear, ["--out", str(out_path), "--all", "--with-objdump-asm", "--archs", "x86-64",
                   "--work-dir", str(tmp_path / "work")])

    lines = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(lines) == 2
    assert {(l["fname"], l["asm_was_static"]) for l in lines} == {("helper", True), ("helper2", False)}
    assert all(l["objdump_asm"]["x86-64"]["O2"] for l in lines)


def test_cli_split_all_writes_one_file_per_split(tmp_path, patch_iter_split, run_main, record_factory):
    rec = record_factory(fname="helper", func_def=STATIC_FUNC_DEF, signature=["int", "int"], path="o/r/a.c")
    patch_iter_split(ear, [rec])

    out_dir = tmp_path / "asm_regen_field"
    run_main(ear, ["--split", "all", "--all", "--out-dir", str(out_dir),
                   "--work-dir", str(tmp_path / "work")])

    from scripts.check_split_leakage import ALL_SPLITS
    for split in ALL_SPLITS:
        split_path = out_dir / f"{split}.jsonl"
        assert split_path.exists()
        assert json.loads(split_path.read_text().splitlines()[0])["asm_was_static"] is True


def test_cli_rejects_unknown_arch(run_main, tmp_path):
    with pytest.raises(SystemExit):
        run_main(ear, ["--archs", "not-a-real-arch", "--work-dir", str(tmp_path / "work")])


def test_objdump_asm_cleanup_only_removes_own_files_not_prefix_matches(tmp_path, record_factory):
    c_dir = tmp_path / "C_files"
    c_dir.mkdir()
    bystanders = [c_dir / "t_13.c", c_dir / "t_13_x86-64_O2.out"]
    for p in bystanders:
        p.write_text("other record")
    ear.process_record(record_factory(), work_dir=str(tmp_path), task_id="t_1", archs=["x86-64"],
                       include_objdump_asm=True)
    assert all(p.exists() for p in bystanders)
    assert not any(c_dir.glob("t_1.c")) and not any(c_dir.glob("t_1_*.out"))


FLOAT_CONST_DEF = "float scale_half(float x) {\n    return x * 0.37f + 1.25f;\n}\n"


def _rip_targets(snippet):
    """Absolute addresses of every `disp(%rip)` operand in an x86-64 snippet
    (disp is relative to the next instruction's address)."""
    lines = [l for l in snippet.splitlines() if re.match(r"^\s*[0-9a-f]+:", l)]
    insns = []
    for l in lines:
        addr, rest = l.split(":", 1)
        parts = rest.split("\t")
        nbytes = len(parts[1].split()) if len(parts) > 1 else 0
        insns.append((int(addr, 16), nbytes, parts[2] if len(parts) > 2 else ""))
    targets = []
    for i, (addr, nbytes, text) in enumerate(insns):
        m = re.search(r"(-?0x[0-9a-f]+)\(%rip\)", text)
        if m:
            nxt = next((a for a, _n, t in insns[i + 1:] if t), addr + nbytes)
            targets.append(nxt + int(m.group(1), 16))
    return targets


def test_data_section_is_built_per_level_from_the_objdump_asm_binary(tmp_path, record_factory):
    rec = record_factory(fname="scale_half", func_def=FLOAT_CONST_DEF, real_deps="", angha_deps="")
    out = ear.process_record(rec, work_dir=str(tmp_path), task_id="t", archs=["x86-64"], include_objdump_asm=True)
    assert set(out["data_section"]["x86-64"]) == set(ear.OBJDUMP_LEVELS)
    for level in ("O1", "O2", "O3"):  # merged .rodata.cst4 constants at -O1+
        ds = out["data_section"]["x86-64"][level]
        assert out["data_section_errors"]["x86-64"][level] is None
        assert "not located" not in ds and "zero-initialized" not in ds
        blocks = [(int(a, 16), int(n)) for a, n in re.findall(r"@0x([0-9a-f]+) \((\d+) bytes\)", ds)]
        targets = _rip_targets(out["objdump_asm"]["x86-64"][level])
        assert targets
        for t in targets:  # every constant the asm loads lies inside a data_section block
            assert any(a <= t < a + n for a, n in blocks), (level, hex(t), ds)


BSS_DEF = "int counter;\nint bump(void) {\n    return ++counter;\n}\n"


def test_bss_is_reported_zero_initialized_not_dumped(tmp_path, record_factory):
    rec = record_factory(fname="bump", func_def=BSS_DEF, real_deps="", angha_deps="")
    out = ear.process_record(rec, work_dir=str(tmp_path), task_id="t", archs=["x86-64"], include_objdump_asm=True)
    ds = out["data_section"]["x86-64"]["O0"]
    assert re.search(r"^\.bss @0x[0-9a-f]+ \(4 bytes, zero-initialized\)$", ds, re.M), ds


def test_oversized_function_is_not_disassembled(tmp_path, record_factory):
    rec = record_factory(fname="dummy", path="aosm/gcc_os/gcc/testsuite/gcc.dg/long-branch-middle.c",
                         func_def="int dummy(void) {\n    return 0;\n}\n", real_deps="", angha_deps="")
    out = ear.process_record(rec, work_dir=str(tmp_path), task_id="t", archs=["x86-64"], include_objdump_asm=True)
    for level in ear.OBJDUMP_LEVELS:
        assert out["objdump_asm"]["x86-64"][level] == ""
        assert out["objdump_asm_errors"]["x86-64"][level].startswith("skipped_oversized:")
        assert out["data_section_errors"]["x86-64"][level] is None  # still built and linked
