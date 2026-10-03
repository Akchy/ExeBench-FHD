import json

import pytest

from scripts import regen_asm
from tests.conftest import requires_gcc

pytestmark = [pytest.mark.slow, requires_gcc]


def test_compile_one_o0_and_o2_both_succeed_with_instructions():
    src = "static int add_one(int x) {\n    return x + 1;\n}\n"
    stripped, _ = regen_asm.strip_static(src)

    res_o0 = regen_asm.compile_one(stripped, "O0", "add_one")
    assert "error" not in res_o0
    assert res_o0["n_instr"] > 0
    assert "add_one" in res_o0["func_asm"]

    res_o2 = regen_asm.compile_one(stripped, "O2", "add_one")
    assert "error" not in res_o2
    assert res_o2["n_instr"] > 0


def test_compile_one_static_without_stripping_vanishes_at_o1_plus():
    src = "static int add_one(int x) {\n    return x + 1;\n}\n"
    res_o1 = regen_asm.compile_one(src, "O1", "add_one")
    assert "error" not in res_o1
    assert res_o1["n_instr"] == 0  # GCC deletes an unreferenced static function


def test_compile_one_invalid_source_returns_error():
    res = regen_asm.compile_one("int broken( {{{ not C", "O0", "broken")
    assert "error" in res


def test_cli_main_writes_asm_regen_field(monkeypatch, tmp_path, run_main):
    records = [
        {"fname": "add_one", "func_def": "static int add_one(int x) {\n    return x + 1;\n}\n",
         "real_deps": "", "angha_deps": "", "path": "o/r/a.c", "signature": ["int", "int"],
         "func_head_types": None},
    ]

    def fake_iter_split(split, limit=None):
        data = records if limit is None else records[:limit]
        yield from data

    monkeypatch.setattr(regen_asm, "iter_split", fake_iter_split)

    out_path = tmp_path / "out.jsonl"
    run_main(regen_asm, ["test_real", str(out_path), "--levels", "O0,O1"])

    lines = out_path.read_text().splitlines()
    assert len(lines) == 1
    rec_out = json.loads(lines[0])
    assert rec_out["was_static"] is True
    assert "static" not in rec_out["func_def_compiled"]
    assert set(rec_out["asm_regen"]) == {"O0", "O1"}
    assert rec_out["asm_regen"]["O0"]["n_instr"] > 0
    # stripped of static, so it survives at O1 too (unlike the un-stripped case).
    assert rec_out["asm_regen"]["O1"]["n_instr"] > 0
