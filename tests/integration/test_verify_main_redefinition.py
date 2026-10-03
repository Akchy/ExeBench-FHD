import json

import pytest

from scripts.verify import verify_main_redefinition as vmr
from tests.conftest import requires_gcc

pytestmark = [pytest.mark.slow, requires_gcc]


def test_worker_reproduces_redefinition_of_main_bug(tmp_path, monkeypatch):
    monkeypatch.setattr(vmr, "SCRATCH_BASE", str(tmp_path))
    rec = {"fname": "main", "path": "o/r/a.c", "func_def": "int main(void) { return 0; }",
           "real_deps": "", "angha_deps": ""}

    result = vmr._worker((0, "O0", rec))

    assert result["compiled_ok"] is False
    assert result["reason"] == "redefinition_of_main"


def test_worker_normal_function_compiles_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(vmr, "SCRATCH_BASE", str(tmp_path))
    rec = {"fname": "add", "path": "o/r/b.c", "func_def": "int add(int a, int b) { return a + b; }",
           "real_deps": "", "angha_deps": ""}

    result = vmr._worker((1, "O0", rec))

    assert result["compiled_ok"] is True
    assert result["reason"] == "compiled_ok"


def test_cli_end_to_end_classifies_and_writes_summary(tmp_path, monkeypatch, capsys, run_main):
    monkeypatch.setattr(vmr, "SCRATCH_BASE", str(tmp_path / "scratch"))
    input_path = tmp_path / "main_named.jsonl"
    input_path.write_text(json.dumps({
        "fname": "main", "path": "o/r/a.c", "func_def": "int main(void) { return 0; }",
        "real_deps": "", "angha_deps": "",
    }) + "\n")
    out_path = tmp_path / "results.jsonl"

    run_main(vmr, ["--input", str(input_path), "--out", str(out_path), "--workers", "1"])

    results = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert results[0]["reason"] == "redefinition_of_main"

    summary = json.loads(capsys.readouterr().out)
    assert summary["counts"]["redefinition_of_main"] == 1
    assert summary["pct"]["redefinition_of_main"] == 100.0
