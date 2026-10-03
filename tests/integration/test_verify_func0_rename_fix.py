import json

import pytest

from scripts.verify import verify_func0_rename_fix as vfrf
from tests.conftest import requires_gcc, requires_objdump

pytestmark = [pytest.mark.slow, requires_gcc, requires_objdump]


def test_worker_a_main_named_renames_to_func0_and_compiles_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(vfrf, "SCRATCH_BASE", str(tmp_path))
    rec = {"fname": "main", "path": "o/r/a.c", "func_def": "int main(void) { return 0; }",
           "real_deps": "", "angha_deps": ""}

    result = vfrf._worker_a((0, "O0", rec))

    assert result["compiled_ok"] is True
    assert result["reason"] == "compiled_ok"


def test_worker_b_main_named_extracts_main_label(tmp_path, monkeypatch):
    monkeypatch.setattr(vfrf, "SCRATCH_BASE", str(tmp_path))
    rec = {"fname": "main", "path": "o/r/a.c", "func_def": "int main(void) { return 0; }",
           "real_deps": "", "angha_deps": ""}

    result = vfrf._worker_b((0, "O0", rec, "main_named"))

    assert result["extracted"] is True


def test_worker_b_non_main_renames_to_func0_and_extracts(tmp_path, monkeypatch):
    monkeypatch.setattr(vfrf, "SCRATCH_BASE", str(tmp_path))
    rec = {"fname": "add_one", "path": "o/r/b.c", "func_def": "int add_one(int x) { return x + 1; }",
           "real_deps": "", "angha_deps": ""}

    result = vfrf._worker_b((0, "O0", rec, "non_main"))

    assert result["extracted"] is True


def test_cli_experiment_a_only_end_to_end(tmp_path, monkeypatch, capsys, run_main):
    monkeypatch.setattr(vfrf, "SCRATCH_BASE", str(tmp_path / "scratch"))
    input_path = tmp_path / "main_named.jsonl"
    input_path.write_text(json.dumps({
        "fname": "main", "path": "o/r/a.c", "func_def": "int main(void) { return 0; }",
        "real_deps": "", "angha_deps": "",
    }) + "\n")
    out_path = tmp_path / "results.jsonl"
    e2e_out_path = tmp_path / "e2e.jsonl"

    run_main(vfrf, [
        "--input", str(input_path), "--out", str(out_path),
        "--e2e-out", str(e2e_out_path), "--workers", "1", "--skip-b",
    ])

    results = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert results[0]["reason"] == "compiled_ok"

    summary = json.loads(capsys.readouterr().out)
    assert summary["experiment_a"]["counts"]["compiled_ok"] == 1
    assert "experiment_b" not in summary
