import json

from scripts.survey import survey_unused_param_wrappers as supw


def _rec(fname, wrapper, func_head_types, func_def, path):
    return {"fname": fname, "path": path, "func_head_types": func_head_types, "func_def": func_def,
            "real_exe_wrapper": wrapper, "angha_exe_wrapper": None}


def test_survey_buckets_ok_and_failed_adaptations(monkeypatch, tmp_path):
    ok_rec = _rec(
        "add",
        'int a = input_json["a"];\nint b = input_json["b"];\nint result = add(a, b);\n',
        "int add(int a, int b)", "int add(int a, int b) { return a; }", "o/r/a.c",
    )
    fail_rec = _rec(
        "broken",
        "no recognizable shape at all",
        "int broken(int a, int b)", "int broken(int a, int b) { return a; }", "o/r/b.c",
    )
    no_wrapper_rec = _rec(None, None, "int skip(int a, int b)", "int skip(int a, int b) { return a; }", "o/r/c.c")
    no_wrapper_rec["fname"] = "skip"
    used_rec = _rec(
        "used",
        'int a = input_json["a"];\nint b = input_json["b"];\nint result = used(a, b);\n',
        "int used(int a, int b)", "int used(int a, int b) { return a + b; }", "o/r/d.c",
    )

    records = [ok_rec, fail_rec, no_wrapper_rec, used_rec]
    monkeypatch.setattr(supw, "iter_split", lambda split, limit=None: iter(records))

    sample_dir = tmp_path / "samples"
    result = supw.survey("test_real", n=10, limit=None, sample_dir=str(sample_dir))

    assert result["scanned"] == 4
    assert result["matched_unused_param_with_wrapper"] == 2  # ok_rec + fail_rec only
    assert result["field_ok"]["real_exe_wrapper"] == 1
    assert result["field_attempted"]["real_exe_wrapper"] == 2
    assert result["field_failure_rate"]["real_exe_wrapper"] == 0.5
    assert result["edge_case_count_saved"] == 1

    all_matched = (sample_dir / "test_real_all_matched.jsonl").read_text().splitlines()
    assert len(all_matched) == 2
    edge_cases = (sample_dir / "test_real_edge_cases.jsonl").read_text().splitlines()
    assert len(edge_cases) == 1
    assert json.loads(edge_cases[0])["fname"] == "broken"


def test_cli_writes_report(monkeypatch, tmp_path, run_main):
    ok_rec = _rec(
        "add",
        'int a = input_json["a"];\nint b = input_json["b"];\nint result = add(a, b);\n',
        "int add(int a, int b)", "int add(int a, int b) { return a; }", "o/r/a.c",
    )
    monkeypatch.setattr(supw, "iter_split", lambda split, limit=None: iter([ok_rec]))

    out_path = tmp_path / "report.json"
    sample_dir = tmp_path / "samples"
    run_main(supw, ["--out", str(out_path), "--sample-dir", str(sample_dir), "--n", "5"])

    report = json.loads(out_path.read_text())
    assert report["matched_unused_param_with_wrapper"] == 1
