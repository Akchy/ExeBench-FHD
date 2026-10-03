import json

from scripts.survey import survey_unused_param as sup

_WRAPPER = 'int a = input_json["a"];\nint b = input_json["b"];\nint result = f(a, b);\n'


def _rec(fname, func_head_types, func_def, real_exe_wrapper=None):
    return {"fname": fname, "func_head_types": func_head_types, "func_def": func_def,
            "real_io_pairs": None, "angha_io_pairs": None,
            "real_exe_wrapper": real_exe_wrapper, "angha_exe_wrapper": None}


def test_cli_writes_per_split_and_totals(monkeypatch, tmp_path, run_main):
    train_recs = [_rec("f", "int f(int a, int b)", "int f(int a, int b) { return a; }")]
    eval_recs = [_rec("f", "int f(int a, int b)", "int f(int a, int b) { return a; }", real_exe_wrapper=_WRAPPER)]

    def fake_iter_split(split, limit=None):
        data = train_recs if split == "train_real_compilable" else eval_recs
        yield from data

    monkeypatch.setattr(sup, "iter_split", fake_iter_split)

    out_path = tmp_path / "survey.json"
    sample_dir = tmp_path / "samples"

    run_main(sup, ["--splits", "train_real_compilable", "test_real",
                   "--out", str(out_path), "--sample-dir", str(sample_dir)])

    survey = json.loads(out_path.read_text())
    assert survey["totals"]["scanned"] == 2
    assert survey["totals"]["unused_param_true"] == 2
    assert survey["per_split"]["test_real"]["opt_real_exe_wrapper_ok"] == 1
    # No exe wrapper on the train record, so nothing to adapt.
    assert survey["per_split"]["train_real_compilable"]["opt_real_exe_wrapper_ok"] == 0
    assert (sample_dir / "train_real_compilable.jsonl").exists()
    assert (sample_dir / "test_real.jsonl").exists()
