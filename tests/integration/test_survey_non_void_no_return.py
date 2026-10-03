import json

from scripts.survey import survey_non_void_no_return as snvr


def _rec(fname, signature, func_def):
    return {"fname": fname, "signature": signature, "func_def": func_def,
            "real_io_pairs": None, "angha_io_pairs": None,
            "real_exe_wrapper": None, "angha_exe_wrapper": None}


def test_cli_writes_per_split_and_totals(monkeypatch, tmp_path, run_main):
    train_recs = [_rec("f", ["int", "int"], "int f(int x) {\n    x++;\n}\n")]
    eval_recs = [_rec("g", ["int", "int"], "int g(int x) {\n    return x;\n}\n")]

    def fake_iter_split(split, limit=None):
        data = train_recs if split == "train_real_compilable" else eval_recs
        yield from data

    monkeypatch.setattr(snvr, "iter_split", fake_iter_split)

    out_path = tmp_path / "survey.json"
    sample_dir = tmp_path / "samples"

    run_main(snvr, ["--splits", "train_real_compilable", "test_real",
                     "--out", str(out_path), "--sample-dir", str(sample_dir)])

    survey = json.loads(out_path.read_text())
    assert survey["totals"]["scanned"] == 2
    assert survey["totals"]["non_void_no_return_true"] == 1
    assert survey["per_split"]["train_real_compilable"]["non_void_no_return_true"] == 1
    assert survey["per_split"]["test_real"]["non_void_no_return_true"] == 0
    assert (sample_dir / "train_real_compilable.jsonl").exists()
    assert (sample_dir / "test_real.jsonl").exists()
