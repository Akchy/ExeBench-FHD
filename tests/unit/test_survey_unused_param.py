from scripts.survey import survey_unused_param as sup

_WRAPPER = 'int a = input_json["a"];\nint b = input_json["b"];\nint result = f(a, b);\n'


def _rec(fname="f", func_head_types="int f(int a, int b)", func_def=None, real_io_pairs=None,
         real_exe_wrapper=None):
    return {
        "fname": fname,
        "func_head_types": func_head_types,
        "func_def": func_def or "int f(int a, int b) { return a; }",
        "real_io_pairs": real_io_pairs,
        "angha_io_pairs": None,
        "real_exe_wrapper": real_exe_wrapper,
        "angha_exe_wrapper": None,
    }


def test_survey_split_train_counts_unused_and_main_skipped(monkeypatch):
    records = [
        _rec(fname="main", func_head_types="int main(int argc, char **argv)",
             func_def="int main(int argc, char **argv) { return 0; }"),
        _rec(fname="f", func_def="int f(int a, int b) { return a; }"),
        _rec(fname="g", func_head_types="int g(int a, int b)", func_def="int g(int a, int b) { return a + b; }"),
    ]
    monkeypatch.setattr(sup, "iter_split", lambda split, limit=None: iter(records))

    result = sup.survey_split("train_real_compilable")
    assert result["scanned"] == 3
    assert result["main_skipped"] == 1
    assert result["unused_param_true"] == 1
    assert result["unused_param_count_hist"] == {1: 1}
    # No exe wrapper on the record, so nothing to adapt.
    assert result["opt_real_exe_wrapper_ok"] == 0
    assert result["opt_real_exe_wrapper_failed"] == 0


def test_survey_split_eval_counts_wrapper_adaptation_outcomes(monkeypatch):
    records = [
        _rec(real_exe_wrapper=_WRAPPER),  # adapts cleanly
        _rec(fname="broken", func_head_types="int broken(int a, int b)",
             func_def="int broken(int a, int b) { return a; }",
             real_exe_wrapper="no recognizable shape"),  # fails to adapt
    ]
    monkeypatch.setattr(sup, "iter_split", lambda split, limit=None: iter(records))

    result = sup.survey_split("test_real")
    assert result["is_eval_split"] is True
    assert result["opt_real_exe_wrapper_ok"] == 1
    assert result["opt_real_exe_wrapper_failed"] == 1


def test_survey_split_train_counts_wrapper_adaptation_outcomes_too(monkeypatch):
    records = [
        _rec(real_exe_wrapper=_WRAPPER),  # adapts cleanly
        _rec(fname="broken", func_head_types="int broken(int a, int b)",
             func_def="int broken(int a, int b) { return a; }",
             real_exe_wrapper="no recognizable shape"),  # fails to adapt
    ]
    monkeypatch.setattr(sup, "iter_split", lambda split, limit=None: iter(records))

    result = sup.survey_split("train_real_compilable")
    assert result["is_eval_split"] is False
    assert result["opt_real_exe_wrapper_ok"] == 1
    assert result["opt_real_exe_wrapper_failed"] == 1


def test_survey_split_writes_capped_sample_file(monkeypatch, tmp_path):
    records = [_rec(fname=f"f{i}", func_head_types=f"int f{i}(int a, int b)",
                     func_def=f"int f{i}(int a, int b) {{ return a; }}") for i in range(5)]
    monkeypatch.setattr(sup, "iter_split", lambda split, limit=None: iter(records))

    sample_path = tmp_path / "sample.jsonl"
    sup.survey_split("train_real_compilable", sample_n=2, sample_path=str(sample_path))

    assert len(sample_path.read_text().splitlines()) == 2
