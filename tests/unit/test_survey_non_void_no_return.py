from scripts.survey import survey_non_void_no_return as snvr


def test_survey_split_counts_matches(monkeypatch, record_factory):
    records = [
        record_factory(signature=["int", "int"], func_def="int f(int x) {\n    x++;\n}\n"),
        record_factory(signature=["int", "int"], func_def="int f(int x) {\n    return x + 1;\n}\n"),
        record_factory(signature=["void", "int"], func_def="void f(int x) {\n    x++;\n}\n"),
    ]
    monkeypatch.setattr(snvr, "iter_split", lambda split, limit=None: iter(records))

    result = snvr.survey_split("train_real_compilable")
    assert result["scanned"] == 3
    assert result["non_void_no_return_true"] == 1
    assert result["non_void_no_return_rate"] == 1 / 3


def test_survey_split_writes_capped_sample_file(monkeypatch, tmp_path, record_factory):
    records = [record_factory(signature=["int", "int"],
                               func_def=f"int f{i}(int x) {{\n    x++;\n}}\n") for i in range(5)]
    monkeypatch.setattr(snvr, "iter_split", lambda split, limit=None: iter(records))

    sample_path = tmp_path / "sample.jsonl"
    snvr.survey_split("train_real_compilable", sample_n=2, sample_path=str(sample_path))

    assert len(sample_path.read_text().splitlines()) == 2


def test_survey_split_empty_split_has_zero_rate(monkeypatch):
    monkeypatch.setattr(snvr, "iter_split", lambda split, limit=None: iter([]))

    result = snvr.survey_split("train_real_compilable")
    assert result["scanned"] == 0
    assert result["non_void_no_return_true"] == 0
    assert result["non_void_no_return_rate"] == 0.0
