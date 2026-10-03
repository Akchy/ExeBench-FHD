from collections import Counter

from scripts.survey import survey_unused_param_wrappers as supw


def test_try_adapt_reports_ok_for_a_working_wrapper():
    rec = {
        "fname": "add",
        "real_exe_wrapper": 'int a = input_json["a"];\nint b = input_json["b"];\nint result = add(a, b);\n',
    }
    outcomes = supw.try_adapt(rec, ["b"])
    assert outcomes == {"real_exe_wrapper": "ok"}


def test_try_adapt_skips_empty_wrapper_fields():
    rec = {"fname": "add", "real_exe_wrapper": "", "angha_exe_wrapper": None}
    assert supw.try_adapt(rec, ["b"]) == {}


def test_try_adapt_reports_adapt_error_with_stage_prefix():
    rec = {"fname": "add", "real_exe_wrapper": "no call site or json read here"}
    outcomes = supw.try_adapt(rec, ["b"])
    assert outcomes["real_exe_wrapper"].startswith("adapt_error:json_read:")


def test_try_adapt_shape_counter_updated_on_success():
    rec = {
        "fname": "add",
        "real_exe_wrapper": 'int a = input_json["a"];\nint b = input_json["b"];\nint result = add(a, b);\n',
    }
    counter = Counter()
    outcomes = supw.try_adapt(rec, ["b"], shape_counter=counter)
    assert outcomes == {"real_exe_wrapper": "ok"}
    assert counter == Counter({"plain_scalar": 1})


def test_try_adapt_shape_counter_not_updated_on_failure():
    rec = {"fname": "add", "real_exe_wrapper": "no call site or json read here"}
    counter = Counter()
    supw.try_adapt(rec, ["b"], shape_counter=counter)
    assert counter == Counter()


def test_try_adapt_shape_counter_aggregates_across_both_wrapper_fields():
    # Same record, unused param found via a get_cast shape in BOTH the
    # real and angha wrappers - counter should tally one per field.
    wrapper = 'char c = input_json["c"].get<char>();\nint result = f(c);\n'
    rec = {"fname": "f", "real_exe_wrapper": wrapper, "angha_exe_wrapper": wrapper}
    counter = Counter()
    outcomes = supw.try_adapt(rec, ["c"], shape_counter=counter)
    assert outcomes == {"real_exe_wrapper": "ok", "angha_exe_wrapper": "ok"}
    assert counter == Counter({"get_cast_scalar": 2})


def test_try_adapt_shape_counter_none_is_backward_compatible():
    rec = {
        "fname": "add",
        "real_exe_wrapper": 'int a = input_json["a"];\nint b = input_json["b"];\nint result = add(a, b);\n',
    }
    assert supw.try_adapt(rec, ["b"]) == {"real_exe_wrapper": "ok"}


def test_survey_shape_distribution_in_output(monkeypatch, tmp_path):
    records = [
        {
            "fname": "add",
            "real_exe_wrapper": 'int a = input_json["a"];\nint b = input_json["b"];\nint result = add(a, b);\n',
            "angha_exe_wrapper": "",
            "func_head_types": "int add(int a, int b)",
            "func_def": "int add(int a, int b) {\n    return a;\n}\n",
        },
        {
            "fname": "f",
            "real_exe_wrapper": 'char c = input_json["c"].get<char>();\nint result = f(c);\n',
            "angha_exe_wrapper": "",
            "func_head_types": "int f(char c)",
            "func_def": "int f(char c) {\n    return 0;\n}\n",
        },
    ]
    monkeypatch.setattr(supw, "iter_split", lambda split, limit=None: iter(records))
    monkeypatch.setattr(
        supw, "find_unused_params",
        lambda rec: [{"index": 0, "name": "b" if rec["fname"] == "add" else "c", "type": "int"}],
    )

    result = supw.survey("test_real", n=10, limit=None, sample_dir=str(tmp_path))

    assert result["shape_distribution"] == {"plain_scalar": 1, "get_cast_scalar": 1}
    assert result["shape_distribution_pct"] == {"plain_scalar": 50.0, "get_cast_scalar": 50.0}


def test_survey_shape_distribution_empty_when_no_matches(monkeypatch, tmp_path):
    monkeypatch.setattr(supw, "iter_split", lambda split, limit=None: iter([]))
    result = supw.survey("test_real", n=10, limit=None, sample_dir=str(tmp_path))
    assert result["shape_distribution"] == {}
    assert result["shape_distribution_pct"] == {}
