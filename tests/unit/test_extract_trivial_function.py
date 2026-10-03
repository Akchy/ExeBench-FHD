from scripts.extract import extract_trivial_function as etf


def test_flags_empty_body(record_factory):
    rec = record_factory(func_def="int f(int x) {\n}\n")
    assert etf.is_trivial_function(rec) is True


def test_flags_return_only_body(record_factory):
    rec = record_factory(func_def="int f(int x) {\n    return x;\n}\n")
    assert etf.is_trivial_function(rec) is True


def test_flags_two_loc_body(record_factory):
    rec = record_factory(func_def="int f(int x) {\n    x++;\n    return x;\n}\n")
    assert etf.is_trivial_function(rec) is True


def test_does_not_flag_body_with_branches(record_factory):
    rec = record_factory(func_def=(
        "int f(int x) {\n"
        "    if (x > 0) {\n"
        "        return x;\n"
        "    }\n"
        "    return -x;\n"
        "}\n"
    ))
    assert etf.is_trivial_function(rec) is False


def test_does_not_flag_longer_zero_branch_body(record_factory):
    rec = record_factory(func_def=(
        "int f(int x) {\n"
        "    int a = x + 1;\n"
        "    int b = a + 1;\n"
        "    int c = b + 1;\n"
        "    return c;\n"
        "}\n"
    ))
    assert etf.is_trivial_function(rec) is False


def test_unparseable_body_is_not_flagged(record_factory):
    rec = record_factory(func_def="int f(int x)")
    assert etf.is_trivial_function(rec) is False


def test_process_record_adds_trivial_function_true_for_a_match(record_factory):
    rec = record_factory(func_def="int f(int x) {\n    return x;\n}\n")
    out = etf.process_record(rec)
    assert out["trivial_function"] is True


def test_process_record_adds_trivial_function_false_by_default(record_factory):
    rec = record_factory(func_def=(
        "int f(int x) {\n"
        "    if (x > 0) {\n"
        "        return x;\n"
        "    }\n"
        "    return -x;\n"
        "}\n"
    ))
    out = etf.process_record(rec)
    assert out["trivial_function"] is False


def test_process_record_does_not_mutate_the_original_record(record_factory):
    rec = record_factory(func_def="int f(int x) {\n    return x;\n}\n")
    etf.process_record(rec)
    assert "trivial_function" not in rec


def test_process_record_preserves_every_original_field(record_factory):
    rec = record_factory(func_def="int f(int x) {\n    return x;\n}\n")
    out = etf.process_record(rec)
    for key, value in rec.items():
        assert out[key] == value
