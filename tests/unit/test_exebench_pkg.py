from scripts import exebench as eb


def test_diff_io_scalars_equal_and_not_equal():
    assert eb.diff_io(1, 1) is True
    assert eb.diff_io(1, 2) is False


def test_diff_io_type_mismatch_is_false():
    assert eb.diff_io(1, 1.0) is False
    assert eb.diff_io("1", 1) is False


def test_diff_io_floats_use_isclose_tolerance():
    assert eb.diff_io(0.1 + 0.2, 0.3) is True
    assert eb.diff_io(1.0, 2.0) is False


def test_diff_io_lists_recurse_and_check_length():
    assert eb.diff_io([1, 2, 3], [1, 2, 3]) is True
    assert eb.diff_io([1, 2], [1, 2, 3]) is False
    assert eb.diff_io([1, 2], [1, 3]) is False


def test_diff_io_dicts_recurse_and_require_observed_keys_present():
    assert eb.diff_io({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3}) is True
    assert eb.diff_io({"a": 1, "b": 2}, {"a": 1}) is False
    assert eb.diff_io({"a": 1}, {"a": 2}) is False


def test_diff_io_nested_structures():
    observed = {"a": [1, {"b": 2.0}]}
    expected = {"a": [1, {"b": 2.0}], "extra": "ignored"}
    assert eb.diff_io(observed, expected) is True


def test_exebench_dict_to_dict_zips_var_and_value():
    out = eb.exebench_dict_to_dict({"var": ["x", "y"], "value": ["3", "4.5"]})
    assert out == {"x": 3, "y": 4.5}


def test_exebench_dict_to_dict_literal_evals_string_values():
    out = eb.exebench_dict_to_dict({"var": ["s"], "value": ["'hello'"]})
    assert out == {"s": "hello"}


def test_exebench_dict_to_dict_empty_is_empty_dict():
    assert eb.exebench_dict_to_dict({"var": [], "value": []}) == {}


def test_compile_error_is_an_exception():
    assert issubclass(eb.CompileError, Exception)
