import pytest

from scripts.extract import adapt_unused_param_wrapper as auw


def test_drop_call_args_removes_named_arg():
    src = "int result = add(a, b, c);\n"
    out = auw._drop_call_args(src, "add", ["b"])
    assert out == "int result = add(a, c);\n"


def test_drop_call_args_noop_when_target_not_called():
    src = "int result = other(a, b);\n"
    assert auw._drop_call_args(src, "add", ["b"]) == src


def test_drop_def_param_removes_matching_parameter():
    src = "void write_output(int a, double *rates, int result) {\n    return;\n}\n"
    out = auw._drop_def_param(src, "write_output", ["rates"])
    assert out.startswith("void write_output(int a, int result) {")


def test_drop_def_param_noop_when_not_defined():
    src = "int main() { return 0; }\n"
    assert auw._drop_def_param(src, "write_output", ["x"]) == src


def test_drop_json_read_scalar_shape():
    src = 'int a = input_json["a"];\nint b = input_json["b"];\n'
    out = auw._drop_json_read(src, "a")
    assert 'input_json["a"]' not in out
    assert 'input_json["b"]' in out


def test_drop_json_read_get_cast_shape():
    src = 'char c = input_json["c"].get<char>();\nint x = 1;\n'
    out = auw._drop_json_read(src, "c")
    assert "get<char>" not in out
    assert "int x = 1;" in out


def test_drop_json_read_string_shape():
    src = 'char* text = strdup(input_json["text"].get<std::string>().c_str());\n'
    out = auw._drop_json_read(src, "text")
    assert out == ""


def test_drop_json_read_vector_shape():
    src = (
        "std::vector<int> input_temp_1_vec;\n"
        'for (auto& elem : input_json["arr"]) {\n'
        "    input_temp_1_vec.push_back(elem);\n"
        "}\n"
        "int* arr = &input_temp_1_vec[0];\n"
    )
    out = auw._drop_json_read(src, "arr")
    assert out == ""


def test_drop_json_read_struct_by_value_shape():
    src = (
        'int x__dummy = input_json["x"]["dummy"];\n'
        "struct foo x = {x__dummy};\n"
    )
    out = auw._drop_json_read(src, "x")
    assert out == ""


def test_drop_json_read_single_struct_pointer_shape():
    src = (
        'int dev_pointer__dummy = input_json["dev"]["dummy"];\n'
        "struct block_device dev_pointer = {dev_pointer__dummy};\n"
        "struct block_device* dev = &dev_pointer;\n"
    )
    out = auw._drop_json_read(src, "dev")
    assert out == ""


def test_drop_json_read_unknown_shape_raises():
    src = "some_other_line();\n"
    with pytest.raises(auw.WrapperAdaptError) as exc_info:
        auw._drop_json_read(src, "missing")
    assert exc_info.value.name == "missing"
    assert exc_info.value.stage == "json_read"


def test_drop_output_echo_scalar_shape_removed():
    src = 'output_json["result"] = result;\noutput_json["other"] = other;\n'
    out = auw._drop_output_echo(src, "result")
    assert 'output_json["result"]' not in out
    assert 'output_json["other"]' in out


def test_drop_output_echo_missing_shape_is_noop_not_error():
    src = "// no echo here\n"
    assert auw._drop_output_echo(src, "missing") == src


def test_check_no_dangling_refs_raises_on_real_leftover_reference():
    src = "void write_output(int a) {\n    output_json[\"a\"] = a + 1;\n}\n"
    with pytest.raises(auw.WrapperAdaptError):
        auw._check_no_dangling_refs(src, "write_output", ["a"])


def test_check_no_dangling_refs_ignores_member_access_and_string_literal():
    src = 'void write_output(int other) {\n    x.dummy = 1;\n    y->dummy = 2;\n    z["dummy"] = 3;\n}\n'
    auw._check_no_dangling_refs(src, "write_output", ["dummy"])  # must not raise


def test_check_no_dangling_refs_no_definition_is_noop():
    src = "int main() { return 0; }\n"
    auw._check_no_dangling_refs(src, "write_output", ["x"])  # must not raise


def test_adapt_exe_wrapper_no_unused_names_is_noop():
    src = "anything at all"
    assert auw.adapt_exe_wrapper(src, "f", []) == src


def test_adapt_exe_wrapper_end_to_end_simple_scalar():
    src = (
        'int a = input_json["a"];\n'
        'int b = input_json["b"];\n'
        "int result = add(a, b);\n"
        'output_json["result"] = result;\n'
    )
    out = auw.adapt_exe_wrapper(src, "add", ["b"])
    assert 'input_json["b"]' not in out
    assert "add(a);" in out


def test_adapt_exe_wrapper_falls_back_to_bench_suffix_call_site():
    src = (
        'int x = input_json["x"];\n'
        'int y = input_json["y"];\n'
        "int result = pow__bench(x, y);\n"
    )
    out = auw.adapt_exe_wrapper(src, "pow", ["y"])
    assert "pow__bench(x);" in out


def test_adapt_exe_wrapper_missing_call_site_raises():
    src = 'int a = input_json["a"];\n'
    with pytest.raises(auw.WrapperAdaptError) as exc_info:
        auw.adapt_exe_wrapper(src, "nonexistent_fn", ["a"])
    assert exc_info.value.stage == "call_site"


def test_drop_json_read_logs_shape_name_for_scalar():
    src = 'int a = input_json["a"];\n'
    log = []
    auw._drop_json_read(src, "a", shape_log=log)
    assert log == ["plain_scalar"]


def test_drop_json_read_logs_shape_name_for_vector():
    src = (
        "std::vector<int> input_temp_1_vec;\n"
        'for (auto& elem : input_json["arr"]) {\n'
        "    input_temp_1_vec.push_back(elem);\n"
        "}\n"
        "int* arr = &input_temp_1_vec[0];\n"
    )
    log = []
    auw._drop_json_read(src, "arr", shape_log=log)
    assert log == ["vector"]


def test_drop_json_read_logs_shape_name_for_get_cast_scalar():
    src = 'char c = input_json["c"].get<char>();\n'
    log = []
    auw._drop_json_read(src, "c", shape_log=log)
    assert log == ["get_cast_scalar"]


def test_drop_json_read_logs_shape_name_for_string_strdup():
    src = 'char* text = strdup(input_json["text"].get<std::string>().c_str());\n'
    log = []
    auw._drop_json_read(src, "text", shape_log=log)
    assert log == ["string_strdup"]


def test_drop_json_read_logs_shape_name_for_struct_by_value():
    src = (
        'int x__dummy = input_json["x"]["dummy"];\n'
        "struct foo x = {x__dummy};\n"
    )
    log = []
    auw._drop_json_read(src, "x", shape_log=log)
    assert log == ["struct_by_value"]


def test_drop_json_read_logs_shape_name_for_single_struct_pointer():
    src = (
        'int dev_pointer__dummy = input_json["dev"]["dummy"];\n'
        "struct block_device dev_pointer = {dev_pointer__dummy};\n"
        "struct block_device* dev = &dev_pointer;\n"
    )
    log = []
    auw._drop_json_read(src, "dev", shape_log=log)
    assert log == ["single_struct_pointer"]


def test_drop_json_read_without_shape_log():
    src = 'int a = input_json["a"];\n'
    assert auw._drop_json_read(src, "a") == ""


def test_drop_json_read_shape_log_untouched_on_error():
    log = []
    with pytest.raises(auw.WrapperAdaptError):
        auw._drop_json_read("no match here", "missing", shape_log=log)
    assert log == []


def test_adapt_exe_wrapper_shape_log_records_one_entry_per_unused_param():
    src = (
        'int a = input_json["a"];\n'
        'char c = input_json["c"].get<char>();\n'
        "int result = add(a, c);\n"
    )
    log = []
    out = auw.adapt_exe_wrapper(src, "add", ["a", "c"], shape_log=log)
    assert log == ["plain_scalar", "get_cast_scalar"]
    assert 'input_json["a"]' not in out
    assert 'input_json["c"]' not in out


def test_adapt_exe_wrapper_shape_log_records_libc_collision_rename():
    src = (
        'int x = input_json["x"];\n'
        'int y = input_json["y"];\n'
        "int result = pow__bench(x, y);\n"
    )
    log = []
    auw.adapt_exe_wrapper(src, "pow", ["y"], shape_log=log)
    assert log == ["plain_scalar", "libc_collision_rename"]


def test_adapt_exe_wrapper_shape_log_default_matches_explicit_none():
    src = (
        'int a = input_json["a"];\n'
        'int b = input_json["b"];\n'
        "int result = add(a, b);\n"
        'output_json["result"] = result;\n'
    )
    out_default = auw.adapt_exe_wrapper(src, "add", ["b"])
    out_explicit_none = auw.adapt_exe_wrapper(src, "add", ["b"], shape_log=None)
    assert out_default == out_explicit_none
    assert 'input_json["b"]' not in out_default
    assert "add(a);" in out_default


def test_adapt_exe_wrapper_drops_write_output_param_and_call():
    src = (
        'int a = input_json["a"];\n'
        'double b_val = 0;\n'
        'std::vector<double> input_temp_1_vec;\n'
        'for (auto& elem : input_json["rates"]) {\n'
        '    input_temp_1_vec.push_back(elem);\n'
        '}\n'
        'double* rates = &input_temp_1_vec[0];\n'
        "int result = compute(a, rates);\n"
        "void write_output(int a, double* rates, int result) {\n"
        '    output_json["result"] = result;\n'
        "}\n"
        "write_output(a, rates, result);\n"
    )
    out = auw.adapt_exe_wrapper(src, "compute", ["rates"])
    assert 'input_json["rates"]' not in out
    assert "compute(a);" in out
    assert "write_output(int a, int result) {" in out
    assert "write_output(a, result);" in out
