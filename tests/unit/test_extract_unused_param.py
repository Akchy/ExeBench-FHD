from scripts.extract import extract_unused_param as eup


def test_split_top_level_ignores_nested_commas():
    assert eup.split_top_level("int a, char *b, int (*fn)(int, int)") == [
        "int a", " char *b", " int (*fn)(int, int)",
    ]


def test_extract_param_list_takes_last_top_level_group():
    header = '__attribute__((unused)) int f(int a, int b)'
    assert eup.extract_param_list(header) == "int a, int b"


def test_extract_param_list_no_parens_returns_none():
    assert eup.extract_param_list("not a header") is None


def test_param_name_scalar():
    assert eup.param_name("int a") == "a"


def test_param_name_pointer():
    assert eup.param_name("const char *msg") == "msg"


def test_param_name_function_pointer_uses_inner_group():
    assert eup.param_name("int (*fn)(const char *file, int flag)") == "fn"


def test_param_name_void_or_ellipsis_or_empty_is_none():
    assert eup.param_name("void") is None
    assert eup.param_name("...") is None
    assert eup.param_name("") is None


def test_param_name_array_with_integer_literal_suffix_picks_real_identifier():
    assert eup.param_name("long A[((4L-2L)+2L)*16L]") == "A"


def test_parse_params_void_no_arg_is_empty_list():
    assert eup.parse_params({"func_head_types": "int f(void)"}) == []
    assert eup.parse_params({"func_head_types": "int f()"}) == []


def test_parse_params_strips_trailing_comment_before_extracting_name():
    header = 'int f(int nMs /* Milliseconds until "stop time" */, int pErr)'
    params = eup.parse_params({"func_head_types": header})
    assert [p["name"] for p in params] == ["nMs", "pErr"]


def test_parse_params_falls_back_to_func_head():
    params = eup.parse_params({"func_head": "int f(int x)"})
    assert [p["name"] for p in params] == ["x"]


def test_function_body_excludes_param_list_and_kr_decls():
    fdef = "int f(a, b)\n    int a;\n    int b;\n{\n    return a + b;\n}"
    body = eup.function_body(fdef)
    assert body.startswith("{")
    assert "int a;" not in body


def test_find_unused_params_detects_unreferenced_param():
    rec = {"fname": "f", "func_head_types": "int f(int a, int b)",
           "func_def": "int f(int a, int b) { return a; }"}
    unused = eup.find_unused_params(rec)
    assert [p["name"] for p in unused] == ["b"]
    assert unused[0]["index"] == 1


def test_find_unused_params_main_is_always_empty():
    rec = {"fname": "main", "func_head_types": "int main(int argc, char **argv)",
           "func_def": "int main(int argc, char **argv) { return 0; }"}
    assert eup.find_unused_params(rec) == []


def test_find_unused_params_word_boundary_avoids_substring_false_use():
    rec = {"fname": "f", "func_head_types": "int f(int a, int b)",
           "func_def": "int f(int a, int b) { int bx = a; return bx; }"}
    assert [p["name"] for p in eup.find_unused_params(rec)] == ["b"]


def test_unused_param_bool_wraps_find_unused_params():
    used = {"fname": "f", "func_head_types": "int f(int a)", "func_def": "int f(int a) { return a; }"}
    unused = {"fname": "f", "func_head_types": "int f(int a)", "func_def": "int f(int a) { return 0; }"}
    assert eup.unused_param(used) is False
    assert eup.unused_param(unused) is True

