from scripts.preprocess import helper_func


def test_extract_c_function_returns_full_body_with_nested_braces():
    code = """
int helper(int x) {
    return x;
}

int target(int a, int b) {
    if (a > b) {
        return a;
    }
    return b;
}
"""
    body, length = helper_func.extract_c_function(code, "target")
    assert body.startswith("int target(int a, int b) {")
    assert body.rstrip().endswith("}")
    assert "if (a > b) {" in body
    assert length == len(body)


def test_extract_c_function_drops_blank_and_line_comment_lines():
    code = """
int target(void) {
    // a comment

    return 0;
}
"""
    body, _ = helper_func.extract_c_function(code, "target")
    assert "//" not in body
    assert "" not in body.split("\n")


def test_extract_c_function_missing_function_returns_false_and_zero():
    code = "int other(void) { return 0; }"
    assert helper_func.extract_c_function(code, "target") == (False, 0)


def test_extract_function_headers_returns_first_header_only():
    code = """
int add(int a, int b) {
    return a + b;
}

int sub(int a, int b) {
    return a - b;
}
"""
    assert helper_func.extract_function_headers(code) == "int add(int a, int b)"


def test_extract_function_headers_strips_comments_before_matching():
    code = """
/* a block comment with a fake header int fake(void) { */
// a line comment
int real(void) {
    return 0;
}
"""
    assert helper_func.extract_function_headers(code) == "int real(void)"


def test_extract_function_headers_no_function_returns_false():
    assert helper_func.extract_function_headers("int x = 0;") is False


def test_parse_header_for_datatypes_multiple_args_strips_names():
    parsed = helper_func.parse_header_for_datatypes("int add(int a, const char *msg)")
    assert parsed["return_type"] == "int"
    assert parsed["arg_types"] == ["int", "const char *"]


def test_parse_header_for_datatypes_void_args_is_empty_list():
    parsed = helper_func.parse_header_for_datatypes("int noop(void)")
    assert parsed["arg_types"] == []


def test_parse_header_for_datatypes_no_args_is_empty_list():
    parsed = helper_func.parse_header_for_datatypes("int noop()")
    assert parsed["arg_types"] == []


def test_parse_header_for_datatypes_unparseable_header_returns_none():
    assert helper_func.parse_header_for_datatypes("not a valid header") is None


def test_get_all_optz_flag_present_negates_every_other_flag(monkeypatch):
    monkeypatch.setitem(
        helper_func._GCC_OPTS_CACHE, helper_func._GCC_OPTS_PATH,
        {"O2": ["-ftree-vectorize", "-finline-functions", "-fpeel-loops"]},
    )
    out = helper_func.get_all_optz("O2", "-finline-functions")
    assert sorted(out) == ["-fno-peel-loops", "-fno-tree-vectorize"]


def test_get_all_optz_flag_none_negates_every_flag(monkeypatch):
    monkeypatch.setitem(
        helper_func._GCC_OPTS_CACHE, helper_func._GCC_OPTS_PATH,
        {"O2": ["-ftree-vectorize", "-finline-functions"]},
    )
    out = helper_func.get_all_optz("O2", "none")
    assert sorted(out) == ["-fno-inline-functions", "-fno-tree-vectorize"]


def test_get_all_optz_unknown_flag_returns_false(monkeypatch):
    monkeypatch.setitem(
        helper_func._GCC_OPTS_CACHE, helper_func._GCC_OPTS_PATH,
        {"O2": ["-ftree-vectorize"]},
    )
    assert helper_func.get_all_optz("O2", "-fsomething-else") is False


def test_get_all_optz_unknown_opt_level_returns_false(monkeypatch):
    monkeypatch.setitem(
        helper_func._GCC_OPTS_CACHE, helper_func._GCC_OPTS_PATH, {"O2": []},
    )
    assert helper_func.get_all_optz("O9", "none") is False
