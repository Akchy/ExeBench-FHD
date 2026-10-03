from scripts import exebench_eval as ee


def test_extract_function_headers_exeb_modern_header():
    header, new_code = ee.extract_function_headers_exeb("int add(int a, int b) {\n    return a + b;\n}")
    assert header == "int add(int a, int b)"
    assert new_code == "int add(int a, int b) {\n    return a + b;\n}"


def test_extract_function_headers_exeb_strips_static_from_new_code():
    header, new_code = ee.extract_function_headers_exeb("static int helper(int x) {\n    return x;\n}")
    assert header == "static int helper(int x)"  # original header text preserved
    assert "static" not in new_code
    assert new_code == " int helper(int x) {\n    return x;\n}"


def test_extract_function_headers_exeb_pointer_return_type():
    header, new_code = ee.extract_function_headers_exeb("char *make_str(int n) {\n    return 0;\n}")
    assert header == "char *make_str(int n)"


def test_extract_function_headers_exeb_kr_style_multiline_decls_rewritten():
    src = "int add(a, b)\n    int a;\n    int b;\n{\n    return a + b;\n}"
    header, new_code = ee.extract_function_headers_exeb(src)
    assert header == "int add(int a, int b)"
    assert new_code == "int add(int a, int b)\n{\n    return a + b;\n}"


def test_extract_function_headers_exeb_kr_style_comma_joined_decl():
    src = "int add(a, b)\nint a, b;\n{\n    return a + b;\n}"
    header, new_code = ee.extract_function_headers_exeb(src)
    assert header == "int add(int a, int b)"


def test_extract_function_headers_exeb_no_function_returns_false_pair():
    assert ee.extract_function_headers_exeb("no function here at all") == (False, False)


def test_rewrite_kr_header_unmatched_declaration_returns_none():
    # "b" is never declared, so the rewrite can't be done safely.
    assert ee._rewrite_kr_header("int add(a, b)", "int a;") is None


def test_rewrite_kr_header_non_identifier_param_list_returns_none():
    assert ee._rewrite_kr_header("int add(a + 1)", "int a;") is None


def test_rewrite_kr_header_empty_decl_block_returns_none():
    assert ee._rewrite_kr_header("int add(a, b)", "   ") is None


def test_rewrite_kr_header_pointer_parameter():
    out = ee._rewrite_kr_header("int add(a, b)", "int *a;\nint b;\n")
    assert out == "int add(int * a, int b)"
