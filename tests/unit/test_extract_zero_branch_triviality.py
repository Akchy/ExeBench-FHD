from scripts.extract import extract_zero_branch_triviality as ezbt


def test_extract_body_returns_text_between_outer_braces():
    assert ezbt.extract_body("int f(void) {\n    return 0;\n}\n") == "\n    return 0;\n"


def test_extract_body_no_brace_returns_none():
    assert ezbt.extract_body("int f(int x);") is None


def test_extract_body_matches_outermost_braces_only():
    body = ezbt.extract_body("int f(void) { if (1) { x++; } return x; }")
    assert body.strip() == "if (1) { x++; } return x;"


def test_classify_empty_body():
    assert ezbt.classify({"func_def": "int f(void) {\n}\n"}) == "empty"


def test_classify_empty_body_with_only_a_comment():
    assert ezbt.classify({"func_def": "int f(void) {\n    // nothing here\n}\n"}) == "empty"


def test_classify_return_only_bare():
    assert ezbt.classify({"func_def": "int f(void) {\n    return;\n}\n"}) == "return_only"


def test_classify_return_only_with_expression():
    assert ezbt.classify({"func_def": "int f(void) {\n    return 0;\n}\n"}) == "return_only"


def test_classify_two_loc_not_return_only():
    fdef = "int f(int x) {\n    x++;\n    return x;\n}\n"
    assert ezbt.classify({"func_def": fdef}) == "two_loc"


def test_classify_gap_straightline_no_branches():
    fdef = "int f(int x) {\n    x = x + 1;\n    x = x * 2;\n    x = x - 3;\n    return x;\n}\n"
    assert ezbt.classify({"func_def": fdef}) == "gap"


def test_classify_multi_line_body_with_branch_is_unclassified():
    fdef = "int f(int x) {\n    if (x > 0) {\n        x = x + 1;\n    }\n    x = x * 2;\n    return x;\n}\n"
    assert ezbt.classify({"func_def": fdef}) is None


def test_classify_ternary_counts_as_a_branch():
    fdef = "int f(int x) {\n    int y = x > 0 ? 1 : 2;\n    y = y + 1;\n    y = y * 2;\n    return y;\n}\n"
    assert ezbt.classify({"func_def": fdef}) is None


def test_classify_short_body_with_branch_still_caught_by_two_loc():
    assert ezbt.classify({"func_def": "int f(int x) {\n    if (x) return 1;\n}\n"}) == "two_loc"


def test_classify_no_brace_is_unclassified():
    assert ezbt.classify({"func_def": "int f(int x);"}) is None


def test_classify_missing_func_def_is_unclassified():
    assert ezbt.classify({}) is None
