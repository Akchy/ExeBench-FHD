from scripts.extract import extract_truncated_funcs as etf


def test_truncated_funcs_true_for_missing_non_int():
    rec = {"func_head": "void sUalrm()", "fname": "sUalrm", "func_def": " sUalrm(){\n    exit(0);\n}\n"}
    assert etf.truncated_funcs(rec) is True


def test_truncated_funcs_true_for_fname_not_found():
    rec = {"func_head": "unsigned int format_sector(int cyl)", "fname": "format_sector",
           "func_def": "rmat_sector(unsigned int cyl){\n    return 0;\n}\n"}
    assert etf.truncated_funcs(rec) is True


def test_truncated_funcs_false_for_legitimate_kr_implicit_int():
    rec = {"func_head": "int main()", "fname": "main", "func_def": "main()\n{\n    return 0;\n}\n"}
    assert etf.truncated_funcs(rec) is False


def test_truncated_funcs_false_for_normal_function():
    rec = {"func_head": "int add(int a, int b)", "fname": "add",
           "func_def": "int add(int a, int b){\n    return a+b;\n}\n"}
    assert etf.truncated_funcs(rec) is False
