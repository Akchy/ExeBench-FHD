from scripts.extract import extract_missing_return_type as emrt


def test_classify_kr_implicit_int_style():
    rec = {"func_head": "int main()", "fname": "main", "func_def": "main()\n{\n    return 0;\n}\n"}
    assert emrt.classify(rec) == "missing_int_implicit"


def test_classify_missing_non_int_return_type():
    rec = {"func_head": "void sUalrm()", "fname": "sUalrm", "func_def": " sUalrm(){\n    exit(0);\n}\n"}
    assert emrt.classify(rec) == "missing_non_int"


def test_classify_fname_corrupted():
    rec = {"func_head": "unsigned int format_sector(int cyl)", "fname": "format_sector",
           "func_def": "rmat_sector(unsigned int cyl){\n    return 0;\n}\n"}
    assert emrt.classify(rec) == "fname_not_found"


def test_classify_normal_function_is_not_flagged():
    rec = {"func_head": "int add(int a, int b)", "fname": "add",
           "func_def": "int add(int a, int b){\n    return a+b;\n}\n"}
    assert emrt.classify(rec) is None


def test_classify_redundant_parens_declarator_not_flagged():
    rec = {"func_head": "int facility_ind_receive(void)", "fname": "facility_ind_receive",
           "func_def": "static int ((facility_ind_receive))(void){\n    return 0;\n}\n"}
    assert emrt.classify(rec) is None


def test_classify_empty_func_def_or_fname_returns_none():
    assert emrt.classify({"func_head": "int f()", "fname": "f", "func_def": ""}) is None
    assert emrt.classify({"func_head": "int f()", "fname": "", "func_def": "f(){return 0;}"}) is None


def test_strip_quals_strips_multiple_stacked_qualifiers():
    assert emrt._strip_quals("static inline ") == ""
    assert emrt._strip_quals("extern __attribute__((unused)) ") == ""
    assert emrt._strip_quals("int") == "int"


def test_head_ret_type_extracts_prefix_before_fname():
    rec = {"func_head": "char *make_str(int n)", "fname": "make_str"}
    assert emrt._head_ret_type(rec) == "char *"


def test_head_ret_type_missing_head_or_fname_returns_none():
    assert emrt._head_ret_type({"func_head": "", "fname": "f"}) is None
    assert emrt._head_ret_type({"func_head": "int f()", "fname": ""}) is None


def test_find_def_split_primary_match():
    assert emrt._find_def_split("int add(int a, int b){}", "add") == "int "


def test_find_def_split_fname_not_present_returns_none():
    assert emrt._find_def_split("int other(void){}", "add") is None


def test_find_def_split_tolerates_line_marker_gap():
    fdef = 'long long (\n# 301 "include/trace/events/kmem.h"\natoll)(char const *__nptr){}'
    assert emrt._find_def_split(fdef, "atoll") is not None
