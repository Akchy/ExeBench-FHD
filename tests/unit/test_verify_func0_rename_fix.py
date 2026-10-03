from scripts.verify import verify_func0_rename_fix as vfrf


def test_build_c_func_prepends_deps_when_present():
    rec = {"real_deps": "typedef int foo_t;", "angha_deps": "", "func_def": "int f(void){return 0;}"}
    assert vfrf._build_c_func(rec) == "typedef int foo_t;\nint f(void){return 0;}"


def test_build_c_func_no_deps_returns_func_def_only():
    rec = {"real_deps": "", "angha_deps": "", "func_def": "int f(void){return 0;}"}
    assert vfrf._build_c_func(rec) == "int f(void){return 0;}"


def test_classify_redefinition_of_main():
    assert vfrf._classify("error: redefinition of 'main'") == "redefinition_of_main"


def test_classify_other_error_for_unrelated_failure():
    assert vfrf._classify("error: 'undefined_symbol' undeclared") == "other_error"


def test_read_log_tail_missing_file_returns_empty_string(tmp_path):
    assert vfrf._read_log_tail(str(tmp_path)) == ""


def test_read_log_tail_reads_existing_log(tmp_path):
    (tmp_path / "gcc_errors.log").write_text("some gcc error text")
    assert vfrf._read_log_tail(str(tmp_path)) == "some gcc error text"
