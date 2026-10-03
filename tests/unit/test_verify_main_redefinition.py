from scripts.verify import verify_main_redefinition as vmr


def test_classify_redefinition_of_main():
    assert vmr._classify("error: redefinition of 'main'") == "redefinition_of_main"


def test_classify_case_insensitive():
    assert vmr._classify("Error: Redefinition Of 'MAIN'") == "redefinition_of_main"


def test_classify_other_error_when_main_not_mentioned():
    assert vmr._classify("error: 'foo' undeclared") == "other_error"


def test_classify_other_error_when_redefinition_of_something_else():
    assert vmr._classify("error: redefinition of 'helper'") == "other_error"


def test_read_log_tail_missing_file_returns_empty_string(tmp_path):
    assert vmr._read_log_tail(str(tmp_path)) == ""


def test_read_log_tail_reads_existing_log(tmp_path):
    (tmp_path / "gcc_errors.log").write_text("head text " + "x" * 5000 + " tail text")
    tail = vmr._read_log_tail(str(tmp_path))
    assert tail.endswith("tail text")
    assert len(tail) == 4000
