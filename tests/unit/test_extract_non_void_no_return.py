from scripts.extract import extract_non_void_no_return as envnr


def test_flags_non_void_function_with_no_return(record_factory):
    rec = record_factory(signature=["int", "int"], func_def="int f(int x) {\n    x++;\n}\n")
    assert envnr.is_non_void_no_return(rec) is True


def test_does_not_flag_void_function(record_factory):
    rec = record_factory(signature=["void", "int"], func_def="void f(int x) {\n    x++;\n}\n")
    assert envnr.is_non_void_no_return(rec) is False


def test_does_not_flag_function_with_a_return_statement(record_factory):
    rec = record_factory(signature=["int", "int"], func_def="int f(int x) {\n    return x + 1;\n}\n")
    assert envnr.is_non_void_no_return(rec) is False


def test_word_boundary_does_not_match_inside_longer_identifier(record_factory):
    rec = record_factory(signature=["int", "int"],
                          func_def="int f(int x) {\n    int returned_value = x;\n}\n")
    assert envnr.is_non_void_no_return(rec) is True


def test_return_inside_string_literal_still_counts_as_a_match():
    # Known limitation: string literals are not stripped.
    rec = {"signature": ["int", "int"], "func_def": 'int f(int x) {\n    printf("return");\n}\n'}
    assert envnr.is_non_void_no_return(rec) is False


def test_missing_or_empty_signature_is_not_flagged(record_factory):
    assert envnr.is_non_void_no_return(record_factory(signature=[])) is False
    assert envnr.is_non_void_no_return(record_factory(signature=None)) is False


def test_empty_return_type_string_is_not_flagged(record_factory):
    rec = record_factory(signature=["", "int"], func_def="f(int x) {\n    x++;\n}\n")
    assert envnr.is_non_void_no_return(rec) is False


def test_main_with_no_return_is_not_flagged(record_factory):
    rec = record_factory(fname="main", signature=["int", "int", "char **"],
                          func_def="int main(int argc, char **argv) {\n    do_stuff();\n}\n")
    assert envnr.is_non_void_no_return(rec) is False


def test_main_with_a_return_statement_is_still_not_flagged(record_factory):
    rec = record_factory(fname="main", signature=["int", "int", "char **"],
                          func_def="int main(int argc, char **argv) {\n    return 0;\n}\n")
    assert envnr.is_non_void_no_return(rec) is False


def test_process_record_adds_no_return_true_for_a_match(record_factory):
    rec = record_factory(signature=["int", "int"], func_def="int f(int x) {\n    x++;\n}\n")
    out = envnr.process_record(rec)
    assert out["no_return"] is True


def test_process_record_adds_no_return_false_by_default(record_factory):
    rec = record_factory(signature=["int", "int"], func_def="int f(int x) {\n    return x + 1;\n}\n")
    out = envnr.process_record(rec)
    assert out["no_return"] is False


def test_process_record_does_not_mutate_the_original_record(record_factory):
    rec = record_factory(signature=["int", "int"], func_def="int f(int x) {\n    x++;\n}\n")
    envnr.process_record(rec)
    assert "no_return" not in rec


def test_process_record_preserves_every_original_field(record_factory):
    rec = record_factory(signature=["void", "int"], func_def="void f(int x) {\n    x++;\n}\n")
    out = envnr.process_record(rec)
    for key, value in rec.items():
        assert out[key] == value
