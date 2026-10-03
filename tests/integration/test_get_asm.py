import pytest

from scripts.preprocess import get_asm
from tests.conftest import requires_gcc, requires_objdump

pytestmark = [pytest.mark.slow, requires_gcc, requires_objdump]


def test_get_processed_asm_normal_function_end_to_end(tmp_path, dict_db_factory):
    dict_db = dict_db_factory(
        c_func="int add_one(int x) {\n    return x + 1;\n}",
        fname="add_one",
        task_id="1",
        type="O0",
    )
    norm_asm = get_asm.get_processed_asm(dict_db, str(tmp_path))
    assert norm_asm
    assert "<func0>:" in norm_asm


def test_get_processed_asm_main_named_function_end_to_end(tmp_path, dict_db_factory):
    dict_db = dict_db_factory(
        c_func="int main(void) {\n    return 42;\n}",
        fname="main",
        task_id="2",
        type="O0",
    )
    norm_asm = get_asm.get_processed_asm(dict_db, str(tmp_path))
    assert norm_asm
    assert "<main>:" in norm_asm


def test_get_processed_asm_compile_failure_returns_false(tmp_path, dict_db_factory):
    dict_db = dict_db_factory(
        c_func="int broken( {{{ not valid C at all",
        fname="broken",
        task_id="3",
        type="O0",
    )
    assert get_asm.get_processed_asm(dict_db, str(tmp_path), cleanup=False) is False
    # The error log goes to the parent of C_files/.
    assert (tmp_path / "gcc_errors.log").exists()


def test_get_processed_asm_cleanup_removes_generated_files(tmp_path, dict_db_factory):
    dict_db = dict_db_factory(
        c_func="int add_one(int x) {\n    return x + 1;\n}",
        fname="add_one",
        task_id="4",
        type="O0",
    )
    get_asm.get_processed_asm(dict_db, str(tmp_path), cleanup=True)
    c_folder = tmp_path / "C_files"
    assert not (c_folder / "4_O0.c").exists()
    assert not (c_folder / "4_O0.out").exists()


def test_compile_c_program_and_disassemble_out_file_directly(tmp_path):
    c_path = tmp_path / "t.c"
    c_path.write_text("int func0(void) { return 5; }\nint main(void) { return 0; }\n")

    assert get_asm.compile_c_program(str(c_path), "O0") is True

    asm = get_asm.disassemble_out_file(str(c_path))
    assert isinstance(asm, str)
    assert "<func0>:" in asm


def test_compile_c_program_invalid_source_returns_false(tmp_path):
    c_path = tmp_path / "bad.c"
    c_path.write_text("this is not C at all {{{")
    assert get_asm.compile_c_program(str(c_path), "O0") is False


def test_disassemble_out_file_missing_binary_returns_false(tmp_path):
    c_path = tmp_path / "nope.c"
    assert get_asm.disassemble_out_file(str(c_path)) is False
