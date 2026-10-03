from scripts.preprocess import get_asm


def test_mod_path_swaps_extension_same_dir():
    assert get_asm.mod_path("/tmp/x/1_O0.c", "out") == "/tmp/x/1_O0.out"


def test_mod_path_parent_path_flag_returns_parent_dir():
    new_path, parent = get_asm.mod_path("/tmp/x/y/1_O0.c", "asm", parent_path_flag=True)
    assert new_path == "/tmp/x/y/1_O0.asm"
    assert str(parent) == "/tmp/x"


def test_rename_to_func0_renames_definition_and_recursive_call():
    src = "int fib(int n) {\n    if (n < 2) return n;\n    return fib(n - 1) + fib(n - 2);\n}\n"
    out = get_asm.rename_to_func0(src, "fib")
    assert "int func0(int n)" in out
    assert out.count("func0(") == 3  # definition + two recursive calls
    assert "fib" not in out


def test_rename_to_func0_does_not_rename_non_call_occurrences():
    src = "int fib_helper(int fib) {\n    return fib;\n}\n"
    out = get_asm.rename_to_func0(src, "fib")
    # Neither fib_helper nor the bare parameter `fib` is call-shaped.
    assert out == src


def test_rename_to_func0_empty_fname_is_noop():
    src = "int main(void) { return 0; }"
    assert get_asm.rename_to_func0(src, "") == src
    assert get_asm.rename_to_func0(src, None) == src


def test_create_c_file_normal_fname_renames_and_appends_driver(tmp_path, dict_db_factory):
    dict_db = dict_db_factory(
        c_func="int add_one(int x) {\n    return x + 1;\n}",
        fname="add_one",
        task_id="42",
        type="O1",
    )
    c_path = get_asm.create_c_file(dict_db, str(tmp_path))
    assert c_path == str(tmp_path / "C_files" / "42_O1.c")

    src = open(c_path).read()
    assert "int func0(int x)" in src
    assert "add_one" not in src
    assert "int main() {" in src


def test_create_c_file_main_fname_renames_and_appends_driver(tmp_path, dict_db_factory):
    dict_db = dict_db_factory(
        c_func="int main(void) {\n    return 0;\n}",
        fname="main",
        task_id="7",
        type="O0",
    )
    c_path = get_asm.create_c_file(dict_db, str(tmp_path))
    src = open(c_path).read()
    # Only the driver's main remains.
    assert src.count("int main(") == 1
    assert "int func0(void)" in src
    assert "int main() {" in src
