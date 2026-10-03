from scripts import regen_asm


def test_strip_static_removes_static_keyword():
    out, was_static = regen_asm.strip_static("static int helper(void) {\n    return 1;\n}")
    assert was_static is True
    assert out.startswith("int helper(void)")
    assert "static" not in out


def test_strip_static_removes_inline_keyword():
    out, was_static = regen_asm.strip_static("inline int helper(void) { return 1; }")
    assert was_static is True
    assert "inline" not in out


def test_strip_static_preserves_attribute():
    out, was_static = regen_asm.strip_static('static __attribute__((unused)) int helper(void) { return 1; }')
    assert was_static is True
    assert "__attribute__((unused))" in out
    assert "static" not in out


def test_strip_static_no_specifiers_is_noop():
    src = "int helper(void) { return 1; }"
    out, was_static = regen_asm.strip_static(src)
    assert was_static is False
    assert out == src


def test_strip_static_does_not_touch_local_static_in_body():
    src = "int helper(void) {\n    static int cache;\n    return cache;\n}"
    out, was_static = regen_asm.strip_static(src)
    assert was_static is False
    assert out == src


def test_pick_deps_prefers_real_deps_when_content_bearing():
    rec = {"real_deps": "typedef int foo_t;\n# 1", "angha_deps": "typedef int bar_t;"}
    assert regen_asm.pick_deps(rec) == "typedef int foo_t;\n# 1"


def test_pick_deps_falls_back_to_angha_deps_when_real_deps_only_has_marker():
    rec = {"real_deps": "  # 1", "angha_deps": "typedef int bar_t;"}
    assert regen_asm.pick_deps(rec) == "typedef int bar_t;"


def test_pick_deps_empty_both_returns_empty_string():
    assert regen_asm.pick_deps({}) == ""
    assert regen_asm.pick_deps({"real_deps": None, "angha_deps": None}) == ""


def test_instruction_count_skips_directives_and_labels():
    asm = "func0:\n\t.cfi_startproc\n\tpush %rbp\n\tpop %rbp\n\tret\n\t.cfi_endproc\n"
    assert regen_asm.instruction_count(asm) == 3


def test_instruction_count_empty_body_is_zero():
    assert regen_asm.instruction_count("func0:\n\t.cfi_startproc\n\t.cfi_endproc\n") == 0


def test_split_asm_extracts_function_block_and_trims_pre_asm():
    full_asm = (
        "\t.text\n"
        "\t.globl\tother\n"
        "other:\n"
        "\tret\n"
        "\t.globl\ttarget\n"
        "\t.type\ttarget, @function\n"
        "target:\n"
        "\tpush\t%rbp\n"
        "\tpop\t%rbp\n"
        "\tret\n"
        "\t.cfi_endproc\n"
        "\t.size\ttarget, .-target\n"
    )
    pre, func = regen_asm.split_asm(full_asm, "target")
    assert "other:" in pre
    assert "target:" not in pre
    assert "target:" in func
    assert "push" in func
    assert func.strip().endswith(".size\ttarget, .-target")


def test_split_asm_symbol_not_found_returns_everything_as_pre():
    full_asm = "other:\n\tret\n"
    pre, func = regen_asm.split_asm(full_asm, "missing")
    assert pre == full_asm
    assert func == ""
