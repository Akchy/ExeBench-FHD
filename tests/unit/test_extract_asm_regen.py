from scripts.extract import extract_asm_regen as ear


def test_needs_asm_regen_true_for_static_function(record_factory):
    rec = record_factory(func_def="static int helper(void) {\n    return 1;\n}\n")
    assert ear.needs_asm_regen(rec) is True


def test_needs_asm_regen_true_for_inline_function(record_factory):
    rec = record_factory(func_def="inline int helper(void) {\n    return 1;\n}\n")
    assert ear.needs_asm_regen(rec) is True


def test_needs_asm_regen_false_for_plain_function(record_factory):
    rec = record_factory(func_def="int helper(void) {\n    return 1;\n}\n")
    assert ear.needs_asm_regen(rec) is False


def test_needs_asm_regen_does_not_flag_local_static_in_body(record_factory):
    rec = record_factory(func_def="int helper(void) {\n    static int cache;\n    return cache;\n}\n")
    assert ear.needs_asm_regen(rec) is False


def test_build_source_strips_static_and_wraps_noinline(record_factory):
    rec = record_factory(func_def="static int helper(void) {\n    return 1;\n}\n")
    csrc, was_static, fname = ear.build_source(rec)
    assert was_static is True
    assert fname == rec["fname"]
    assert "static int helper" not in csrc
    assert "__attribute__((noinline))" in csrc
    assert "int helper(void)" in csrc


def test_build_source_wraps_noinline_even_when_not_static(record_factory):
    rec = record_factory(func_def="int helper(void) {\n    return 1;\n}\n")
    csrc, was_static, _fname = ear.build_source(rec)
    assert was_static is False
    assert "__attribute__((noinline))" in csrc


def test_build_source_prepends_deps(record_factory):
    rec = record_factory(func_def="int helper(void) {\n    return 1;\n}\n",
                          real_deps="typedef int foo_t;\n# 1", angha_deps="")
    csrc, _was_static, _fname = ear.build_source(rec)
    assert "typedef int foo_t;" in csrc


def test_build_objdump_source_appends_noop_driver_for_non_main(record_factory):
    rec = record_factory(fname="helper", func_def="static int helper(void) {\n    return 1;\n}\n")
    src, was_static, fname = ear.build_objdump_source(rec)

    assert was_static is True
    assert fname == "helper"
    assert "__attribute__((noinline))" in src
    assert "static int helper" not in src
    assert src.rstrip().endswith("int main(void) {\n    return 0;\n}")
    # the driver must not call the target
    driver = src.split("int main(void)")[-1]
    assert "helper(" not in driver


def test_build_objdump_source_skips_driver_when_fname_is_main(record_factory):
    rec = record_factory(fname="main", func_def="int main(void) {\n    return 0;\n}\n")
    src, _was_static, fname = ear.build_objdump_source(rec)

    assert fname == "main"
    assert src.count("int main") == 1


def test_build_objdump_source_uses_shared_strip_static_logic(record_factory):
    rec = record_factory(fname="helper", func_def="int helper(void) {\n    return 1;\n}\n")
    src, was_static, _fname = ear.build_objdump_source(rec)

    assert was_static is False
    assert "__attribute__((noinline))" in src
