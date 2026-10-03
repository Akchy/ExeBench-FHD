from scripts.extract import extract_func_call as efc


def test_analyze_params_scalar_types_are_kept():
    params, skip_reason = efc.analyze_params(["int", "int", "float"])
    assert skip_reason is None
    assert params == [("int", "int_family"), ("float", "float_family")]


def test_analyze_params_void_placeholder_dropped():
    params, skip_reason = efc.analyze_params(["int", "void"])
    assert skip_reason is None
    assert params == []


def test_analyze_params_varargs_dropped_and_stops_scanning():
    params, skip_reason = efc.analyze_params(["int", "int", "...", "double"])
    assert skip_reason is None
    assert params == [("int", "int_family")]


def test_analyze_params_struct_by_value_triggers_skip():
    params, skip_reason = efc.analyze_params(["int", "struct foo"])
    assert params is None
    assert skip_reason == "struct_or_union_by_value"


def test_analyze_params_simd_type_triggers_skip():
    params, skip_reason = efc.analyze_params(["int", "__m128i"])
    assert params is None
    assert skip_reason == "simd_or_complex"


def test_analyze_params_uncategorized_type_triggers_skip():
    params, skip_reason = efc.analyze_params(["int", "FooBarUnknownType_t"])
    assert params is None
    assert skip_reason == "uncategorized"


def test_build_arg_expr_int_family_uses_strtoll():
    assert efc.build_arg_expr("int", 2) == "(int)strtoll(argv[2], NULL, 10)"


def test_build_arg_expr_float_family_uses_strtod():
    assert efc.build_arg_expr("double", 1) == "(double)strtod(argv[1], NULL)"


def test_build_arg_expr_pointer_type_casts_argv():
    assert efc.build_arg_expr("char *", 3) == "(char *)(void*)argv[3]"
    assert efc.build_arg_expr("struct foo *", 1) == "(struct foo *)(void*)argv[1]"


def test_build_driver_source_includes_call_and_marker(record_factory):
    rec = record_factory(real_deps="", angha_deps="")
    src = efc.build_driver_source(rec, "add_one", "int", [("int", "int_family")])
    assert "__func_call_marker" in src
    assert "func0(" in src
    assert "int main(int argc, char **argv)" in src
    assert "__FUNC_CALL_NOIPA\nint func0(" in src


def test_build_driver_source_noipa_prelude_falls_back_to_noinline(record_factory):
    src = efc.build_driver_source(record_factory(), "add_one", "int", [("int", "int_family")])
    prelude = src[src.index("#if defined(__has_attribute)"):src.index("static volatile")]
    assert "#  if __has_attribute(noipa)" in prelude
    assert "#    define __FUNC_CALL_NOIPA __attribute__((noipa))" in prelude
    assert "#ifndef __FUNC_CALL_NOIPA\n#  define __FUNC_CALL_NOIPA __attribute__((noinline))" in prelude
    # prelude must come before the target that uses the macro
    assert src.index("#ifndef __FUNC_CALL_NOIPA") < src.index("__FUNC_CALL_NOIPA\nint func0(")


def test_build_driver_source_void_return_has_no_result_var(record_factory):
    rec = record_factory()
    src = efc.build_driver_source(rec, "do_thing", "void", [])
    assert "__func_call_result" not in src
    assert "func0();" in src


def test_extract_call_window_slices_between_marker_and_target():
    asm = """
0000000000001100 <main>:
    1100:\te8 00 00 00 00       \tcall   1200 <__func_call_marker>
    1105:\tbf 01 00 00 00       \tmov    $0x1,%edi
    110a:\te8 00 00 00 00       \tcall   1300 <func0>
    110f:\tb8 00 00 00 00       \tmov    $0x0,%eax
    1114:\tc3                   \tret
"""
    window = efc.extract_call_window(asm)
    assert window is not None
    assert "func0" in window
    assert "__func_call_marker" not in window
    assert "ret" not in window


def test_extract_call_window_missing_marker_returns_none():
    asm = """
0000000000001100 <main>:
    1100:\te8 00 00 00 00       \tcall   1300 <func0>
"""
    assert efc.extract_call_window(asm) is None


def test_extract_call_window_tolerates_clone_suffix():
    asm = """
0000000000001100 <main>:
    1100:\te8 00 00 00 00       \tcall   1200 <__func_call_marker>
    1105:\te8 00 00 00 00       \tcall   1300 <func0.constprop.0>
"""
    window = efc.extract_call_window(asm)
    assert window is not None
    assert "func0.constprop.0" in window


def test_collapse_identical():
    assert efc.collapse_identical({"O0": "e", "O1": "e"}) == "e"
    assert efc.collapse_identical({"O0": None, "O1": None}) is None
    assert efc.collapse_identical({"O0": None, "O1": "e"}) == {"O0": None, "O1": "e"}


def test_func_call_by_arch_reads_all_shapes():
    per_level = {"x86": {"O0": "a", "O2": "b"}, "ARM64": "shared error", "ARM32": None}
    assert efc.func_call_by_arch(per_level, "O2") == {"x86": "b", "ARM64": "shared error", "ARM32": None}
    legacy = {"x86": "o3-snippet"}
    assert efc.func_call_by_arch(legacy, "O0") == legacy
    assert efc.func_call_by_arch(None) is None


def test_normalize_error_masks_volatile_parts():
    a = ("compile_failed: /usr/bin/ld: /tmp/ccMHWiEo.o: in function `func0':\n"
         "output_files/w/w12/C_files/s_3_fc.c:(.text+0x30): undefined reference to `nowhere'")
    b = a.replace("ccMHWiEo", "ccQ9xYz1").replace("+0x30", "+0x25").replace("w12", "w99")
    assert efc.normalize_error(a) == efc.normalize_error(b)
    assert "<tmp>.o" in efc.normalize_error(a) and "(.text+<off>)" in efc.normalize_error(a)
    assert "undefined reference to `nowhere'" in efc.normalize_error(a)
    asm_a = "compile_failed: /tmp/ccTq4zyZ.s: Assembler messages:\n/tmp/ccTq4zyZ.s:70: Error: bad instruction `lock'"
    asm_b = asm_a.replace("ccTq4zyZ", "ccdrhoTu").replace(":70:", ":48:")
    assert efc.normalize_error(asm_a) == efc.normalize_error(asm_b)
    assert efc.normalize_error(None) is None
