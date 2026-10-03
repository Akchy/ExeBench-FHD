from scripts.preprocess import preprocess_asm

ASM_TWO_FUNCS = """
0000000000001139 <func0>:
    1139:\t55                   \tpush   %rbp
    113a:\t48 89 e5             \tmov    %rsp,%rbp
    113d:\t89 7d fc             \tmov    %edi,-0x4(%rbp)
    1140:\t8b 45 fc             \tmov    -0x4(%rbp),%eax
    1143:\t83 c0 01             \tadd    $0x1,%eax
    1146:\t5d                   \tpop    %rbp
    1147:\tc3                   \tret

0000000000001148 <other_func>:
    1148:\t55                   \tpush   %rbp
"""


def test_convert_hex_in_line_converts_immediates_to_decimal():
    assert preprocess_asm.convert_hex_in_line("add $0x1,%eax   ; 0x10") == "add $1,%eax   ; 16"


def test_convert_hex_in_line_no_hex_is_unchanged():
    line = "push   %rbp"
    assert preprocess_asm.convert_hex_in_line(line) == line


def test_extract_function_default_mode_strips_percent_spaces_commas_and_hex():
    out = preprocess_asm.extract_function(ASM_TWO_FUNCS, mode="default", func_name="func0")
    assert out[0] == "<func0>:\n"
    assert len(out) == 8  # header + 7 instructions
    assert "%" not in "".join(out)
    assert "add    $1 , eax" in out[5]
    assert "0x1" not in out[5]


def test_extract_function_stops_before_next_function_label():
    out = preprocess_asm.extract_function(ASM_TWO_FUNCS, mode="default", func_name="func0")
    assert not any("other_func" in line for line in out)
    assert not any("1148" in line for line in out)


def test_extract_function_non_default_mode_leaves_raw_text():
    out = preprocess_asm.extract_function(ASM_TWO_FUNCS, mode="full", func_name="func0")
    assert "mov    %rsp,%rbp" in out[2]


def test_extract_function_missing_function_returns_empty_list():
    assert preprocess_asm.extract_function(ASM_TWO_FUNCS, mode="default", func_name="nope") == []


def test_extract_function_missing_function_debug_print(capsys):
    preprocess_asm.extract_function(ASM_TWO_FUNCS, mode="default", func_name="nope", debug_print=True)
    assert "No <nope> function found" in capsys.readouterr().out


def test_add_labels_to_file_skips_first_line():
    out = preprocess_asm.add_labels_to_file(["<func0>:", "push rbp", "ret"])
    assert out[0] == "<func0>:\n"
    assert out[1] == "push rbp <label-1>\n"
    assert out[2] == "ret <label-2>\n"


def test_process_jump_inst_rewrites_known_target_to_label():
    # The jump is rewritten to the target line's own <label-N>, not its symbol.
    lines = [
        "1000: 90       nop <label-1>",
        "1004: eb 05    jmp 100a <other> <label-2>",
        "100a: 90       nop <target> <label-3>",
    ]
    out = preprocess_asm.process_jump_inst(lines)
    assert "<label-3>" in out[1]
    assert out[0] == lines[0]
    assert out[2] == lines[2]


def test_process_jump_inst_unknown_target_keeps_original_line():
    lines = ["1004: eb 05    jmp ffff <other> <label-1>"]
    out = preprocess_asm.process_jump_inst(lines)
    assert out == lines


def test_process_jump_inst_no_jumps_returns_input_unchanged():
    lines = ["1000: 90 nop <label-1>"]
    assert preprocess_asm.process_jump_inst(lines) == lines


def test_clean_init_address_strips_address_and_bytes_prefix():
    lines = ["<func0>:", "    1139:\t55  \tpush   rbp"]
    out = preprocess_asm.clean_init_address(lines)
    assert out.splitlines()[0] == "<func0>:"
    assert out.splitlines()[1] == "push   rbp"


def test_clean_init_address_empty_input_returns_joined_original():
    assert preprocess_asm.clean_init_address([]) == ""


def test_processing_asm_default_mode_full_pipeline():
    out = preprocess_asm.processing_asm(ASM_TWO_FUNCS, "func0", "default")
    lines = out.splitlines()
    assert lines[0] == "<func0>:"
    assert lines[5] == "add    $1 , eax <label-5>"
    assert "other_func" not in out


def test_processing_asm_no_process_w_addr_mode_keeps_raw_addresses():
    out = preprocess_asm.processing_asm(ASM_TWO_FUNCS, "func0", "no_process_w_addr")
    assert "1139:" in out
    assert "%rbp" in out


def test_processing_asm_asm_only_strips_labels_but_not_default_processing():
    out = preprocess_asm.processing_asm(ASM_TWO_FUNCS, "func0", "default", asm_only=True)
    assert "<label-" not in out
    assert out.splitlines()[-1] == "ret"


def test_processing_asm_function_not_found_returns_empty_string():
    assert preprocess_asm.processing_asm(ASM_TWO_FUNCS, "does_not_exist", "default") == ""


def test_strip_asm_comment_x86_comment_removed():
    line = "    1154:\t48 8b 05 b5 2e 00 00 \tmov    0x2eb5(%rip),%rax        # 4010 <sink>"
    assert preprocess_asm.strip_asm_comment(line) == "    1154:\t48 8b 05 b5 2e 00 00 \tmov    0x2eb5(%rip),%rax"


def test_strip_asm_comment_keeps_arm_immediates():
    for line in (
        " 69c:\tf9400660 \tldr\tx0, [x19, #8]",
        " 414:\t220a      \tmovs\tr2, #10",
        " 718:\td10083ff \tsub\tsp, sp, #0x20",
        " 6a0:\ta9bf7bfd \tstp\tx29, x30, [sp, #-16]!",
    ):
        assert preprocess_asm.strip_asm_comment(line) == line


def test_strip_asm_comment_arm32_at_comment_removed_immediate_kept():
    line = " 474:\t4b0b      \tldr\tr3, [pc, #44]\t@ (4a4 <main+0x60>)"
    assert preprocess_asm.strip_asm_comment(line) == " 474:\t4b0b      \tldr\tr3, [pc, #44]"


def test_strip_asm_comment_arm64_slash_comment_removed_immediate_kept():
    line = " 698:\td2800001 \tmov\tx1, #0x0                   \t// #0"
    assert preprocess_asm.strip_asm_comment(line) == " 698:\td2800001 \tmov\tx1, #0x0"


def test_strip_asm_comment_keeps_plt_symbol():
    line = " 6a0:\t97ffffdc \tbl\t610 <strtoll@plt>"
    assert preprocess_asm.strip_asm_comment(line) == line


ASM_ARM64 = """
0000000000000880 <func0>:
 880:\td10043ff \tsub\tsp, sp, #0x10
 884:\tb9000fe0 \tstr\tw0, [sp, #12]
 888:\td2800000 \tmov\tx0, #0x0                   \t// #0
 88c:\t910043ff \tadd\tsp, sp, #0x10
 890:\td65f03c0 \tret
"""


def test_extract_function_arm64_keeps_immediates_and_no_blank_lines():
    out = preprocess_asm.extract_function(ASM_ARM64, mode="no_process_w_addr", func_name="func0")
    assert out[1] == " 880:\td10043ff \tsub\tsp, sp, #0x10"
    assert out[2] == " 884:\tb9000fe0 \tstr\tw0, [sp, #12]"
    assert out[3] == " 888:\td2800000 \tmov\tx0, #0x0"
    assert not any("\n" in line for line in out[1:])
