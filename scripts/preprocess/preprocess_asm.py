"""Turn `objdump -d` output into the normalized per-function asm text used by the dataset."""
import re


def convert_hex_in_line(line) -> str:
    return re.sub(r'0x[0-9A-Fa-f]+', lambda m: str(int(m.group(), 16)), line)


# objdump's trailing comment: `# ...` (x86), `@ ...` (ARM32), `// ...` (ARM64).
# Whitespace is required on both sides because on ARM `#` also prefixes
# immediates (`ldr x0, [x19, #8]`), which a bare split("#") would cut.
_ASM_COMMENT_RE = re.compile(r'\s+(?:#|@|//)\s.*$')


def strip_asm_comment(line) -> str:
    return _ASM_COMMENT_RE.sub('', line)


def extract_function(asm, mode='full', func_name="func0", debug_print=False) -> list:
    """Return the `<func_name>:` header plus its instruction lines from objdump output.

    In 'default' mode lines are normalized (no `%`, spaced punctuation, hex
    to decimal); any other mode keeps them as-is. Returns [] if not found.
    """
    inside_func = False
    func_code = []

    # e.g. "0000000000001139 <func0>:"
    func_header = re.compile(r'^[0-9a-fA-F]+\s+<([^>]+)>:$')
    instruction_pattern = re.compile(r'^\s*([0-9a-fA-F]+:)\s*(.*)$')

    for line in asm.splitlines():
        line = strip_asm_comment(line)
        match = func_header.match(line.strip())

        if match:
            if match.group(1) == func_name:
                inside_func = True
                func_code.append(f"<{func_name}>:\n")
                continue
            elif inside_func:
                break

        if inside_func and instruction_pattern.match(line):
            if mode == 'default':
                line = line.replace("%", "")
                line = re.sub(r'([,()])', r' \1 ', line)
                func_code.append(convert_hex_in_line(line))
            else:
                func_code.append(line)

    if not func_code and debug_print:
        print(f"[-] No <{func_name}> function found.")
    return func_code


def add_labels_to_file(lines: list) -> list:
    """Append ` <label-i>` to every line after the header."""
    new_lines = [lines[0].rstrip() + '\n'] if lines else []
    for i, line in enumerate(lines[1:], start=1):
        new_lines.append(f"{line.rstrip()} <label-{i}>\n")
    return new_lines


def process_jump_inst(assembly_lines: list, debug_print=False) -> list:
    """Rewrite jump targets from raw addresses to the `<label-i>` of the target line."""
    address_to_label = {}
    for line in assembly_lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(r'^([0-9A-Fa-f]+):?\s+.*<([^>]+)>\s*$', line)
        if match:
            address_to_label[match.group(1).upper()] = match.group(2)

    jump_re = re.compile(
        r'^(\s*?[0-9A-Fa-f]+):\s+([0-9A-Fa-f\s]+)\s+(j\w*)\s+([0-9A-Fa-f]+)\s+(<[^>]+>)\s+(<[^>]+>)\s*$'
    )
    processed_lines = []
    for line in assembly_lines:
        m = jump_re.match(line)
        target = m.group(4).upper() if m else None
        if target in address_to_label:
            line_address, raw_bytes, jump_instruction = m.group(1, 2, 3)
            processed_lines.append(
                f"{line_address}: {raw_bytes} \t{jump_instruction} <{address_to_label[target]}> {m.group(6)}"
            )
        else:
            processed_lines.append(line)

    if processed_lines:
        return processed_lines
    if debug_print:
        print("[-] No jump instructions found.")
    return assembly_lines


def clean_init_address(assembly_lines: list, debug_print=False) -> str:
    """Drop the leading `addr: hex bytes` columns, keeping only the instruction text."""
    processed_lines = []
    for i, line in enumerate(assembly_lines):
        if i == 0:
            processed_lines.append(line.rstrip())
            continue

        original_line = line
        line = line.strip()
        if not line:
            processed_lines.append(original_line)
            continue

        init_addr_match = re.match(r'^\s*([0-9a-fA-F]+:)\s*([0-9a-fA-F\s]*[0-9a-fA-F])\s*\t(.*)', line)
        if init_addr_match:
            processed_lines.append(init_addr_match.group(3))
        else:
            matching_zeros = re.match(r'^\s*([0-9a-fA-F]+:)\s*([00\s]+)(.*)', line)
            if matching_zeros:
                processed_lines.append(matching_zeros.group(3))

    if processed_lines:
        return '\n'.join(processed_lines)
    if debug_print:
        print("[-] No lines to process.")
    return '\n'.join(assembly_lines)


def processing_asm(asm, func_name, mode, asm_only=False, debug_print=False) -> str:
    """Extract `func_name` from objdump output and normalize it; '' if not found."""
    lines = extract_function(asm, mode, func_name, debug_print)

    if mode == 'no_process_w_addr':
        return '\n'.join(lines)
    elif asm_only:
        return clean_init_address(lines)

    if not lines:
        return ''
    labeled = add_labels_to_file(lines)
    return clean_init_address(process_jump_inst(labeled, debug_print), debug_print)
