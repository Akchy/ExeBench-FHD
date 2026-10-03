"""Small C-source and gcc-flag helpers shared by the preprocessing scripts."""
import re
import os
import json
from pathlib import Path
from collections import Counter, defaultdict


def count_logs(source_file):
    """Tally `--- Correction Attempt: <id>` lines in correction_prompt.txt next to `source_file`.

    Writes per-id counts to counts.log and a 1..6 histogram to histogram.log
    in the same directory. Does nothing if the log is missing.
    """
    parent_dir = Path(source_file).parent
    log_file_path = f"{parent_dir}/correction_prompt.txt"
    if not Path(log_file_path).is_file():
        return

    pattern = re.compile(r"--- Correction Attempt:\s*([A-Za-z0-9_]+)")
    counts = Counter()
    with open(log_file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                counts[match.group(1)] += 1

    histogram = defaultdict(int)
    for count in counts.values():
        if 1 <= count <= 6:
            histogram[count] += 1

    with open(f"{parent_dir}/counts.log", "w+") as outfile:
        for file_id, count in counts.items():
            outfile.write(f"{file_id}: {count}\n")

    with open(f"{parent_dir}/histogram.log", "w+") as outfile:
        for i in range(1, 7):
            outfile.write(f"Files with count {i}: {histogram[i]}\n")


def c_files_to_json(folder_path, output_json):
    """Dump every .c file in `folder_path` to a JSON list of {file_name, code}."""
    data_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".c"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                data_list.append({'file_name': filename, 'code': f.read()})

    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(data_list, json_file, indent=4)

    print(f"Saved C files to {output_json}")


def extract_c_function(c_code, function_name):
    """Return (body, len(body)) for `function_name`, minus blank and // lines, or (False, 0)."""
    header_pattern = rf'\b(\w+(?:\s*\*)*)\s+{re.escape(function_name)}\s*\([^)]*\)\s*\{{'
    match = re.search(header_pattern, c_code, re.MULTILINE)
    if not match:
        return False, 0

    func_start = match.start()
    brace_count = 1
    pos = match.end()
    while pos < len(c_code) and brace_count > 0:
        if c_code[pos] == '{':
            brace_count += 1
        elif c_code[pos] == '}':
            brace_count -= 1
        pos += 1

    if brace_count == 0:
        lines = c_code[func_start:pos].split('\n')
        meaningful = '\n'.join(
            line for line in lines
            if line.strip() != "" and not line.strip().startswith("//")
        )
        return meaningful, len(meaningful)

    return False, 0


def extract_function_headers(code):
    """Return the first `ret name(args)` header defined in `code`, or False."""
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//.*', '', code)

    pattern = re.compile(
        r"""
        ([\w\s\*\(\)]+?)            # return type
        \s*
        (\w+)                       # function name
        \s*
        \(([^)]*)\)                 # argument list
        \s*
        \{
        """,
        re.VERBOSE
    )

    match = pattern.search(code)
    if not match:
        return False
    return f"{match.group(1).strip()} {match.group(2).strip()}({match.group(3).strip()})"


def parse_header_for_datatypes(header):
    """Parse a C prototype into {"return_type", "arg_types"}, or None if it doesn't look like one."""
    header = " ".join(header.strip().split())
    match = re.match(r'^\s*([\w\*\s]+?)\s+(\w+)\s*\((.*?)\)\s*;?$', header)
    if not match:
        return None

    return_type = match.group(1).strip()
    args = match.group(3).strip()

    if args in ("void", ""):
        arg_types = []
    else:
        # Drop the trailing parameter name: "const char *msg" -> "const char *".
        arg_types = [re.sub(r'\s*\b\w+\s*$', '', p.strip()) for p in args.split(",")]

    return {"return_type": return_type, "arg_types": arg_types}


_GCC_OPTS_PATH = 'datasets/gcc_opts.json'
_GCC_OPTS_CACHE = {}


def _load_gcc_opts(path=_GCC_OPTS_PATH):
    """Map opt_level -> list of -f flags from gcc_opts.json, read once per path."""
    if path not in _GCC_OPTS_CACHE:
        with open(path, 'r') as f:
            _GCC_OPTS_CACHE[path] = {
                entry['opt_level']: entry['flags'] for entry in json.load(f)
            }
    return _GCC_OPTS_CACHE[path]


def get_all_optz(opt_level, ind_flag):
    """Return `-fno-*` for every flag of `opt_level` except `ind_flag`.

    `ind_flag='none'` negates all of them. Returns False for an unknown level
    or a flag that isn't part of that level.
    """
    opts_list = _load_gcc_opts().get(opt_level)
    if opts_list is None:
        return False

    if ind_flag in opts_list:
        return ['-fno-' + opt[2:] for opt in opts_list if opt != ind_flag]
    if ind_flag == 'none':
        return ['-fno-' + opt[2:] for opt in opts_list]
    return False
