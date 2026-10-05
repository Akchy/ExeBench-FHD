#!/usr/bin/env python3
"""Extract the `data_section` field: the bytes of every data-like section
(.data, .rodata, .bss, .tdata, ...) the target function's object contributes,
per architecture (x86, x86-64, ARM32, ARM64).

We link a full executable and read values from it rather than from .s/.o,
because relocated pointers (e.g. `char *p = "lit"`) only hold their real
value after linking. The GNU ld link map tells us which byte ranges came from
our object; `objdump -s` supplies the bytes.

Usage:
    python3 -m scripts.extract.extract_data_sections \
        --split train_real_compilable --n 100 \
        --out output_files/data_section_sample.jsonl
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

from scripts.check_split_leakage import ALL_SPLITS, iter_split
from scripts.extract.extract_func_call import run_tool
from scripts.preprocess.get_asm import rename_to_func0
from scripts.regen_asm import pick_deps, strip_static

GCC_TIMEOUT = 60
OPT_FLAG = "-O0"

ARCH_TOOLCHAIN = {
    "x86":     ["gcc", "-m32"],
    "x86-64":  ["gcc", "-m64"],
    "ARM32":   ["arm-linux-gnueabihf-gcc"],
    "ARM64":   ["aarch64-linux-gnu-gcc"],
}

_OBJDUMP_BIN = {
    "x86": "objdump",
    "x86-64": "objdump",
    "ARM32": "arm-linux-gnueabihf-objdump",
    "ARM64": "aarch64-linux-gnu-objdump",
}

# Sections that hold source-level values (no .text, .got/.plt, .init_array,
# debug or note sections). Matched against the linker's output section name.
_DATA_SECTION_NAME_RE = re.compile(r"^\.(data|rodata|bss|tdata|tbss)(\.|$)")

# Column-0 line of a GNU ld -Map file: output section, address, size.
_MAP_TOP_RE = re.compile(r"^(\.\S+)\s+(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)\s*$")

# Indented input-section line: [name] address size object. GNU ld puts a long
# name (e.g. .data.rel.local) on its own line, so the name is optional here.
_MAP_MEMBER_RE = re.compile(r"^\s+(?:(\.\S+)\s+)?(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)\s+(\S+)\s*$")
_MAP_WRAPPED_NAME_RE = re.compile(r"^\s+(\.\S+)\s*$")

_NOBITS_RE = re.compile(r"^\.t?bss(\.|$)")

# Mergeable sections (.rodata.cstN, .rodata.strW.A, used from -O1 up). The
# linker deduplicates them, so the map gives their pre-merge address; we take
# the bytes from the object file and find them in the output by content.
_MERGE_SECTION_RE = re.compile(r"^\.rodata\.(?:cst(\d+)|str(\d+)\.\d+)$")

_OBJDUMP_HDR_RE = re.compile(r"^Contents of section (\S+):")
_OBJDUMP_ROW_RE = re.compile(r"^\s*([0-9a-fA-F]+)\s+(.*)$")


def data_section_by_arch(field, level="O0"):
    """`data_section` or `data_section_errors` as {arch: value} at one level.

    Handles both [arch][level] rows and single -O0 [arch] rows (returned as-is
    for any level). None stays None.
    """
    if field is None:
        return None
    return {arch: (v.get(level) if isinstance(v, dict) else v) for arch, v in field.items()}


def build_source(rec):
    """deps + func_def with static/inline stripped."""
    deps = pick_deps(rec)
    func_def, _was_static = strip_static(rec.get("func_def") or "")
    return deps + "\n" + func_def + "\n"


def build_driver_c_file(rec, path):
    """Write the source plus an empty main() to `path`, as get_asm.create_c_file
    does. A target named main is left as-is since it is already the entry point."""
    fname = rec.get("fname") or ""
    src = build_source(rec)
    if fname == "main":
        text = src.strip() + "\n"
    else:
        text = rename_to_func0(src, fname).strip() + "\nint main(void) { return 0; }\n"
    with open(path, "w") as fh:
        fh.write(text)


def compile_and_link(c_path, obj_path, out_path, map_path, compiler_argv,
                      opt_flag=OPT_FLAG, timeout=GCC_TIMEOUT):
    """Compile to an object, then link with a -Map file. Returns None on
    success, else an error string. Warnings are silenced (-w); ExeBench code
    is full of implicit declarations."""
    cmd_c = [*compiler_argv, "-c", opt_flag, "-w", "-o", obj_path, c_path]
    try:
        p = run_tool(cmd_c, timeout)
    except subprocess.TimeoutExpired:
        return f"compile timeout after {timeout}s"
    except OSError as e:
        return f"toolchain not found: {e!r}"
    if p.returncode != 0:
        return "compile_failed: " + (p.stderr or "").strip()[:2000]

    cmd_l = [*compiler_argv, "-o", out_path, f"-Wl,-Map={map_path}", obj_path, "-lm"]
    try:
        p = run_tool(cmd_l, timeout)
    except subprocess.TimeoutExpired:
        return f"link timeout after {timeout}s"
    if p.returncode != 0:
        return "link_failed: " + (p.stderr or "").strip()[:2000]
    return None


def parse_map_members(map_text):
    """Input-section lines of data-like output sections in a GNU ld map, as
    dicts {out_section, in_section, addr, size, obj} in map order."""
    members = []
    current_section = None
    pending_name = None
    for line in map_text.splitlines():
        if not line.strip():
            continue
        m = _MAP_TOP_RE.match(line)
        if m:
            current_section, pending_name = m.group(1), None
            continue
        if not (line[:1].isspace() and current_section and _DATA_SECTION_NAME_RE.match(current_section)):
            continue
        m = _MAP_MEMBER_RE.match(line)
        if m:
            name, addr_s, size_s, obj = m.groups()
            members.append({"out_section": current_section, "in_section": name or pending_name,
                            "addr": int(addr_s, 16), "size": int(size_s, 16), "obj": obj})
            pending_name = None
            continue
        m = _MAP_WRAPPED_NAME_RE.match(line)
        pending_name = m.group(1) if m else None
    return members


def parse_map_contributions(map_text, obj_path):
    """Yield (output_section, addr, size) for each non-empty contribution from
    `obj_path` (exact match against the path given to the linker)."""
    for m in parse_map_members(map_text):
        if m["obj"] == obj_path and m["size"]:
            yield m["out_section"], m["addr"], m["size"]


def parse_objdump_sections(objdump_text):
    """Parse `objdump -s` output into {section_name: (base_addr, bytes)}.
    NOBITS sections (.bss/.tbss) have no contents block and don't appear."""
    sections = {}
    name = None
    base = None
    buf = bytearray()

    def flush():
        if name is not None:
            sections[name] = (base if base is not None else 0, bytes(buf))

    for line in objdump_text.splitlines():
        m = _OBJDUMP_HDR_RE.match(line)
        if m:
            flush()
            name, base, buf = m.group(1), None, bytearray()
            continue
        if name is None:
            continue
        m = _OBJDUMP_ROW_RE.match(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        hexpart = re.split(r"\s{2,}", m.group(2), maxsplit=1)[0].replace(" ", "")
        try:
            row_bytes = bytes.fromhex(hexpart)
        except ValueError:
            continue
        if base is None:
            base = addr
        gap = addr - (base + len(buf))
        if gap > 0:
            buf += b"\x00" * gap
        buf += row_bytes
    flush()
    return sections


def hexdump(data, width=16):
    lines = []
    for i in range(0, len(data), width):
        chunk = data[i:i + width]
        hexs = " ".join(f"{b:02x}" for b in chunk)
        ascii_ = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{i:04x}  {hexs:<{width * 3 - 1}}  |{ascii_}|")
    return "\n".join(lines)


def format_contribution(section, addr, size, data):
    if data is None:
        return f"{section} @0x{addr:x} ({size} bytes, zero-initialized)"
    return f"{section} @0x{addr:x} ({size} bytes):\n{hexdump(data)}"


def _merge_pieces(in_section, content):
    """Split a mergeable section into the units the linker merges (N-byte
    constants, or NUL-terminated strings). Returns (pieces, alignment)."""
    m = _MERGE_SECTION_RE.match(in_section)
    if m.group(1):
        n = int(m.group(1))
        return [content[i:i + n] for i in range(0, len(content), n)], n
    w = int(m.group(2))
    pieces, start = [], 0
    for i in range(0, len(content) - w + 1, w):
        if content[i:i + w] == b"\0" * w:
            pieces.append(content[start:i + w])
            start = i + w
    if start < len(content):
        pieces.append(content[start:])
    return pieces, w


def _locate_merged(pieces, align, out_base, out_bytes, search_from):
    """[(addr, piece)] with each piece's address in the output section: the
    first aligned match at or after `search_from`, else anywhere, else None.
    Picking the wrong identical copy can't misreport a value."""
    found = []
    for piece in pieces:
        addr = None
        for lo in (max(search_from - out_base, 0), 0):
            off = out_bytes.find(piece, lo)
            while off != -1 and (out_base + off) % align:
                off = out_bytes.find(piece, off + 1)
            if off != -1:
                addr = out_base + off
                break
        found.append((addr, piece))
    return found


def _render_merged(out_section, in_section, located):
    """Render located pieces like format_contribution, merging contiguous
    pieces. Pieces not found get their own block with the object-file bytes."""
    blocks = []
    run_addr, run = None, b""
    for addr, piece in sorted({(a, p) for a, p in located if a is not None}):
        if run and addr == run_addr + len(run):
            run += piece
        elif run and addr < run_addr + len(run):
            continue  # tail-merged into a piece already shown
        else:
            if run:
                blocks.append(format_contribution(out_section, run_addr, len(run), run))
            run_addr, run = addr, piece
    if run:
        blocks.append(format_contribution(out_section, run_addr, len(run), run))
    for addr, piece in located:
        if addr is None:
            blocks.append(f"{out_section} ({in_section}, {len(piece)} bytes, not located in linked output):\n"
                          f"{hexdump(piece)}")
    return blocks


def _objdump_sections(path, objdump_bin, timeout, sections):
    """`objdump -s` limited to `sections`; dumping .text can be enormous."""
    cmd = [objdump_bin, "-s", *(a for s in sections for a in ("-j", s)), path]
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or "").strip()[:2000])
    return parse_objdump_sections(p.stdout)


def extract_data_from_binary(map_text, obj_path, out_path, timeout=GCC_TIMEOUT, objdump_bin="objdump"):
    """Return (rendered, error); error is set only if objdump fails, and no
    data contributions gives ("", None)."""
    members = parse_map_members(map_text)
    contributions = [m for m in members if m["obj"] == obj_path and m["size"]]
    if not contributions:
        return "", None
    try:
        # NOBITS sections are left out: their absence from the dump is how
        # they are recognized below as zero-initialized.
        wanted = sorted({m["out_section"] for m in contributions
                         if not _NOBITS_RE.match(m["out_section"])})
        sections = _objdump_sections(out_path, objdump_bin, timeout, wanted) if wanted else {}
        merge_names = sorted({m["in_section"] for m in contributions
                              if _MERGE_SECTION_RE.match(m["in_section"] or "")})
        obj_sections = _objdump_sections(obj_path, objdump_bin, timeout, merge_names) if merge_names else {}
    except subprocess.TimeoutExpired:
        return "", f"objdump timeout after {timeout}s"
    except (RuntimeError, OSError) as e:
        return "", f"objdump_failed: {e}"

    blocks = []
    for m in contributions:
        section, addr, size, in_section = m["out_section"], m["addr"], m["size"], m["in_section"] or ""
        base_and_data = sections.get(section)
        if _MERGE_SECTION_RE.match(in_section) and base_and_data and in_section in obj_sections:
            base, section_bytes = base_and_data
            pieces, align = _merge_pieces(in_section, obj_sections[in_section][1][:size])
            search_from = min(x["addr"] for x in members if x["in_section"] == in_section)
            blocks.extend(_render_merged(section, in_section,
                                         _locate_merged(pieces, align, base, section_bytes, search_from)))
            continue
        if base_and_data is None:
            data = None  # NOBITS: .bss/.tbss
        else:
            base, section_bytes = base_and_data
            off = addr - base
            if not (0 <= off and off + size <= len(section_bytes)):
                blocks.append(f"{section} @0x{addr:x} ({size} bytes, not located in linked output)")
                continue
            data = section_bytes[off:off + size]
        blocks.append(format_contribution(section, addr, size, data))
    return "\n\n".join(blocks), None


def process_record(rec, archs=None, opt_flag=OPT_FLAG, timeout=GCC_TIMEOUT):
    """Return (data_section, data_section_errors), both keyed by arch.

    data_section[arch] is "" on failure or when there is no data; the error
    is None on success.
    """
    archs = archs or list(ARCH_TOOLCHAIN)

    data_section = {}
    data_section_errors = {}
    for arch in archs:
        with tempfile.TemporaryDirectory() as td:
            c_path = os.path.join(td, "f.c")
            obj_path = os.path.join(td, "f.o")
            out_path = os.path.join(td, "f.out")
            map_path = os.path.join(td, "f.map")

            build_driver_c_file(rec, c_path)
            err = compile_and_link(c_path, obj_path, out_path, map_path,
                                    ARCH_TOOLCHAIN[arch], opt_flag, timeout)
            if err:
                data_section[arch] = ""
                data_section_errors[arch] = err
                continue

            try:
                with open(map_path, "r", errors="replace") as fh:
                    map_text = fh.read()
            except OSError as e:
                data_section[arch] = ""
                data_section_errors[arch] = f"map_read_failed: {e!r}"
                continue

            rendered, err = extract_data_from_binary(map_text, obj_path, out_path, timeout,
                                                     objdump_bin=_OBJDUMP_BIN[arch])
            data_section[arch] = rendered
            data_section_errors[arch] = err
    return data_section, data_section_errors


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train_real_compilable", choices=ALL_SPLITS)
    ap.add_argument("--n", type=int, default=100, help="number of records to process")
    ap.add_argument("--out", default="output_files/data_section_sample.jsonl")
    ap.add_argument("--archs", default=",".join(ARCH_TOOLCHAIN),
                     help="comma-separated subset of {}".format(list(ARCH_TOOLCHAIN)))
    ap.add_argument("--opt", default=OPT_FLAG, help="optimization flag, e.g. -O0/-O2")
    ap.add_argument("--limit", type=int, default=None, help="cap records scanned (debugging)")
    ap.add_argument("--summary-out", default="output_files/data_section_survey.json")
    args = ap.parse_args()

    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    unknown = [a for a in archs if a not in ARCH_TOOLCHAIN]
    if unknown:
        ap.error(f"unknown arch(es) {unknown}, choose from {list(ARCH_TOOLCHAIN)}")

    scanned = 0
    written = 0
    nonempty = {a: 0 for a in archs}
    compile_failed = {a: 0 for a in archs}

    with open(args.out, "w") as out:
        for rec in iter_split(args.split, limit=args.limit):
            scanned += 1
            data_section, data_section_errors = process_record(
                rec, archs=archs, opt_flag=args.opt)

            for a in archs:
                if data_section[a]:
                    nonempty[a] += 1
                if data_section_errors[a] is not None:
                    compile_failed[a] += 1

            rec_out = dict(rec)
            rec_out["data_section"] = data_section
            rec_out["data_section_errors"] = data_section_errors
            out.write(json.dumps(rec_out) + "\n")
            written += 1

            if written % 50 == 0:
                print(f"  {written} records processed", file=sys.stderr, flush=True)
            if written >= args.n:
                break

    summary = {
        "split": args.split,
        "archs": archs,
        "opt": args.opt,
        "out": args.out,
        "scanned": scanned,
        "written": written,
        "nonempty": nonempty,
        "nonempty_pct": {a: round(100 * nonempty[a] / written, 2) if written else 0.0 for a in archs},
        "compile_failed": compile_failed,
    }
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
