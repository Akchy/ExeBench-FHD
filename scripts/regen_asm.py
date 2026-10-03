#!/usr/bin/env python3
"""Regenerate ExeBench assembly at O0-O3 with `static`/`inline` stripped.

A standalone `static` function has no caller, so GCC drops it at -O1 and up and
the shipped `func_asm` holds no instructions. Stripping the specifier keeps the
symbol alive. Writes one JSON line per record with an `asm_regen` field:
{level: {"pre_asm", "func_asm", "n_instr"}} or {level: {"error": stderr}}.

    python3 scripts/regen_asm.py test_real out.jsonl --limit 5
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

from scripts.check_split_leakage import iter_split

LEVELS = ["O0", "O1", "O2", "O3"]

# Assembler directive or label, i.e. not an instruction.
_DIRECTIVE_RE = re.compile(r"^\s*(?:\.|#|//|\w+:)")

# Anchored at the start so a `static` local inside the body is left alone.
_LEADING_SPECIFIERS_RE = re.compile(
    r"\A\s*(?:(?:static|inline|__inline|__inline__|__attribute__\s*\(\([^)]*\)\))\s+)+"
)
_STATIC_OR_INLINE_RE = re.compile(r"\b(?:static|inline|__inline|__inline__)\b")


def strip_static(func_def):
    """Drop `static`/`inline` from the definition's leading specifiers.

    Returns (new_func_def, was_static). `__attribute__((...))` in that run is kept.
    """
    m = _LEADING_SPECIFIERS_RE.match(func_def)
    if not m:
        return func_def, False
    head = m.group(0)
    if not _STATIC_OR_INLINE_RE.search(head):
        return func_def, False
    new_head = _STATIC_OR_INLINE_RE.sub(" ", head)
    new_head = re.sub(r"[ \t]{2,}", " ", new_head).lstrip()
    return new_head + func_def[m.end():], True


# Every non-null `real_deps` ends in a bare `# 1` line marker, so an otherwise
# empty `real_deps` is still a non-empty string.
_DANGLING_LINE_MARKER_RE = re.compile(r"#\s*1\s*\Z")


def pick_deps(rec):
    """Return `real_deps` if it has real content, else `angha_deps`, else "".

    Never both: angha_deps' synthetic stdlib prototypes conflict with the real
    header declarations in real_deps.
    """
    real_deps = rec.get("real_deps") or ""
    if _DANGLING_LINE_MARKER_RE.sub("", real_deps).strip():
        return real_deps
    return rec.get("angha_deps") or ""


def instruction_count(asm):
    return sum(
        1 for line in asm.split("\n")
        if line.strip() and not _DIRECTIVE_RE.match(line.strip())
    )


def split_asm(full_asm, fname):
    """Split a .s file into (pre_asm, func_asm) the way ExeBench stores it.

    Returns (full_asm, "") when `fname`'s label isn't found.
    """
    lines = full_asm.split("\n")
    start = None
    label = fname + ":"
    for i, line in enumerate(lines):
        if line.strip() == label or re.match(r"^\s*" + re.escape(fname) + r"\s*:", line):
            start = i
            break
    if start is None:
        return full_asm, ""

    # Walk back over the directives that belong to this symbol (.globl/.type/.align).
    head = start
    while head > 0:
        prev = lines[head - 1].strip()
        if re.match(r"\.(globl|type|align|p2align|weak|hidden|size|section|text)\b", prev) \
                and fname in prev or re.match(r"\.(align|p2align)\b", prev):
            head -= 1
        else:
            break

    end = len(lines)
    for j in range(start, len(lines)):
        s = lines[j].strip()
        if s.startswith(".size") and fname in s:
            end = j + 1
            break
        if s.startswith(".cfi_endproc"):
            end = j + 1
            # keep a trailing .size if it directly follows
            if end < len(lines) and lines[end].strip().startswith(".size"):
                end += 1
            break
    return "\n".join(lines[:head]), "\n".join(lines[head:end])


def compile_one(csrc, level, fname, timeout=60, extra_flags=()):
    """Run gcc -S at `level`; returns {"pre_asm", "func_asm", "n_instr"} or {"error"}."""
    with tempfile.TemporaryDirectory() as td:
        cpath = os.path.join(td, "f.c")
        with open(cpath, "w") as fh:
            fh.write(csrc)
        # -w: ExeBench sources are full of implicit-declaration warnings.
        cmd = ["gcc", "-S", "-o", "-", "-" + level, "-w", *extra_flags, cpath]
        try:
            # String literals in the source may not be valid UTF-8 and GCC echoes
            # them into the .s output, so decode leniently.
            p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"error": f"timeout after {timeout}s"}
        if p.returncode != 0:
            return {"error": (p.stderr or "").strip()[:2000]}
        pre, func = split_asm(p.stdout, fname)
        return {"pre_asm": pre, "func_asm": func, "n_instr": instruction_count(func)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split", help="e.g. test_real, test_synth, train_real_compilable")
    ap.add_argument("out", help="output .jsonl path")
    ap.add_argument("--limit", type=int, default=None, help="only the first N records")
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--keep-static", action="store_true",
                    help="do NOT strip static/inline (for A/B comparison)")
    args = ap.parse_args()
    levels = [l.strip() for l in args.levels.split(",") if l.strip()]

    stats = {"n": 0, "static_stripped": 0}
    per_level_ok = {l: 0 for l in levels}
    per_level_empty = {l: 0 for l in levels}

    with open(args.out, "w") as out:
        for rec in iter_split(args.split, limit=args.limit):
            fname = rec.get("fname") or ""
            func_def = rec.get("func_def") or ""
            deps = pick_deps(rec)

            if args.keep_static:
                new_def, was_static = func_def, False
            else:
                new_def, was_static = strip_static(func_def)
            stats["static_stripped"] += int(was_static)

            csrc = deps + "\n" + new_def + "\n"
            asm_regen = {}
            for level in levels:
                res = compile_one(csrc, level, fname)
                asm_regen[level] = res
                if "error" not in res:
                    per_level_ok[level] += 1
                    if res["n_instr"] == 0:
                        per_level_empty[level] += 1

            out.write(json.dumps({
                "fname": fname,
                "path": rec.get("path"),
                "signature": rec.get("signature"),
                "func_head_types": rec.get("func_head_types"),
                "func_def": func_def,
                "func_def_compiled": new_def,
                "was_static": was_static,
                "deps": deps,
                "asm_regen": asm_regen,
            }) + "\n")
            stats["n"] += 1
            if stats["n"] % 200 == 0:
                print(f"  ... {stats['n']} records", file=sys.stderr, flush=True)

    print(json.dumps({
        "split": args.split, "out": args.out, **stats,
        "compiled_ok": per_level_ok, "empty_asm": per_level_empty,
    }, indent=2))


if __name__ == "__main__":
    main()
