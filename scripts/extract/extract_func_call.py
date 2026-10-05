#!/usr/bin/env python3
"""Extract the `func_call` field: the call-site assembly of each function on
x86, x86-64, ARM32 and ARM64.

On x86-64 an unused or reordered float argument leaves no gap in the integer
registers, so the target's own optimized asm can hide parameter order. We
wrap the target in a driver that calls it (noipa) with argv-derived
arguments, compile per architecture, and keep the instructions that set up
the call. Functions with a parameter type we can't synthesize (struct by
value, SIMD, unknown typedefs) are written to --skipped-out with a reason.

Needs libc6-dev-i386 plus the ARM cross compilers (not gcc-multilib, which
conflicts with them).

Usage:
    python3 -m scripts.extract.extract_func_call --split train_real_compilable --n 20 \
        --out output_files/func_call_sample.jsonl \
        --skipped-out output_files/func_call_skipped.jsonl
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.check_split_leakage import ALL_SPLITS, iter_split
from scripts.preprocess.get_asm import rename_to_func0
from scripts.preprocess.preprocess_asm import extract_function
from scripts.regen_asm import pick_deps, strip_static
from scripts.survey.survey_func_call_param_types import categorize

GCC_TIMEOUT = 60
OPT_FLAG = "-O3"  # -O0 never shows the ambiguity
# Used only in per-level mode (--levels); the default output is a single -O3 build.
FUNC_CALL_LEVELS = ["O0", "O1", "O2", "O3", "Os"]

ARCH_TOOLCHAIN = {
    "x86":     {"compiler": ["gcc", "-m32"], "objdump": "objdump"},
    "x86-64":  {"compiler": ["gcc", "-m64"], "objdump": "objdump"},
    "ARM32":   {"compiler": ["arm-linux-gnueabihf-gcc"], "objdump": "arm-linux-gnueabihf-objdump"},
    "ARM64":   {"compiler": ["aarch64-linux-gnu-gcc"], "objdump": "aarch64-linux-gnu-objdump"},
}

# No safe generic argument synthesis for these, so the whole function is skipped.
_SKIP_CATEGORIES = {"struct_or_union_by_value", "simd_or_complex", "uncategorized"}


def analyze_params(signature):
    """Return (params, skip_reason) for signature[1:].

    `params` is a list of (type_text, category); `(void)` and a trailing `...`
    are dropped. If any parameter is unsupported, returns (None, category).
    """
    params = []
    for ptype in signature[1:]:
        cat = categorize(ptype)
        if cat == "void_placeholder":
            continue
        if cat == "varargs":
            break
        if cat in _SKIP_CATEGORIES:
            return None, cat
        params.append((ptype.strip(), cat))
    return params, None


def build_arg_expr(ptype, argv_index):
    """C expression of type `ptype` read from argv[argv_index].

    The driver is never run, so any pointer type can just be argv[i] cast;
    it only has to be something the compiler can't constant-fold.
    """
    cat = categorize(ptype)
    if cat == "int_family":
        return f"({ptype})strtoll(argv[{argv_index}], NULL, 10)"
    if cat == "float_family":
        return f"({ptype})strtod(argv[{argv_index}], NULL)"
    return f"({ptype})(void*)argv[{argv_index}]"


# noinline alone isn't enough at -O3: GCC can still clone the target
# (func0.constprop.0, func0.isra.0) with parameters removed, or use IPA register
# allocation to move main()'s own code across the call. noipa (GCC >= 8)
# prevents both. The check is nested because `#if __has_attribute(...)` is a
# preprocessing error on compilers that lack __has_attribute.
_NOIPA_MACRO = "__FUNC_CALL_NOIPA"
_NOIPA_PRELUDE = (
    "#if defined(__has_attribute)\n"
    "#  if __has_attribute(noipa)\n"
    f"#    define {_NOIPA_MACRO} __attribute__((noipa))\n"
    "#  endif\n"
    "#endif\n"
    f"#ifndef {_NOIPA_MACRO}\n"
    f"#  define {_NOIPA_MACRO} __attribute__((noinline))\n"
    "#endif\n"
)


def build_driver_source(rec, fname, ret_type, params):
    """Standalone .c source: deps, the target renamed to func0 (noipa, not
    static), and a main() that calls it with runtime-unknown arguments."""
    # Only one deps source: angha_deps' stub libc prototypes conflict with
    # real_deps' real headers.
    deps = pick_deps(rec).strip()
    func_def = rec.get("func_def") or ""

    renamed = rename_to_func0(func_def, fname) if fname != "func0" else func_def
    stripped, _ = strip_static(renamed)
    noipa_def = f"{_NOIPA_MACRO}\n" + stripped.strip() + "\n"

    c_func = (deps + "\n\n" + noipa_def) if deps else noipa_def

    arg_exprs = [build_arg_expr(ptype, i + 1) for i, (ptype, _cat) in enumerate(params)]
    call_expr = f"func0({', '.join(arg_exprs)})"

    if ret_type == "void":
        body = f"    {call_expr};\n"
    else:
        # Reading one byte of the result through volatile keeps it live for any
        # return type, including structs.
        body = (
            f"    {ret_type} __func_call_result = {call_expr};\n"
            f"    __func_call_sink ^= *((volatile unsigned char *)&__func_call_result);\n"
        )

    # __func_call_marker is called right before the argument setup and gives
    # extract_call_window() an anchor on every architecture.
    return (
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n\n"
        f"{_NOIPA_PRELUDE}\n"
        "static volatile unsigned char __func_call_sink;\n\n"
        "__attribute__((noinline)) static void __func_call_marker(void) {\n"
        "    __func_call_sink ^= 1;\n"
        "}\n\n"
        f"{c_func}\n"
        "int main(int argc, char **argv) {\n"
        "    (void)argc;\n"
        "    __func_call_marker();\n"
        f"{body}"
        "    return 0;\n"
        "}\n"
    )


def remove_files(paths):
    """Delete exactly `paths`, ignoring missing ones.

    Not a glob on task_id: in a parallel run, `x_1*` also matches `x_10`.
    """
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


def run_tool(cmd, timeout):
    """`subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)`
    for compiler/linker calls, minus two leaks. GCC writes its intermediate
    objects to $TMPDIR (default /tmp) and deletes them only on a normal
    exit, and a plain timeout kills just the `gcc` driver, not its cc1/as/ld
    children - so a timed-out build of a function with a huge static array
    left a 17 GB `/tmp/cc*.o` behind per attempt (172 GB from one row). Here
    each call gets a private TMPDIR that is always removed, and a timeout
    kills the whole process group. Raises TimeoutExpired / OSError as
    `subprocess.run` does."""
    tmp = tempfile.mkdtemp(prefix="exb_tool_")
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             errors="replace", env={**os.environ, "TMPDIR": tmp}, start_new_session=True)
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.communicate()
            raise
        return subprocess.CompletedProcess(cmd, p.returncode, out, err)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def compile_arch(source_file, output_file, compiler_argv, opt_flag=OPT_FLAG, timeout=GCC_TIMEOUT):
    cmd = compiler_argv + [opt_flag, str(source_file), "-o", str(output_file), "-lm"]
    try:
        # errors="replace": GCC echoes source string literals, which need not be UTF-8.
        result = run_tool(cmd, timeout)
        if result.returncode == 0:
            return True, None
        return False, result.stderr.strip()[:2000]
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except OSError as e:
        return False, f"toolchain not found: {e!r}"


def disassemble_arch(output_file, objdump_bin, timeout=GCC_TIMEOUT, symbol=None):
    """`objdump -d`, optionally restricted to `symbol`. Restricting matters:
    some binaries hold a huge unrelated function that takes minutes to print."""
    cmd = [objdump_bin, "-d", *([f"--disassemble={symbol}"] if symbol else []), str(output_file)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
        if result.returncode == 0:
            return result.stdout, None
        return None, result.stderr.strip()[:2000]
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout}s"
    except OSError as e:
        return None, f"toolchain not found: {e!r}"


# The suffix allows for GCC clones such as func0.constprop.0 or func0.isra.1.
_MARKER_CALL_RE = re.compile(r"<__func_call_marker(?:\.\w+)*>")
_TARGET_CALL_RE = re.compile(r"<func0(?:\.\w+)*>")


def extract_call_window(asm):
    """Lines of main() after the marker call up to and including the func0
    call, with addresses and raw bytes kept. None if either anchor is missing
    or they are out of order."""
    lines = extract_function(asm, mode="no_process_w_addr", func_name="main")
    if not lines:
        return None

    marker_idx = None
    target_idx = None
    for i, line in enumerate(lines):
        if marker_idx is None and _MARKER_CALL_RE.search(line):
            marker_idx = i
        if _TARGET_CALL_RE.search(line):
            target_idx = i

    if marker_idx is None or target_idx is None or target_idx <= marker_idx:
        return None

    window = lines[marker_idx + 1: target_idx + 1]
    return "\n".join(window) if window else None


def process_record(rec, work_dir, task_id, archs=None, timeout=GCC_TIMEOUT, cleanup=True, levels=None):
    """Return (record_out, skip_reason).

    On success `func_call[arch]` is the snippet, or "" if that architecture
    was tried and failed (reason in `func_call_errors[arch]`). If all four
    architectures fail, skip_reason is "all_architectures_failed". Skips
    decided before compiling leave both fields None.

    With `levels` (e.g. FUNC_CALL_LEVELS), each arch maps to {level: ...};
    errors are normalized and collapsed to a single value when identical
    across levels. Use func_call_by_arch to read either shape.
    """
    archs = archs or list(ARCH_TOOLCHAIN)
    fname = rec.get("fname")
    signature = rec.get("signature") or []

    if not signature:
        return _skipped(rec, "missing_signature")
    if fname == "main":
        return _skipped(rec, "fname_is_main")

    ret_type = (signature[0] or "void").strip() or "void"
    params, skip_reason = analyze_params(signature)
    if skip_reason:
        return _skipped(rec, f"unsupported_param_type:{skip_reason}")

    driver_src = build_driver_source(rec, fname, ret_type, params)

    c_dir = Path(work_dir) / "C_files"
    c_dir.mkdir(parents=True, exist_ok=True)
    c_file = c_dir / f"{task_id}.c"
    c_file.write_text(driver_src)
    created = [c_file]

    func_call = {}
    func_call_errors = {}
    try:
        for arch in archs:
            tc = ARCH_TOOLCHAIN[arch]
            if levels is None:
                out_file = c_dir / f"{task_id}_{arch}.out"
                created.append(out_file)
                func_call[arch], func_call_errors[arch] = _call_site(c_file, out_file, tc, OPT_FLAG, timeout)
                continue
            func_call[arch], func_call_errors[arch] = {}, {}
            for level in levels:
                out_file = c_dir / f"{task_id}_{arch}_{level}.out"
                created.append(out_file)
                func_call[arch][level], func_call_errors[arch][level] = _call_site(
                    c_file, out_file, tc, f"-{level}", timeout)
            func_call_errors[arch] = collapse_identical(
                {lv: normalize_error(e) for lv, e in func_call_errors[arch].items()})
    finally:
        if cleanup:
            remove_files(created)

    rec_out = dict(rec)
    rec_out["func_call"] = func_call
    rec_out["func_call_errors"] = func_call_errors
    # Only a full four-arch run can claim that nothing works.
    if set(archs) == set(ARCH_TOOLCHAIN) and not any(_has_snippet(func_call.get(arch)) for arch in archs):
        rec_out["func_call_skip_reason"] = "all_architectures_failed"
        return rec_out, "all_architectures_failed"
    rec_out["func_call_skip_reason"] = None
    return rec_out, None


def _call_site(c_file, out_file, tc, opt_flag, timeout):
    """Compile, disassemble main, slice the call window. Returns (snippet, None) or ("", error)."""
    ok, err = compile_arch(c_file, out_file, tc["compiler"], opt_flag=opt_flag, timeout=timeout)
    if not ok:
        return "", f"compile_failed: {err}"
    asm, derr = disassemble_arch(out_file, tc["objdump"], timeout=timeout, symbol="main")
    if not asm:
        return "", f"disassemble_failed: {derr}"
    snippet = extract_call_window(asm)
    if not snippet:
        return "", "call_site_not_found"
    return snippet, None


_ERR_OUT_RE = re.compile(r"\S*/C_files/\S+?_(?:x86-64|x86|ARM32|ARM64)(?:_O[0-3s])?\.out")
_ERR_WORK_RE = re.compile(r"\S*/C_files/")
_ERR_TMP_RE = re.compile(r"/tmp/(?:exb_tool_\w+/)?cc\w+\.(o|s)")  # bare /tmp, or run_tool's private TMPDIR
_ERR_SLINE_RE = re.compile(r"(<tmp>\.s):\d+:")
_ERR_OFF_RE = re.compile(r"\(([.\w]+)\+0x[0-9a-f]+\)")


def normalize_error(err):
    """Mask per-invocation noise in a toolchain message (work-dir paths, GCC
    temp file names and their line numbers, code offsets) so the same error
    compares equal across levels and runs."""
    if err is None:
        return None
    err = _ERR_OUT_RE.sub("<out>", err)
    err = _ERR_WORK_RE.sub("<work>/", err)
    err = _ERR_TMP_RE.sub(r"<tmp>.\1", err)
    err = _ERR_SLINE_RE.sub(r"\1:<line>:", err)
    return _ERR_OFF_RE.sub(r"(\1+<off>)", err)


def collapse_identical(per_level):
    """{level: error} -> the shared value if all levels agree, else unchanged."""
    values = list(per_level.values())
    return values[0] if values and all(v == values[0] for v in values) else per_level


def func_call_by_arch(field, level="O3"):
    """`func_call` or `func_call_errors` as {arch: value} at one level.

    Handles per-level rows, collapsed errors, and single -O3 rows ({arch: v},
    returned as-is for any level). None stays None.
    """
    if field is None:
        return None
    return {arch: (v.get(level) if isinstance(v, dict) else v) for arch, v in field.items()}


def _has_snippet(value):
    return any(value.values()) if isinstance(value, dict) else bool(value)


def _skipped(rec, reason):
    rec_out = dict(rec)
    rec_out["func_call"] = None
    rec_out["func_call_errors"] = None
    rec_out["func_call_skip_reason"] = reason
    return rec_out, reason


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train_real_compilable", choices=ALL_SPLITS)
    ap.add_argument("--n", type=int, default=20, help="number of non-skipped records to collect")
    ap.add_argument("--out", default="output_files/func_call_sample.jsonl")
    ap.add_argument("--skipped-out", default="output_files/func_call_skipped.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="cap records scanned (debugging)")
    ap.add_argument("--work-dir", default="output_files/func_call_work")
    ap.add_argument("--timeout", type=int, default=GCC_TIMEOUT)
    ap.add_argument("--keep-artifacts", action="store_true",
                     help="don't delete generated .c/.out files (debugging)")
    ap.add_argument("--archs", default=",".join(ARCH_TOOLCHAIN),
                     help="comma-separated subset of {} (default: all)".format(list(ARCH_TOOLCHAIN)))
    ap.add_argument("--levels", default=None,
                     help="comma-separated opt levels (e.g. {}) for per-level output "
                          "func_call[arch][level]; default: single {} build, "
                          "func_call[arch]".format(",".join(FUNC_CALL_LEVELS), OPT_FLAG))
    args = ap.parse_args()
    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()] if args.levels else None
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    unknown = [a for a in archs if a not in ARCH_TOOLCHAIN]
    if unknown:
        ap.error(f"unknown arch(es) {unknown}, choose from {list(ARCH_TOOLCHAIN)}")

    scanned = 0
    matched = 0
    skipped = 0
    skip_reasons = {}

    with open(args.out, "w") as out_f, open(args.skipped_out, "w") as skip_f:
        for rec in iter_split(args.split, limit=args.limit):
            scanned += 1
            task_id = f"fc_{scanned}"
            rec_out, skip_reason = process_record(
                rec, args.work_dir, task_id, archs=archs,
                timeout=args.timeout, cleanup=not args.keep_artifacts, levels=levels,
            )
            if skip_reason is None:
                out_f.write(json.dumps(rec_out) + "\n")
                matched += 1
            else:
                skip_f.write(json.dumps(rec_out) + "\n")
                skipped += 1
                skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1

            if scanned % 1000 == 0:
                print(f"  scanned {scanned}, matched {matched}, skipped {skipped}",
                      file=sys.stderr, flush=True)

            if matched >= args.n:
                break

    print(json.dumps({
        "split": args.split,
        "out": args.out,
        "skipped_out": args.skipped_out,
        "scanned": scanned,
        "matched": matched,
        "skipped": skipped,
        "skip_reasons": skip_reasons,
    }, indent=2))


if __name__ == "__main__":
    main()
