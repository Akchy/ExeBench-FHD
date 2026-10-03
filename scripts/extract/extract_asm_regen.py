#!/usr/bin/env python3
"""Regenerate assembly with `static`/`inline` stripped from the target.

ExeBench compiles each function alone, so a static or inline function has no
caller and GCC deletes it at -O1 and above, leaving directives-only asm. We
strip those specifiers and mark the target noinline, then compile.

Fields added:
  asm[O0..O3]                  gcc -S output from the host gcc
  asm_was_static               whether func_def had static/inline
  objdump_asm[arch][level]     linked-binary disassembly with addresses and
                               raw bytes, O0-O3 and Os; only with
                               --with-objdump-asm (large, meant for samples),
                               otherwise None. Also rebuilds data_section per
                               level from the same binary.

By default only the first --n records that needed stripping are written;
--all writes every record. --split all writes one file per split to --out-dir.

Usage:
    python3 -m scripts.extract.extract_asm_regen --split train_real_compilable --all \
        --out output_files/train_real_compilable_asm_regen.jsonl
    python3 -m scripts.extract.extract_asm_regen --n 10 --with-objdump-asm \
        --archs x86-64,ARM64 --out output_files/asm_regen_objdump_sample.jsonl
"""
import argparse
import json
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

from scripts.check_split_leakage import ALL_SPLITS, iter_split
from scripts.extract.extract_data_sections import compile_and_link, extract_data_from_binary
from scripts.extract.extract_func_call import ARCH_TOOLCHAIN, disassemble_arch, remove_files
from scripts.preprocess.preprocess_asm import extract_function
from scripts.regen_asm import LEVELS, compile_one, pick_deps, strip_static

GCC_TIMEOUT = 60
OBJDUMP_LEVELS = ["O0", "O1", "O2", "O3", "Os"]

# Functions too large to disassemble. They are still compiled and linked, so
# data_section is extracted as usual; objdump_asm is "" with this reason.
OVERSIZED_FUNCTIONS = {
    ("aosm/gcc_os/gcc/testsuite/gcc.dg/long-branch-middle.c", "dummy"): (
        "skipped_oversized: body is `__asm__(\".fill 32*1024*1024, 4, 0x12345678\")` - "
        "128 MiB of data words, ~33.5M objdump lines per arch/level - not disassembled"
    ),
}


def needs_asm_regen(rec):
    """True if func_def starts with static/inline (text check, no compile)."""
    _, was_static = strip_static(rec.get("func_def") or "")
    return was_static


def build_source(rec):
    """deps + target with static/inline stripped and noinline added.

    noinline also covers the rare in-TU caller (recursion, a sibling in deps).
    Returns (csrc, was_static, fname).
    """
    fname = rec.get("fname") or ""
    func_def = rec.get("func_def") or ""
    deps = pick_deps(rec)

    stripped, was_static = strip_static(func_def)
    noinline_def = "__attribute__((noinline))\n" + stripped.strip() + "\n"

    csrc = (deps + "\n" + noinline_def) if deps else noinline_def
    return csrc, was_static, fname


def build_objdump_source(rec):
    """build_source plus an empty main() so it links (unless the target is main)."""
    csrc, was_static, fname = build_source(rec)
    if fname == "main":
        src = csrc
    else:
        src = csrc + "\nint main(void) {\n    return 0;\n}\n"
    return src, was_static, fname


def _disassemble_function(out_file, objdump_bin, func_name, timeout):
    """Return (snippet, None) for func_name's block, or ("", error)."""
    asm, err = disassemble_arch(out_file, objdump_bin, timeout=timeout, symbol=func_name)
    if asm is None:
        return "", f"disassemble_failed: {err}"
    lines = extract_function(asm, mode="no_process_w_addr", func_name=func_name)
    if not lines:
        return "", "function_not_found_in_disassembly"
    return "\n".join(lines), None


def process_record(rec, work_dir=None, task_id=None, archs=None, timeout=GCC_TIMEOUT, cleanup=True,
                    include_objdump_asm=False):
    """Return `rec` with asm, asm_was_static, objdump_asm and objdump_asm_errors.

    With include_objdump_asm, also compiles every arch in `archs` at each of
    OBJDUMP_LEVELS and sets data_section[arch][level] from the same binaries.
    work_dir defaults to a temporary directory.
    """
    csrc, was_static, fname = build_source(rec)
    asm = {level: compile_one(csrc, level, fname, timeout=timeout) for level in LEVELS}

    out = dict(rec)
    out["asm"] = asm
    out["asm_was_static"] = was_static

    if not include_objdump_asm:
        out["objdump_asm"] = None
        out["objdump_asm_errors"] = None
        return out

    archs = archs or list(ARCH_TOOLCHAIN)
    obj_src, _, target_label = build_objdump_source(rec)

    own_tmp = work_dir is None
    if own_tmp:
        work_dir = tempfile.mkdtemp(prefix="asm_regen_")
    if task_id is None:
        task_id = uuid.uuid4().hex[:12]

    c_dir = Path(work_dir) / "C_files"
    c_dir.mkdir(parents=True, exist_ok=True)
    c_file = c_dir / f"{task_id}.c"
    c_file.write_text(obj_src)
    created = [c_file]

    objdump_asm = {arch: {} for arch in archs}
    objdump_asm_errors = {arch: {} for arch in archs}
    data_section = {arch: {} for arch in archs}
    data_section_errors = {arch: {} for arch in archs}
    try:
        for arch in archs:
            tc = ARCH_TOOLCHAIN[arch]
            for level in OBJDUMP_LEVELS:
                # Disassembly and data_section come from the same binary, so
                # their addresses agree.
                stem = c_dir / f"{task_id}_{arch}_{level}"
                obj_file, out_file, map_file = (stem.with_suffix(ext) for ext in (".o", ".out", ".map"))
                created += [obj_file, out_file, map_file]
                err = compile_and_link(str(c_file), str(obj_file), str(out_file), str(map_file),
                                       tc["compiler"], opt_flag=f"-{level}", timeout=timeout)
                if err:
                    objdump_asm[arch][level] = ""
                    objdump_asm_errors[arch][level] = err
                    data_section[arch][level] = ""
                    data_section_errors[arch][level] = err
                    continue
                oversized = OVERSIZED_FUNCTIONS.get((rec.get("path"), rec.get("fname")))
                if oversized:
                    snippet, derr = "", oversized
                else:
                    snippet, derr = _disassemble_function(out_file, tc["objdump"], target_label, timeout)
                objdump_asm[arch][level] = snippet
                objdump_asm_errors[arch][level] = derr
                try:
                    map_text = map_file.read_text(errors="replace")
                except OSError as e:
                    data_section[arch][level] = ""
                    data_section_errors[arch][level] = f"map_read_failed: {e!r}"
                    continue
                rendered, dserr = extract_data_from_binary(map_text, str(obj_file), str(out_file),
                                                           timeout=timeout, objdump_bin=tc["objdump"])
                data_section[arch][level] = rendered
                data_section_errors[arch][level] = dserr
    finally:
        if cleanup:
            remove_files(created)
            if own_tmp:
                shutil.rmtree(work_dir, ignore_errors=True)

    out["objdump_asm"] = objdump_asm
    out["objdump_asm_errors"] = objdump_asm_errors
    out["data_section"] = data_section
    out["data_section_errors"] = data_section_errors
    return out


def _run_split(split, out_path, n, limit, write_all, timeout, archs, work_dir, cleanup, with_objdump_asm):
    scanned = 0
    matched = 0
    per_level_ok = {l: 0 for l in LEVELS}
    per_level_empty = {l: 0 for l in LEVELS}

    with open(out_path, "w") as out:
        for rec in iter_split(split, limit=limit):
            scanned += 1
            flagged = needs_asm_regen(rec)

            if write_all or flagged:
                matched += int(flagged) if write_all else 1
                task_id = f"ar_{split}_{scanned}"
                rec_out = process_record(rec, work_dir=work_dir, task_id=task_id,
                                          archs=archs, timeout=timeout, cleanup=cleanup,
                                          include_objdump_asm=with_objdump_asm)
                for level in LEVELS:
                    res = rec_out["asm"][level]
                    if "error" not in res:
                        per_level_ok[level] += 1
                        if res["n_instr"] == 0:
                            per_level_empty[level] += 1
                out.write(json.dumps(rec_out) + "\n")
                if not write_all and matched >= n:
                    break

            if scanned % 20_000 == 0:
                print(f"  [{split}] scanned {scanned}, matched {matched}",
                      file=sys.stderr, flush=True)

    return {
        "split": split, "out": str(out_path), "scanned": scanned, "matched": matched,
        "compiled_ok": per_level_ok, "empty_asm": per_level_empty,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train_real_compilable", choices=ALL_SPLITS + ["all"],
                     help="one official split, or 'all' for every split in ALL_SPLITS")
    ap.add_argument("--n", type=int, default=100,
                     help="number of matching (asm_was_static=True) rows to collect (ignored with --all)")
    ap.add_argument("--all", action="store_true",
                     help="run every scanned record through process_record() (fields added for both "
                          "True and False asm_was_static records), instead of collecting only the "
                          "first --n records that needed stripping")
    ap.add_argument("--out", default="output_files/asm_regen_sample.jsonl",
                     help="output path, used when --split is a single split")
    ap.add_argument("--out-dir", default="output_files/asm_regen_field",
                     help="directory to write one <split>.jsonl per split into, used only with --split all")
    ap.add_argument("--limit", type=int, default=None,
                     help="cap records scanned per split (debugging, or a quick per-split sample with --all)")
    ap.add_argument("--timeout", type=int, default=GCC_TIMEOUT)
    ap.add_argument("--with-objdump-asm", action="store_true",
                     help="also compute the multi-architecture objdump_asm field "
                          "(large; use with --n/--limit for audit samples)")
    ap.add_argument("--archs", default=",".join(ARCH_TOOLCHAIN),
                     help="comma-separated subset of {} for objdump_asm "
                          "(default: all)".format(list(ARCH_TOOLCHAIN)))
    ap.add_argument("--work-dir", default="output_files/asm_regen_work",
                     help="scratch directory for the .c/.out files objdump_asm compiles")
    ap.add_argument("--keep-artifacts", action="store_true",
                     help="don't delete generated .c/.out files (debugging)")
    args = ap.parse_args()

    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    unknown = [a for a in archs if a not in ARCH_TOOLCHAIN]
    if unknown:
        ap.error(f"unknown arch(es) {unknown}, choose from {list(ARCH_TOOLCHAIN)}")

    if args.split == "all":
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for split in ALL_SPLITS:
            print(f"=== {split} ===", file=sys.stderr, flush=True)
            out_path = out_dir / f"{split}.jsonl"
            results.append(_run_split(split, out_path, args.n, args.limit, args.all, args.timeout,
                                       archs, args.work_dir, not args.keep_artifacts, args.with_objdump_asm))
        summary = {
            "splits": ALL_SPLITS,
            "per_split": results,
            "total_scanned": sum(r["scanned"] for r in results),
            "total_matched": sum(r["matched"] for r in results),
        }
        print(json.dumps(summary, indent=2))
    else:
        result = _run_split(args.split, args.out, args.n, args.limit, args.all, args.timeout,
                             archs, args.work_dir, not args.keep_artifacts, args.with_objdump_asm)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
