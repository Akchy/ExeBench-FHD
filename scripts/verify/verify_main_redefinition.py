#!/usr/bin/env python3
"""Reproduce the "redefinition of main" failure for main-named records without the func0 fix.

dict_db is built without `fname`, so create_c_file skips the rename and the
target's own `main` collides with the appended driver at compile time. Each
record in --input is classified as compiled_ok, redefinition_of_main or
other_error. See verify_func0_rename_fix.py for the fixed path.

    python3 scripts/verify/verify_main_redefinition.py --limit 2000 --workers 32
"""
import argparse
import json
import multiprocessing
import os
import shutil
import sys

from scripts.preprocess.get_asm import create_c_file, compile_c_program
from scripts.regen_asm import pick_deps

SCRATCH_BASE = os.environ.get(
    "MAIN_REDEF_SCRATCH",
    "output_files/main_redefinition_work",
)


def _read_log_tail(parent_dir):
    log_path = f"{parent_dir}/gcc_errors.log"
    try:
        with open(log_path, "r", errors="replace") as fh:
            return fh.read()[-4000:]
    except OSError:
        return ""


def _classify(err_text):
    low = err_text.lower()
    if "redefinition" in low and "main" in low:
        return "redefinition_of_main"
    return "other_error"


def _worker(args):
    idx, opt, rec = args
    task_id = f"main_{idx}"
    # Per-record dir so concurrent workers don't share a gcc_errors.log.
    parent_dir = os.path.join(SCRATCH_BASE, task_id)
    os.makedirs(parent_dir, exist_ok=True)

    deps = pick_deps(rec)
    func_def = rec.get("func_def") or ""
    c_func = (deps + "\n" + func_def) if deps else func_def

    dict_db = {"type": opt, "task_id": task_id, "c_func": c_func}
    result = {
        "task_id": task_id,
        "fname": rec.get("fname"),
        "path": rec.get("path"),
    }
    try:
        c_file_path = create_c_file(dict_db, parent_dir)
        ok = compile_c_program(c_file_path, opt, timeout=20)
        result["compiled_ok"] = ok
        if ok:
            result["reason"] = "compiled_ok"
        else:
            err_text = _read_log_tail(parent_dir)
            result["reason"] = _classify(err_text)
            # gcc's first error line is the useful one.
            result["error_excerpt"] = err_text[:800]
    except Exception as e:  # never let one bad record kill the pool
        result["compiled_ok"] = False
        result["reason"] = "harness_error"
        result["error_excerpt"] = repr(e)
    finally:
        shutil.rmtree(parent_dir, ignore_errors=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="output_files/main_named_functions.jsonl")
    ap.add_argument("--out", default="output_files/main_redefinition_results.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=min(32, multiprocessing.cpu_count()))
    ap.add_argument("--opt", default="O0", help="single opt level - the failure is "
                    "parse-time, so it does not vary with -O0..-O3")
    args = ap.parse_args()

    os.makedirs(SCRATCH_BASE, exist_ok=True)

    records = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            records.append(json.loads(line))

    print(f"Loaded {len(records)} records; compiling with {args.workers} workers "
          f"at -{args.opt} ...", file=sys.stderr)

    tasks = [(i, args.opt, rec) for i, rec in enumerate(records)]
    counts = {"compiled_ok": 0, "redefinition_of_main": 0, "other_error": 0, "harness_error": 0}

    with open(args.out, "w") as out, multiprocessing.Pool(args.workers) as pool:
        done = 0
        for result in pool.imap_unordered(_worker, tasks, chunksize=32):
            out.write(json.dumps(result) + "\n")
            counts[result["reason"]] = counts.get(result["reason"], 0) + 1
            done += 1
            if done % 5000 == 0:
                print(f"  ... {done}/{len(records)}", file=sys.stderr, flush=True)

    shutil.rmtree(SCRATCH_BASE, ignore_errors=True)

    total = len(records)
    summary = {
        "input": args.input,
        "out": args.out,
        "opt": args.opt,
        "total": total,
        "counts": counts,
        "pct": {k: round(100 * v / total, 2) if total else 0.0 for k, v in counts.items()},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
