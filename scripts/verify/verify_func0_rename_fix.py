#!/usr/bin/env python3
"""Check that renaming the target to func0 fixes main-named records.

create_c_file renames every target to func0 before appending its `main`
driver, and get_processed_asm restores the `main` label afterwards.
  A. compile every record in --input (main-named functions) via create_c_file
     and confirm no "redefinition of main" errors remain.
  B. run get_processed_asm end-to-end on --sample-n main-named and non-main
     records and confirm both extract asm.

    python3 scripts/verify/verify_func0_rename_fix.py --workers 32 --sample-n 300
"""
import argparse
import json
import multiprocessing
import os
import shutil
import sys

from scripts.check_split_leakage import iter_split
from scripts.preprocess.get_asm import (
    create_c_file, compile_c_program, get_processed_asm,
)
from scripts.regen_asm import pick_deps

SCRATCH_BASE = os.environ.get(
    "FUNC0_RENAME_SCRATCH",
    "output_files/func0_rename_work",
)


def _build_c_func(rec):
    deps = pick_deps(rec)
    func_def = rec.get("func_def") or ""
    return (deps + "\n" + func_def) if deps else func_def


def _read_log_tail(parent_dir, n=800):
    try:
        with open(f"{parent_dir}/gcc_errors.log", "r", errors="replace") as fh:
            return fh.read()[:n]
    except OSError:
        return ""


def _classify(err_text):
    low = err_text.lower()
    if "redefinition" in low and "main" in low:
        return "redefinition_of_main"
    return "other_error"


def _worker_a(args):
    idx, opt, rec = args
    task_id = f"func0_{idx}"
    parent_dir = os.path.join(SCRATCH_BASE, "a", task_id)
    os.makedirs(parent_dir, exist_ok=True)
    c_func = _build_c_func(rec)
    dict_db = {"type": opt, "task_id": task_id, "c_func": c_func, "fname": rec.get("fname")}
    result = {"task_id": task_id, "fname": rec.get("fname"), "path": rec.get("path")}
    try:
        c_file_path = create_c_file(dict_db, parent_dir)
        ok = compile_c_program(c_file_path, opt, timeout=20)
        result["compiled_ok"] = ok
        if ok:
            result["reason"] = "compiled_ok"
        else:
            err_text = _read_log_tail(parent_dir)
            result["reason"] = _classify(err_text)
            result["error_excerpt"] = err_text
    except Exception as e:
        result["compiled_ok"] = False
        result["reason"] = "harness_error"
        result["error_excerpt"] = repr(e)
    finally:
        shutil.rmtree(parent_dir, ignore_errors=True)
    return result


def run_experiment_a(records, out_path, workers, opt):
    tasks = [(i, opt, rec) for i, rec in enumerate(records)]
    counts = {}
    with open(out_path, "w") as out, multiprocessing.Pool(workers) as pool:
        done = 0
        for result in pool.imap_unordered(_worker_a, tasks, chunksize=32):
            out.write(json.dumps(result) + "\n")
            counts[result["reason"]] = counts.get(result["reason"], 0) + 1
            done += 1
            if done % 10000 == 0:
                print(f"  [A] ... {done}/{len(records)}", file=sys.stderr, flush=True)
    return counts


def _worker_b(args):
    idx, opt, rec, label = args
    task_id = f"e2e_{label}_{idx}"
    parent_dir = os.path.join(SCRATCH_BASE, "b", task_id)
    os.makedirs(parent_dir, exist_ok=True)
    c_func = _build_c_func(rec)
    dict_db = {"type": opt, "task_id": task_id, "c_func": c_func, "fname": rec.get("fname")}
    result = {"task_id": task_id, "fname": rec.get("fname"), "path": rec.get("path"), "label": label}
    try:
        norm_asm = get_processed_asm(dict_db, parent_dir, cleanup=True)
        result["extracted"] = bool(norm_asm)
    except Exception as e:
        result["extracted"] = False
        result["error"] = repr(e)
    finally:
        shutil.rmtree(parent_dir, ignore_errors=True)
    return result


def run_experiment_b(main_records, nonmain_records, out_path, workers, opt):
    tasks = ([(i, opt, r, "main_named") for i, r in enumerate(main_records)] +
             [(i, opt, r, "non_main") for i, r in enumerate(nonmain_records)])
    counts = {"main_named": {"extracted": 0, "not_extracted": 0},
              "non_main": {"extracted": 0, "not_extracted": 0}}
    with open(out_path, "w") as out, multiprocessing.Pool(workers) as pool:
        for result in pool.imap_unordered(_worker_b, tasks, chunksize=8):
            out.write(json.dumps(result) + "\n")
            key = "extracted" if result["extracted"] else "not_extracted"
            counts[result["label"]][key] += 1
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="output_files/main_named_functions.jsonl")
    ap.add_argument("--out", default="output_files/func0_rename_results.jsonl")
    ap.add_argument("--e2e-out", default="output_files/func0_rename_e2e_results.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="cap experiment A record count")
    ap.add_argument("--sample-n", type=int, default=300, help="records per group for experiment B")
    ap.add_argument("--workers", type=int, default=min(32, multiprocessing.cpu_count()))
    ap.add_argument("--opt", default="O0")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args()

    os.makedirs(SCRATCH_BASE, exist_ok=True)
    summary = {}

    if not args.skip_a:
        records = []
        with open(args.input) as f:
            for i, line in enumerate(f):
                if args.limit is not None and i >= args.limit:
                    break
                records.append(json.loads(line))
        print(f"[A] Compiling {len(records)} main-named records (renamed to func0, driver "
              f"appended) at -{args.opt} with {args.workers} workers ...", file=sys.stderr)
        counts_a = run_experiment_a(records, args.out, args.workers, args.opt)
        summary["experiment_a"] = {
            "description": "fname=='main': renamed to func0 + driver appended (create_c_file), compile-only",
            "total": len(records), "counts": counts_a,
            "pct": {k: round(100 * v / len(records), 2) for k, v in counts_a.items()},
        }

    if not args.skip_b:
        main_sample = []
        with open(args.input) as f:
            for line in f:
                main_sample.append(json.loads(line))
                if len(main_sample) >= args.sample_n:
                    break
        nonmain_sample = []
        for rec in iter_split("train_real_compilable"):
            if (rec.get("fname") or "") != "main":
                nonmain_sample.append(rec)
                if len(nonmain_sample) >= args.sample_n:
                    break
        print(f"[B] End-to-end get_processed_asm on {len(main_sample)} main-named + "
              f"{len(nonmain_sample)} non-main records ...", file=sys.stderr)
        counts_b = run_experiment_b(main_sample, nonmain_sample, args.e2e_out,
                                     args.workers, args.opt)
        summary["experiment_b"] = {
            "description": "run UNMODIFIED get_processed_asm end-to-end (compile -> objdump "
                            "-> label search: 'main' for main-named records, 'func0' -- via "
                            "rename -- for everything else)",
            "counts": counts_b,
        }

    shutil.rmtree(SCRATCH_BASE, ignore_errors=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
