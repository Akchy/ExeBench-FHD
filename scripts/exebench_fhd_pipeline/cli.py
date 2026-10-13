#!/usr/bin/env python3
"""Run scripts.exebench_fhd_pipeline.core.process_record over a split and
write one merged JSON record per line.

Compile-heavy, so --n/--limit default small. --workers parallelizes across
records; --resume continues a killed run of the same --out path.

Usage:
    python3 -m scripts.exebench_fhd_pipeline --split train_real_compilable --n 20 \\
        --out output_files/exebench_fhd_pipeline_sample.jsonl
    python3 -m scripts.exebench_fhd_pipeline --split test_real --all --workers 32 --resume \\
        --out output_files/test_real_exebench_fhd.jsonl
    python3 -m scripts.exebench_fhd_pipeline --split all --all --limit 5 \\
        --out-dir output_files/exebench_fhd_pipeline_field_sample
"""
import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path

from scripts.check_split_leakage import ALL_SPLITS, iter_split
from scripts.exebench_fhd_pipeline.core import GCC_TIMEOUT, process_record
from scripts.extract.extract_func_call import ARCH_TOOLCHAIN


def _worker_process_record(args):
    """Pool worker. Returns (scanned, record); an exception is recorded in
    `_pipeline_error` instead of killing the pool. `scanned` is returned
    because imap_unordered loses order and --resume needs the exact index.
    """
    scanned, split, task_id, rec, archs, timeout, cleanup, work_dir = args
    try:
        result = process_record(rec, split, work_dir, task_id, archs=archs,
                                 timeout=timeout, cleanup=cleanup)
    except Exception as e:
        result = dict(rec)
        result["_pipeline_error"] = f"{type(e).__name__}: {e}"
    return scanned, result


def _tasks(split, n, limit, write_all, work_dir, archs, timeout, cleanup,
           skip_indices=frozenset()):
    """Lazily yield one task per record, stopping after --n unless --all.
    Indices in `skip_indices` are streamed past but not yielded."""
    scanned = 0
    for rec in iter_split(split, limit=limit):
        scanned += 1
        if scanned not in skip_indices:
            yield (scanned, split, f"{split}_{scanned}", rec, archs, timeout, cleanup, work_dir)
        if not write_all and scanned >= n:
            return


def _progress_path(out_path):
    return f"{out_path}.progress"


def _atomic_rewrite_lines(path, lines):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        for line in lines:
            f.write(line + "\n")
    os.replace(tmp_path, path)


def _load_completed_indices(progress_path):
    """Return the 1-based record indices listed in the progress ledger.
    A torn last line (killed mid-write) is dropped and the ledger rewritten."""
    if not os.path.exists(progress_path):
        return set()
    indices = []
    pending = None
    with open(progress_path) as f:
        for raw in f:
            if not raw.strip():
                continue
            if pending is not None:
                # Only the last line can be torn; anything earlier must parse.
                indices.append(int(pending))
            pending = raw.rstrip("\n")
    torn = False
    if pending is not None:
        try:
            indices.append(int(pending))
        except ValueError:
            torn = True
    if torn:
        _atomic_rewrite_lines(progress_path, [str(idx) for idx in indices])
    return set(indices)


def _reconcile_output_file(out_path, done_count):
    """Trim `out_path` to `done_count` valid JSON lines.

    After a crash the output is at most one record ahead of the ledger, plus
    possibly a torn last line. Streams the file (outputs can be ~100GB) and
    only rewrites it if something needs trimming.
    """
    if not os.path.exists(out_path):
        return

    total = 0
    last_line = None
    with open(out_path) as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            last_line = line
    if total == 0:
        return

    torn = False
    if last_line is not None:
        try:
            json.loads(last_line)
        except json.JSONDecodeError:
            torn = True

    target = min(total - 1 if torn else total, done_count)
    if target == total:
        return

    def _first_n_lines_stripped(path, n):
        with open(path) as f:
            written = 0
            for line in f:
                if not line.strip():
                    continue
                if written >= n:
                    return
                written += 1
                yield line.rstrip("\n")

    _atomic_rewrite_lines(out_path, _first_n_lines_stripped(out_path, target))


def _resume_state(out_path, resume):
    """Return (done_indices, out_mode). Without resume, any stale ledger is
    deleted and the output truncated."""
    progress_path = _progress_path(out_path)
    if not resume:
        if os.path.exists(progress_path):
            os.remove(progress_path)
        return set(), "w"

    done_indices = _load_completed_indices(progress_path)
    _reconcile_output_file(out_path, len(done_indices))
    return done_indices, "a"


def _run_split_parallel(split, out_path, n, limit, write_all, work_dir, archs, timeout,
                         cleanup, workers, resume):
    done_indices, out_mode = _resume_state(out_path, resume)
    written = len(done_indices)

    progress = open(_progress_path(out_path), "a") if resume else None
    try:
        with open(out_path, out_mode) as out, multiprocessing.Pool(workers) as pool:
            for scanned, rec_out in pool.imap_unordered(
                    _worker_process_record,
                    _tasks(split, n, limit, write_all, work_dir, archs, timeout, cleanup,
                           skip_indices=done_indices),
                    chunksize=1):
                out.write(json.dumps(rec_out) + "\n")
                out.flush()
                if progress:
                    progress.write(f"{scanned}\n")
                    progress.flush()
                written += 1
                if written % 20 == 0:
                    print(f"  … [{split}] written {written}", file=sys.stderr, flush=True)
    finally:
        if progress:
            progress.close()

    # Workers never raise, so every dispatched task yields one written record.
    return {"split": split, "out": str(out_path), "scanned": written, "written": written}


def _run_split(split, out_path, n, limit, write_all, work_dir, archs, timeout,
                cleanup, workers=1, resume=False):
    if workers > 1:
        return _run_split_parallel(split, out_path, n, limit, write_all, work_dir, archs,
                                    timeout, cleanup, workers, resume)

    done_indices, out_mode = _resume_state(out_path, resume)
    scanned = 0
    written = len(done_indices)

    progress = open(_progress_path(out_path), "a") if resume else None
    try:
        with open(out_path, out_mode) as out:
            for rec in iter_split(split, limit=limit):
                scanned += 1
                if scanned in done_indices:
                    continue
                task_id = f"{split}_{scanned}"
                rec_out = process_record(
                    rec, split, work_dir, task_id, archs=archs, timeout=timeout,
                    cleanup=cleanup)
                out.write(json.dumps(rec_out) + "\n")
                out.flush()
                if progress:
                    progress.write(f"{scanned}\n")
                    progress.flush()
                written += 1

                if scanned % 20 == 0:
                    print(f"  … [{split}] scanned {scanned}, written {written}",
                          file=sys.stderr, flush=True)

                if not write_all and written >= n:
                    break
    finally:
        if progress:
            progress.close()

    return {"split": split, "out": str(out_path), "scanned": scanned, "written": written}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train_real_compilable", choices=ALL_SPLITS + ["all"],
                     help="one official split, or 'all' for every split in ALL_SPLITS")
    ap.add_argument("--n", type=int, default=20,
                     help="number of records to process (ignored with --all)")
    ap.add_argument("--all", action="store_true",
                     help="process every scanned record instead of the first --n "
                          "(combine with --limit for a capped sample)")
    ap.add_argument("--out", default="output_files/exebench_fhd_pipeline_sample.jsonl",
                     help="output path, used when --split is a single split")
    ap.add_argument("--out-dir", default="output_files/exebench_fhd_pipeline_field",
                     help="directory to write one <split>.jsonl per split into, used only with "
                          "--split all")
    ap.add_argument("--limit", type=int, default=None,
                     help="cap records scanned per split")
    ap.add_argument("--archs", default=",".join(ARCH_TOOLCHAIN),
                     help="comma-separated subset of {} for func_call/data_section "
                          "(all four by default)".format(list(ARCH_TOOLCHAIN)))
    ap.add_argument("--work-dir", default="output_files/exebench_fhd_pipeline_work",
                     help="scratch directory for the .c/.out files the compiling steps write")
    ap.add_argument("--timeout", type=int, default=GCC_TIMEOUT,
                     help="per-subprocess (gcc/objdump) timeout in seconds")
    ap.add_argument("--keep-artifacts", action="store_true",
                     help="don't delete generated .c/.out files (debugging)")
    ap.add_argument("--workers", type=int, default=1,
                     help="worker processes per split (default 1 = sequential)")
    ap.add_argument("--resume", action="store_true",
                     help="continue a killed run of the same --out path instead of "
                          "overwriting it")
    args = ap.parse_args()

    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    unknown = [a for a in archs if a not in ARCH_TOOLCHAIN]
    if unknown:
        ap.error(f"unknown arch(es) {unknown}, choose from {list(ARCH_TOOLCHAIN)}")

    if args.workers < 1:
        ap.error("--workers must be >= 1")

    cleanup = not args.keep_artifacts

    if args.split == "all":
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for split in ALL_SPLITS:
            print(f"=== {split} ===", file=sys.stderr, flush=True)
            out_path = out_dir / f"{split}.jsonl"
            results.append(_run_split(split, out_path, args.n, args.limit, args.all,
                                       args.work_dir, archs, args.timeout, cleanup,
                                       args.workers, args.resume))
        summary = {
            "splits": ALL_SPLITS,
            "per_split": results,
            "total_scanned": sum(r["scanned"] for r in results),
            "total_written": sum(r["written"] for r in results),
        }
        print(json.dumps(summary, indent=2))
    else:
        result = _run_split(args.split, args.out, args.n, args.limit, args.all,
                             args.work_dir, archs, args.timeout, cleanup, args.workers,
                             args.resume)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
