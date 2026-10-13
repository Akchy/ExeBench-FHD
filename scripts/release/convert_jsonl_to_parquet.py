#!/usr/bin/env python3
"""Convert the JSONL release splits to Parquet shards (data/<split>-XXXXX-of-NNNNN.parquet) for
Hugging Face. Nested fields in JSON_COLUMNS are stored as JSON strings since several have no fixed
type. Each input chunk becomes one shard, verified against its input; reruns skip finished chunks.

    python -m scripts.release.convert_jsonl_to_parquet --input-dir datasets_jsonl_v2 \\
        --output-dir datasets_parquet [--files test_real.jsonl ...] [--workers 24] [--chunk-gb 4]
"""
import argparse
import glob
import json
import multiprocessing
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq

# Column order: ExeBench's 23 fields, then the 16 ExeBench-FHD adds.
EXEBENCH_FIELDS = [
    "path", "func_def", "func_head", "func_head_types", "fname", "signature", "doc",
    "angha_error", "real_error", "asm", "angha_deps", "real_deps",
    "angha_io_pairs", "real_io_pairs", "angha_exe_wrapper", "real_exe_wrapper",
    "angha_io_error", "real_io_error", "angha_io_pairs_are_trivial", "real_io_pairs_are_trivial",
    "angha_iospec", "real_iospec", "ref",
]
EXEBENCH_FHD_FIELDS = [
    "objdump_asm", "objdump_asm_errors", "func_call", "func_call_errors", "func_call_skip_reason",
    "data_section", "data_section_errors",
    "opt_real_exe_wrapper", "opt_real_exe_wrapper_error",
    "opt_angha_exe_wrapper", "opt_angha_exe_wrapper_error",
    "trivial_function", "no_return", "unused_param", "truncated_funcs", "dummy_udt",
]
FIELDS = EXEBENCH_FIELDS + EXEBENCH_FHD_FIELDS

JSON_COLUMNS = [
    "asm", "angha_io_pairs", "real_io_pairs", "angha_iospec", "real_iospec",
    "objdump_asm", "objdump_asm_errors", "func_call", "func_call_errors",
    "data_section", "data_section_errors",
]
BOOL_COLUMNS = [
    "angha_io_pairs_are_trivial", "real_io_pairs_are_trivial",
    "trivial_function", "no_return", "unused_param", "truncated_funcs", "dummy_udt",
]
LIST_COLUMNS = ["signature"]


def _arrow_type(field):
    if field in BOOL_COLUMNS:
        return pa.bool_()
    if field in LIST_COLUMNS:
        return pa.list_(pa.string())
    return pa.string()


SCHEMA = pa.schema(
    [(f, _arrow_type(f)) for f in FIELDS],
    metadata={"exebench_fhd": json.dumps({"json_columns": JSON_COLUMNS})},
)

_SPLIT_RE = re.compile(r"^\w+$")


def encode_row(rec):
    """A JSONL record -> {column: value} in FIELDS order; raises if its keys are not exactly FIELDS."""
    if set(rec) != set(FIELDS):
        missing, extra = sorted(set(FIELDS) - set(rec)), sorted(set(rec) - set(FIELDS))
        raise ValueError(f"record fields differ from the release schema: missing={missing} extra={extra}")
    return {f: (json.dumps(rec[f], ensure_ascii=False) if f in JSON_COLUMNS and rec[f] is not None else rec[f])
            for f in FIELDS}


def decode_row(row):
    """Inverse of encode_row."""
    return {f: (json.loads(row[f]) if f in JSON_COLUMNS and row[f] is not None else row[f]) for f in FIELDS}


def rows_to_table(rows):
    return pa.Table.from_pylist(rows, schema=SCHEMA)


def _canonical(rec):
    # JSON text rather than ==, since NaN != NaN and True == 1. Keys are sorted because the
    # JSONL key order differs from FIELDS.
    return json.dumps(rec, ensure_ascii=False, sort_keys=True)


def chunk_ranges(path, chunk_bytes):
    """Split a JSONL file into [start, end) byte ranges aligned to line starts."""
    size = os.path.getsize(path)
    bounds = [0]
    with open(path, "rb") as f:
        while bounds[-1] + chunk_bytes < size:
            f.seek(bounds[-1] + chunk_bytes)
            f.readline()
            pos = f.tell()
            if pos >= size:
                break
            bounds.append(pos)
    bounds.append(size)
    return list(zip(bounds[:-1], bounds[1:]))


def _iter_lines(path, start, end):
    with open(path, "rb") as f:
        f.seek(start)
        while f.tell() < end:
            line = f.readline()
            if line.strip():
                yield line


def verify_shard(shard_path, input_path, start, end):
    """Check each shard row decodes to the matching line of input_path[start:end]; returns the count."""
    pf = pq.ParquetFile(shard_path)
    rows = (row for i in range(pf.num_row_groups) for row in pf.read_row_group(i).to_pylist())
    n = 0
    for line in _iter_lines(input_path, start, end):
        row = next(rows, None)
        if row is None:
            raise ValueError(f"{shard_path}: fewer rows than input (stopped at {n})")
        if _canonical(decode_row(row)) != _canonical(json.loads(line)):
            raise ValueError(f"{shard_path}: row {n} does not round-trip")
        n += 1
    if next(rows, None) is not None:
        raise ValueError(f"{shard_path}: more rows than input ({n})")
    return n


def convert_chunk(input_path, tmp_dir, chunk_idx, start, end, batch_rows, verify=True):
    """Convert one byte range into one verified shard; returns (chunk_idx, rows, shard_path)."""
    shard = os.path.join(tmp_dir, f"chunk{chunk_idx:05d}.parquet")
    marker = shard + ".done"
    if os.path.exists(marker):
        with open(marker) as f:
            return chunk_idx, json.load(f)["rows"], shard

    partial = shard + ".partial"
    rows_done, batch = 0, []
    with pq.ParquetWriter(partial, SCHEMA, compression="zstd", compression_level=9) as writer:
        for line in _iter_lines(input_path, start, end):
            batch.append(encode_row(json.loads(line)))
            if len(batch) >= batch_rows:
                writer.write_table(rows_to_table(batch))
                rows_done += len(batch)
                batch = []
        if batch:
            writer.write_table(rows_to_table(batch))
            rows_done += len(batch)
    if verify and verify_shard(partial, input_path, start, end) != rows_done:
        raise ValueError(f"{partial}: verified row count differs from written ({rows_done})")
    os.replace(partial, shard)

    with open(marker + ".tmp", "w") as f:
        json.dump({"rows": rows_done, "start": start, "end": end, "verified": verify}, f)
    os.replace(marker + ".tmp", marker)
    return chunk_idx, rows_done, shard


def finalize_split(output_dir, split, tmp_dir, results):
    """Move the chunk shards into data/ in input order and record the split in splits.json."""
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    shards = [s for _, _, s in sorted(results)]
    for stale in glob.glob(os.path.join(data_dir, f"{split}-*-of-*.parquet")):
        os.remove(stale)
    n_bytes = 0
    for i, shard in enumerate(shards):
        dst = os.path.join(data_dir, f"{split}-{i:05d}-of-{len(shards):05d}.parquet")
        os.replace(shard, dst)
        n_bytes += os.path.getsize(dst)
    for marker in glob.glob(os.path.join(tmp_dir, "*.done")):
        os.remove(marker)
    os.rmdir(tmp_dir)

    stats = {"rows": sum(r for _, r, _ in results), "shards": len(shards), "bytes": n_bytes}
    stats_path = os.path.join(output_dir, "splits.json")
    all_stats = {}
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            all_stats = json.load(f)
    all_stats[split] = stats
    with open(stats_path + ".tmp", "w") as f:
        json.dump(dict(sorted(all_stats.items())), f, indent=2)
    os.replace(stats_path + ".tmp", stats_path)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default="datasets_jsonl_v2")
    ap.add_argument("--output-dir", default="datasets_parquet")
    ap.add_argument("--files", nargs="*", help="JSONL basenames (default: all *.jsonl in --input-dir)")
    ap.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    ap.add_argument("--chunk-gb", type=float, default=4.0, help="input JSONL bytes per chunk (= per output shard)")
    ap.add_argument("--batch-rows", type=int, default=500, help="rows per Parquet row group")
    ap.add_argument("--no-verify", action="store_true", help="skip re-reading each shard against its input")
    args = ap.parse_args(argv)

    files = args.files or sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.input_dir, "*.jsonl")))
    for fname in files:
        if not fname.endswith(".jsonl") or not _SPLIT_RE.match(fname[:-len(".jsonl")]):
            sys.exit(f"not a <split>.jsonl file with a valid split name: {fname}")
    os.makedirs(args.output_dir, exist_ok=True)

    jobs = {}  # split -> (tmp_dir, n_chunks)
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {}
        for fname in files:
            split = fname[:-len(".jsonl")]
            input_path = os.path.join(args.input_dir, fname)
            tmp_dir = os.path.join(args.output_dir, f".tmp_{split}")
            os.makedirs(tmp_dir, exist_ok=True)
            ranges = chunk_ranges(input_path, int(args.chunk_gb * 2**30))
            jobs[split] = (tmp_dir, len(ranges))
            for i, (start, end) in enumerate(ranges):
                fut = pool.submit(convert_chunk, input_path, tmp_dir, i, start, end,
                                  args.batch_rows, not args.no_verify)
                futures[fut] = split
            print(f"[{split}] {len(ranges)} chunk(s) queued", flush=True)

        results = {split: [] for split in jobs}
        t0 = time.time()
        failed = False
        for fut in as_completed(futures):
            split = futures[fut]
            try:
                results[split].append(fut.result())
            except Exception as e:  # keep going; a rerun resumes
                failed = True
                print(f"[{split}] chunk FAILED: {e!r}", file=sys.stderr, flush=True)
                continue
            done, total = len(results[split]), jobs[split][1]
            print(f"[{split}] chunk {done}/{total} done ({time.time() - t0:.0f}s)", flush=True)
            if done == total:
                stats = finalize_split(args.output_dir, split, jobs[split][0], results[split])
                print(f"[{split}] DONE: {stats['rows']} rows, {stats['shards']} shard(s), "
                      f"{stats['bytes'] / 2**30:.2f} GiB", flush=True)

    if failed:
        sys.exit("some chunks failed; fix and rerun (completed chunks are reused)")


if __name__ == "__main__":
    main()
