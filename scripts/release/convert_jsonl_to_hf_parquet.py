#!/usr/bin/env python3
"""Export ExeBench-FHD JSONL splits as Parquet in ExeBench's Hugging Face schema.

Why this exists
----------------
ExeBench-FHD JSONL keeps ExeBench's *raw* release layout (the `.jsonl.zst` rows inside
the HF tarballs). Most ExeBench users never see that layout: they call
`load_dataset('jordiae/exebench', split=...)`, whose loading script (`exebench.py`)
reshapes every row. Code written against that view breaks on the raw layout.

This script reproduces the loader's transformation exactly (`to_hf_row` mirrors
`ExeBench._fixes` + `ExeBench._generate_examples`) and appends the 16 ExeBench-FHD
columns, so code written against the HF ExeBench view runs unchanged:
  - `angha_{deps,io_pairs,exe_wrapper,iospec}` -> `synth_*`
  - `asm` dict -> Sequence({target, code}) with `code` = `func_asm` only
  - io_pairs `input`/`output` -> Sequence({var, value}) with `value` JSON-encoded
  - `real_iospec` / `synth_iospec` -> JSON strings
  - `doc`, `*_error`, `*_io_error`, `*_io_pairs_are_trivial`, `pre_asm`/`post_asm` dropped
The ExeBench-FHD columns have fixed inner keys (arch x opt level), so they are stored
as native typed structs rather than JSON strings. `objdump_asm`, `data_section` and
`func_call` and their error columns are all [arch][level]. In the JSONL,
`func_call_errors[arch]` is collapsed to a single value when every level has the same
(normalized) error; here it is expanded back to one value per level, so the column has
a fixed type and reads the same way as `func_call`.

Parquet is written with the `datasets` feature metadata embedded, so
`load_dataset("parquet", data_files=...)` yields exactly the HF feature types (no
loading script needed -- `datasets>=4` no longer runs them).

Large splits are split into byte-range chunks processed in parallel; each chunk
writes its own shards and a `.done` marker, so a rerun only redoes unfinished chunks.
Final shards are named `<split>-XXXXX-of-NNNNN.parquet` in input row order.

Usage
-----
    python -m scripts.release.convert_jsonl_to_hf_parquet \\
        --input-dir datasets_jsonl_v2 \\
        --output-dir datasets_hf_parquet \\
        [--files test_real.jsonl test_synth.jsonl ...] \\
        [--workers 32] [--chunk-gb 4] [--shard-mb 1000] [--batch-rows 500]
"""
import argparse
import glob
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import datasets
import pyarrow as pa
import pyarrow.parquet as pq

# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------

# Verbatim from jordiae/exebench `exebench.py` (_FEATURES, v1.0.4), in the same order.
_IO_PAIRS = datasets.Sequence({
    "input": datasets.Sequence({"var": datasets.Value("string"), "value": datasets.Value("string")}),
    "output": datasets.Sequence({"var": datasets.Value("string"), "value": datasets.Value("string")}),
    "dummy_funcs": datasets.Value("string"),
    "dummy_funcs_seed": datasets.Value("int64"),
})
HF_FEATURES = {
    "path": datasets.Value("string"),
    "func_def": datasets.Value("string"),
    "func_head": datasets.Value("string"),
    "func_head_types": datasets.Value("string"),
    "fname": datasets.Value("string"),
    "signature": datasets.Sequence(datasets.Value("string")),
    "asm": datasets.Sequence({"target": datasets.Value("string"), "code": datasets.Value("string")}),
    "synth_deps": datasets.Value("string"),
    "real_deps": datasets.Value("string"),
    "synth_io_pairs": _IO_PAIRS,
    "real_io_pairs": _IO_PAIRS,
    "synth_exe_wrapper": datasets.Value("string"),
    "real_exe_wrapper": datasets.Value("string"),
    "ref": datasets.Value("string"),
    "synth_iospec": datasets.Value("string"),
    "real_iospec": datasets.Value("string"),
}

# Same constants as the loader script.
_REMOVED_FEATURES = ["doc", "angha_error", "real_error", "angha_io_error", "real_io_error",
                     "angha_io_pairs_are_trivial", "real_io_pairs_are_trivial"]
_RENAMED_FEATURES = {"angha_deps": "synth_deps", "angha_io_pairs": "synth_io_pairs",
                     "angha_exe_wrapper": "synth_exe_wrapper", "angha_iospec": "synth_iospec"}

ARCHES = ["x86", "x86-64", "ARM32", "ARM64"]
OPT_LEVELS = ["O0", "O1", "O2", "O3", "Os"]

_PER_ARCH_OPT = {a: {o: datasets.Value("string") for o in OPT_LEVELS} for a in ARCHES}

EXEBENCH_FHD_FEATURES = {
    "objdump_asm": _PER_ARCH_OPT,
    "objdump_asm_errors": _PER_ARCH_OPT,
    "func_call": _PER_ARCH_OPT,
    "func_call_errors": _PER_ARCH_OPT,
    "func_call_skip_reason": datasets.Value("string"),
    "data_section": _PER_ARCH_OPT,
    "data_section_errors": _PER_ARCH_OPT,
    "opt_real_exe_wrapper": datasets.Value("string"),
    "opt_real_exe_wrapper_error": datasets.Value("string"),
    "opt_angha_exe_wrapper": datasets.Value("string"),
    "opt_angha_exe_wrapper_error": datasets.Value("string"),
    "trivial_function": datasets.Value("bool"),
    "no_return": datasets.Value("bool"),
    "unused_param": datasets.Value("bool"),
    "truncated_funcs": datasets.Value("bool"),
    "dummy_udt": datasets.Value("bool"),
}
_PER_ARCH_OPT_FIELDS = {"objdump_asm", "objdump_asm_errors", "func_call", "func_call_errors",
                        "data_section", "data_section_errors"}

FEATURES = datasets.Features({**HF_FEATURES, **EXEBENCH_FHD_FEATURES})


# --------------------------------------------------------------------------------------
# Row transformation
# --------------------------------------------------------------------------------------

def _check_keys(name, value, expected):
    if value is not None and set(value) != set(expected):
        raise ValueError(f"{name}: expected keys {sorted(expected)}, got {sorted(value)}")


def _expand_collapsed_levels(per_arch):
    """{arch: value | {level: value}} -> {arch: {level: value}}.

    A non-dict value is one error shared by every level (see
    `extract_func_call.collapse_identical`), so it is repeated for each level.
    """
    if per_arch is None:
        return None
    return {arch: (v if isinstance(v, dict) else dict.fromkeys(OPT_LEVELS, v))
            for arch, v in per_arch.items()}


def to_hf_row(row: dict) -> dict:
    """Map one ExeBench-FHD JSONL row to the HF ExeBench view plus the ExeBench-FHD columns.

    Mirrors ExeBench._fixes() then the io_pairs/iospec step of _generate_examples().
    Raises ValueError if an ExeBench-FHD struct field carries unexpected keys, rather than
    silently dropping data.
    """
    data = dict(row)

    # --- ExeBench._fixes ---
    if "angha_iospec" not in data:
        data["angha_iospec"] = None
    if "real_iospec" not in data:
        data["real_iospec"] = None
    if "func_head_types" not in data:
        data["func_head_types"] = ""
    data["asm"] = [{"target": target, "code": code["func_asm"] if code else None}
                   for (target, code) in data["asm"].items()]
    for removed_key in _REMOVED_FEATURES:
        data.pop(removed_key, None)
    for original_key, new_key in _RENAMED_FEATURES.items():
        data[new_key] = data.pop(original_key)

    # --- ExeBench._generate_examples ---
    for io_pairs_kind in ("synth_io_pairs", "real_io_pairs"):
        if data[io_pairs_kind]:
            new_io_pairs = []
            for e in data[io_pairs_kind]:
                new_io_pairs.append({
                    "input": [{"var": var, "value": json.dumps(value)} for (var, value) in e["input"].items()] if e["input"] else [],
                    "output": [{"var": var, "value": json.dumps(value)} for (var, value) in e["output"].items()] if e["output"] else [],
                    "dummy_funcs": e["dummy_funcs"],
                    "dummy_funcs_seed": e["dummy_funcs_seed"],
                })
            data[io_pairs_kind] = new_io_pairs
    data["synth_iospec"] = json.dumps(data["synth_iospec"])
    data["real_iospec"] = json.dumps(data["real_iospec"])

    # --- ExeBench-FHD columns ---
    data["func_call_errors"] = _expand_collapsed_levels(data.get("func_call_errors"))
    for name in _PER_ARCH_OPT_FIELDS:
        value = data.get(name)
        _check_keys(name, value, ARCHES)
        for arch in (value or {}):
            _check_keys(f"{name}[{arch}]", value[arch], OPT_LEVELS)

    unknown = set(data) - set(FEATURES)
    if unknown:
        raise ValueError(f"unmapped fields: {sorted(unknown)}")
    return {name: data.get(name) for name in FEATURES}


def rows_to_table(rows: list) -> pa.Table:
    """Encode HF-view rows into an Arrow table carrying the `datasets` feature metadata."""
    return datasets.Dataset.from_list(rows, features=FEATURES).data.table


# --------------------------------------------------------------------------------------
# Chunked conversion
# --------------------------------------------------------------------------------------

def chunk_ranges(path: str, chunk_bytes: int) -> list:
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


def convert_chunk(input_path, tmp_dir, split, chunk_idx, start, end,
                  shard_max_bytes, batch_rows):
    """Convert one byte range into tmp parquet parts; returns (chunk_idx, rows, parts)."""
    marker = os.path.join(tmp_dir, f"chunk{chunk_idx:05d}.done")
    if os.path.exists(marker):
        with open(marker) as f:
            info = json.load(f)
        return chunk_idx, info["rows"], info["parts"]

    parts, rows_done = [], 0
    writer, part_bytes = None, 0

    def write(table):
        nonlocal writer, part_bytes
        if writer is None or part_bytes >= shard_max_bytes:
            if writer is not None:
                writer.close()
            part = os.path.join(tmp_dir, f"chunk{chunk_idx:05d}-part{len(parts):05d}.parquet")
            parts.append(part)
            writer = pq.ParquetWriter(part, table.schema, compression="zstd")
            part_bytes = 0
        writer.write_table(table)
        part_bytes += table.nbytes

    batch = []
    with open(input_path, "rb") as f:
        f.seek(start)
        while f.tell() < end:
            line = f.readline()
            if not line.strip():
                continue
            batch.append(to_hf_row(json.loads(line)))
            if len(batch) >= batch_rows:
                write(rows_to_table(batch))
                rows_done += len(batch)
                batch = []
    if batch:
        write(rows_to_table(batch))
        rows_done += len(batch)
    if writer is not None:
        writer.close()

    with open(marker + ".tmp", "w") as f:
        json.dump({"rows": rows_done, "parts": parts, "start": start, "end": end}, f)
    os.replace(marker + ".tmp", marker)
    return chunk_idx, rows_done, parts


def finalize_split(output_dir, split, tmp_dir, results):
    """Rename chunk parts to `<split>-XXXXX-of-NNNNN.parquet` in input order."""
    parts = [p for _, _, ps in sorted(results) for p in ps]
    for stale in glob.glob(os.path.join(output_dir, f"{split}-*-of-*.parquet")):
        os.remove(stale)
    for i, part in enumerate(parts):
        os.replace(part, os.path.join(output_dir, f"{split}-{i:05d}-of-{len(parts):05d}.parquet"))
    for marker in glob.glob(os.path.join(tmp_dir, "*.done")):
        os.remove(marker)
    os.rmdir(tmp_dir)
    return len(parts), sum(r for _, r, _ in results)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default="datasets_jsonl_v2")
    ap.add_argument("--output-dir", default="datasets_hf_parquet")
    ap.add_argument("--files", nargs="*", help="JSONL basenames (default: all *.jsonl in --input-dir)")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--chunk-gb", type=float, default=4.0, help="input bytes per parallel chunk")
    ap.add_argument("--shard-mb", type=int, default=1000, help="max uncompressed MB per output shard")
    ap.add_argument("--batch-rows", type=int, default=500)
    args = ap.parse_args(argv)

    files = args.files or sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.input_dir, "*.jsonl")))
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
                fut = pool.submit(convert_chunk, input_path, tmp_dir, split, i, start, end,
                                  args.shard_mb * 2**20, args.batch_rows)
                futures[fut] = split
            print(f"[{split}] {len(ranges)} chunk(s) queued", flush=True)

        results = {split: [] for split in jobs}
        t0 = time.time()
        failed = False
        for fut in as_completed(futures):
            split = futures[fut]
            try:
                results[split].append(fut.result())
            except Exception as e:  # keep other chunks going; rerun resumes
                failed = True
                print(f"[{split}] chunk FAILED: {e!r}", file=sys.stderr, flush=True)
                continue
            done, total = len(results[split]), jobs[split][1]
            print(f"[{split}] chunk {done}/{total} done ({time.time() - t0:.0f}s)", flush=True)
            if done == total:
                n_shards, n_rows = finalize_split(args.output_dir, split, jobs[split][0], results[split])
                print(f"[{split}] DONE: {n_rows} rows in {n_shards} shard(s)", flush=True)

    if failed:
        sys.exit("some chunks failed; fix and rerun (completed chunks are reused)")


if __name__ == "__main__":
    main()
