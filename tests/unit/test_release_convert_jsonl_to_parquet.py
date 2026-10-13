import json
import math
import os

import pyarrow.parquet as pq
import pytest

from scripts.release import convert_jsonl_to_parquet as cv


def make_full_record(i=0, **overrides):
    """A 39-field record covering the shapes in the real data (mixed func_call_errors, nulls, non-ASCII)."""
    levels = {lv: f"asm {i} {lv}" for lv in ("O0", "O1", "O2", "O3", "Os")}
    rec = {f: None for f in cv.FIELDS}
    rec.update({
        "path": f"owner/repo/file{i}.c", "ref": "master", "fname": f"f{i}",
        "func_def": f"int f{i}(int x) {{ return x; }} // naïve", "func_head": f"int f{i}(int x)",
        "func_head_types": f"int f{i}(int x)", "signature": ["int", "int"],
        "real_deps": "", "angha_deps": None, "angha_error": "err",
        "asm": {"real_gcc_x86_O0": {"target": "real_gcc_x86_O0", "pre_asm": "", "func_asm": "ret", "post_asm": ""}},
        "real_io_pairs": [{"input": {"x": 1, "y": [1.5, -2.25]}, "output": {"returnv": 1},
                           "dummy_funcs": None, "dummy_funcs_seed": None}],
        "real_iospec": {"livein": ["x"], "needs_cast": {"x": False}},
        "objdump_asm": {"x86-64": levels, "ARM64": dict(levels)},
        "objdump_asm_errors": {"x86-64": {lv: None for lv in levels}, "ARM64": {lv: None for lv in levels}},
        "func_call": {"x86-64": levels, "x86": "legacy single -O3 snippet"},
        "func_call_errors": {"x86-64": None, "ARM32": "compile_failed: boom", "x86": {"O0": None, "O2": "e"}},
        "data_section": {"x86-64": {"O2": ".rodata @0x2000 (4 bytes)"}},
        "data_section_errors": {"x86-64": {"O2": None}},
        "real_io_pairs_are_trivial": False, "angha_io_pairs_are_trivial": True,
        "trivial_function": i % 2 == 0, "no_return": False, "unused_param": True,
        "truncated_funcs": False, "dummy_udt": False,
        "opt_real_exe_wrapper": "// wrapper", "opt_real_exe_wrapper_error": None,
    })
    rec.update(overrides)
    return rec


def write_jsonl(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_split(out_dir, split):
    shards = sorted((out_dir / "data").glob(f"{split}-*.parquet"))
    return [cv.decode_row(row) for s in shards for row in pq.read_table(s).to_pylist()]


def test_schema_has_39_fields_in_exebench_then_exebench_fhd_order():
    assert len(cv.EXEBENCH_FIELDS) == 23 and len(cv.EXEBENCH_FHD_FIELDS) == 16
    assert len(set(cv.FIELDS)) == 39
    assert cv.SCHEMA.names == cv.FIELDS
    assert set(cv.JSON_COLUMNS) | set(cv.BOOL_COLUMNS) | set(cv.LIST_COLUMNS) <= set(cv.FIELDS)
    assert json.loads(cv.SCHEMA.metadata[b"exebench_fhd"])["json_columns"] == cv.JSON_COLUMNS


def test_encode_decode_round_trip():
    rec = make_full_record()
    assert cv.decode_row(cv.encode_row(rec)) == rec


def test_encoded_nested_fields_are_json_strings_and_flags_stay_native():
    row = cv.encode_row(make_full_record())
    assert isinstance(row["func_call"], str) and json.loads(row["func_call"])["x86"] == "legacy single -O3 snippet"
    assert row["angha_iospec"] is None  # null stays null, not "null"
    assert row["unused_param"] is True and row["signature"] == ["int", "int"]


def test_encode_rejects_missing_or_extra_fields():
    rec = make_full_record()
    del rec["dummy_udt"]
    with pytest.raises(ValueError, match="missing=\\['dummy_udt'\\]"):
        cv.encode_row(rec)
    with pytest.raises(ValueError, match="extra=\\['surprise'\\]"):
        cv.encode_row(make_full_record(surprise=1))


def test_table_round_trips_through_parquet(tmp_path):
    recs = [make_full_record(i) for i in range(3)]
    pq.write_table(cv.rows_to_table([cv.encode_row(r) for r in recs]), tmp_path / "t.parquet")
    assert [cv.decode_row(r) for r in pq.read_table(tmp_path / "t.parquet").to_pylist()] == recs


def test_chunk_ranges_cover_file_on_line_boundaries(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [make_full_record(i) for i in range(20)])
    ranges = cv.chunk_ranges(str(p), 3000)
    assert len(ranges) > 1
    assert ranges[0][0] == 0 and ranges[-1][1] == os.path.getsize(p)
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    data = p.read_bytes()
    assert all(data[start - 1:start] == b"\n" for start, _ in ranges[1:])


def test_main_converts_multi_chunk_split_in_order(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    recs = [make_full_record(i) for i in range(30)]
    write_jsonl(src / "test_real.jsonl", recs)
    write_jsonl(src / "valid_real.jsonl", recs[:2])
    cv.main(["--input-dir", str(src), "--output-dir", str(out), "--workers", "2",
             "--chunk-gb", str(5000 / 2**30), "--batch-rows", "4"])

    shards = sorted(p.name for p in (out / "data").glob("test_real-*.parquet"))
    assert len(shards) > 1 and shards[0] == f"test_real-00000-of-{len(shards):05d}.parquet"
    assert read_split(out, "test_real") == recs
    assert read_split(out, "valid_real") == recs[:2]
    stats = json.loads((out / "splits.json").read_text())
    assert stats["test_real"]["rows"] == 30 and stats["test_real"]["shards"] == len(shards)
    assert stats["valid_real"]["rows"] == 2
    assert not list(out.glob(".tmp_*"))


def test_verify_accepts_source_with_different_key_order(tmp_path):
    # The real JSONL orders the ExeBench-FHD fields differently from FIELDS.
    rec = make_full_record()
    shuffled = {k: rec[k] for k in reversed(list(rec))}
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    write_jsonl(src / "test_real.jsonl", [shuffled])
    cv.main(["--input-dir", str(src), "--output-dir", str(out), "--workers", "1"])
    assert read_split(out, "test_real") == [rec]


def test_nan_and_bool_types_survive(tmp_path):
    rec = make_full_record(real_io_pairs=[{"input": {"x": math.nan}, "output": {"r": True}}])
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    write_jsonl(src / "test_real.jsonl", [rec])
    cv.main(["--input-dir", str(src), "--output-dir", str(out), "--workers", "1"])
    back = read_split(out, "test_real")[0]
    assert math.isnan(back["real_io_pairs"][0]["input"]["x"]) and back["real_io_pairs"][0]["output"]["r"] is True


def test_convert_chunk_reuses_done_marker(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [make_full_record(i) for i in range(3)])
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    first = cv.convert_chunk(str(p), str(tmp), 0, 0, os.path.getsize(p), batch_rows=2)
    mtime = os.path.getmtime(first[2])
    p.write_text("")  # a rerun must not re-read the input
    assert cv.convert_chunk(str(p), str(tmp), 0, 0, 10**6, batch_rows=2) == first
    assert os.path.getmtime(first[2]) == mtime


def test_verify_shard_detects_mismatch(tmp_path):
    p = tmp_path / "s.jsonl"
    recs = [make_full_record(i) for i in range(3)]
    write_jsonl(p, recs)
    shard = tmp_path / "x.parquet"
    pq.write_table(cv.rows_to_table([cv.encode_row(r) for r in recs]), shard)
    assert cv.verify_shard(str(shard), str(p), 0, os.path.getsize(p)) == 3

    pq.write_table(cv.rows_to_table([cv.encode_row(r) for r in recs[:2]]), shard)
    with pytest.raises(ValueError, match="fewer rows"):
        cv.verify_shard(str(shard), str(p), 0, os.path.getsize(p))

    changed = [recs[0], make_full_record(1, unused_param=False), recs[2]]
    pq.write_table(cv.rows_to_table([cv.encode_row(r) for r in changed]), shard)
    with pytest.raises(ValueError, match="row 1 does not round-trip"):
        cv.verify_shard(str(shard), str(p), 0, os.path.getsize(p))


def test_main_rejects_invalid_split_name(tmp_path):
    (tmp_path / "bad-name.jsonl").write_text("")
    with pytest.raises(SystemExit, match="valid split name"):
        cv.main(["--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "o"), "--files", "bad-name.jsonl"])
