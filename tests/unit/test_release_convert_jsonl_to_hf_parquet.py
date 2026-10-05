import json

import datasets
import pytest

from scripts.release import convert_jsonl_to_hf_parquet as conv

ARCH_ERR = {a: None for a in conv.ARCHES}
ARCH_OPT = {a: {o: f"<{a}-{o}>" for o in conv.OPT_LEVELS} for a in conv.ARCHES}
ARCH_OPT_EMPTY = {a: dict.fromkeys(conv.OPT_LEVELS, "") for a in conv.ARCHES}
ARCH_OPT_NONE = {a: dict.fromkeys(conv.OPT_LEVELS) for a in conv.ARCHES}


def make_row(**overrides):
    row = {
        "path": "a/b.c", "fname": "f", "func_def": "int f(int x){return x;}",
        "func_head": "int f(intx)", "func_head_types": "int f(int x)", "signature": ["int", "int"],
        "ref": "master", "doc": None,
        "asm": {
            "real_gcc_x86_O0": {"pre_asm": "PRE", "func_asm": "FUNC0", "post_asm": "POST", "target": "x"},
            "real_gcc_arm_O0": None,
        },
        "angha_deps": None, "angha_error": "", "angha_exe_wrapper": None, "angha_io_error": None,
        "angha_io_pairs": None, "angha_io_pairs_are_trivial": True, "angha_iospec": None,
        "real_deps": "#include <x>", "real_error": None, "real_exe_wrapper": "W", "real_io_error": None,
        "real_io_pairs": [
            {"input": {"x": 1.5, "p": [1, 2]}, "output": {"returnv": 1.5}, "dummy_funcs": "", "dummy_funcs_seed": 7},
            {"input": {}, "output": None, "dummy_funcs": None, "dummy_funcs_seed": None},
        ],
        "real_io_pairs_are_trivial": False,
        "real_iospec": {"livein": ["x"], "liveout": []},
        "objdump_asm": ARCH_OPT, "objdump_asm_errors": {a: dict.fromkeys(conv.OPT_LEVELS) for a in conv.ARCHES},
        "func_call": ARCH_OPT_EMPTY, "func_call_errors": dict(ARCH_ERR), "func_call_skip_reason": None,
        "data_section": ARCH_OPT_EMPTY, "data_section_errors": ARCH_OPT_NONE,
        "opt_real_exe_wrapper": None, "opt_real_exe_wrapper_error": None,
        "opt_angha_exe_wrapper": None, "opt_angha_exe_wrapper_error": None,
        "trivial_function": False, "no_return": False, "unused_param": False,
        "truncated_funcs": False, "dummy_udt": False,
    }
    row.update(overrides)
    return row


def test_to_hf_row_renames_drops_and_reshapes_like_hf_loader():
    out = conv.to_hf_row(make_row())
    assert list(out) == list(conv.FEATURES)
    for gone in ["doc", "angha_error", "real_error", "angha_io_pairs", "real_io_pairs_are_trivial"]:
        assert gone not in out
    assert out["asm"] == [{"target": "real_gcc_x86_O0", "code": "FUNC0"},
                          {"target": "real_gcc_arm_O0", "code": None}]
    assert out["real_io_pairs"][0]["input"] == [{"var": "x", "value": "1.5"}, {"var": "p", "value": "[1, 2]"}]
    assert out["real_io_pairs"][1]["input"] == [] and out["real_io_pairs"][1]["output"] == []
    assert out["real_iospec"] == json.dumps({"livein": ["x"], "liveout": []})
    assert out["synth_iospec"] == "null"
    assert out["synth_io_pairs"] is None
    assert out["objdump_asm"] == ARCH_OPT


def test_to_hf_row_fills_missing_func_head_types_and_iospec():
    row = make_row()
    del row["func_head_types"], row["real_iospec"]
    out = conv.to_hf_row(row)
    assert out["func_head_types"] == ""
    assert out["real_iospec"] == "null"


def test_to_hf_row_expands_collapsed_func_call_errors():
    per_level = {"O0": None, "O1": "e1", "O2": "e1", "O3": "e1", "Os": "e1"}
    errors = {"x86": None, "x86-64": "link failed", "ARM32": per_level, "ARM64": None}
    out = conv.to_hf_row(make_row(func_call_errors=errors))
    assert out["func_call_errors"]["x86"] == dict.fromkeys(conv.OPT_LEVELS)
    assert out["func_call_errors"]["x86-64"] == dict.fromkeys(conv.OPT_LEVELS, "link failed")
    assert out["func_call_errors"]["ARM32"] == per_level
    assert out["func_call"] == ARCH_OPT_EMPTY


def test_to_hf_row_keeps_skipped_func_call_null():
    out = conv.to_hf_row(make_row(func_call=None, func_call_errors=None,
                                  func_call_skip_reason="fname_is_main"))
    assert out["func_call"] is None and out["func_call_errors"] is None


@pytest.mark.parametrize("field,value", [
    ("data_section", {"x86": "", "mips": ""}),
    ("data_section", {a: "" for a in conv.ARCHES}),
    ("objdump_asm", {a: {"O0": ""} for a in conv.ARCHES}),
])
def test_to_hf_row_rejects_unexpected_struct_keys(field, value):
    with pytest.raises(ValueError, match=field):
        conv.to_hf_row(make_row(**{field: value}))


def test_to_hf_row_rejects_unmapped_fields():
    with pytest.raises(ValueError, match="unmapped"):
        conv.to_hf_row(make_row(new_field=1))


def test_rows_to_table_uses_hf_sequence_encoding():
    ds = datasets.Dataset(conv.rows_to_table([conv.to_hf_row(make_row())]))
    r = ds[0]
    # HF stores Sequence({...}) as a dict of lists, exactly as load_dataset('jordiae/exebench') returns it.
    assert r["asm"] == {"target": ["real_gcc_x86_O0", "real_gcc_arm_O0"], "code": ["FUNC0", None]}
    assert r["real_io_pairs"]["dummy_funcs_seed"] == [7, None]
    assert ds.features == conv.FEATURES


def test_main_writes_ordered_shards_and_resumes(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    rows = [make_row(fname=f"f{i}") for i in range(30)]
    (in_dir / "test_real.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    args = ["--input-dir", str(in_dir), "--output-dir", str(out_dir), "--workers", "2",
            "--chunk-gb", str(4000 / 2**30), "--batch-rows", "4"]
    conv.main(args)
    shards = sorted(out_dir.glob("test_real-*.parquet"))
    assert len(shards) > 1 and shards[0].name.endswith(f"-of-{len(shards):05d}.parquet")
    ds = datasets.load_dataset("parquet", data_files=[str(s) for s in shards], split="train")
    assert ds["fname"] == [f"f{i}" for i in range(30)]
    assert not (out_dir / ".tmp_test_real").exists()

    conv.main(args)  # rerun replaces shards rather than duplicating them
    assert sorted(out_dir.glob("test_real-*.parquet")) == shards
