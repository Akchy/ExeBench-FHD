import json

from scripts.extract import extract_zero_branch_triviality as ezbt


def _rec(fname, func_def):
    return {"fname": fname, "func_def": func_def, "path": f"o/r/{fname}.c"}


RECORDS = [
    _rec("empty1", "int empty1(void) {\n}\n"),
    _rec("ret1", "int ret1(void) {\n    return 0;\n}\n"),
    _rec("two1", "int two1(int x) {\n    x++;\n    return x;\n}\n"),
    _rec("gap1", "int gap1(int x) {\n    x = x + 1;\n    x = x * 2;\n    x = x - 3;\n    return x;\n}\n"),
    _rec("branchy", "int branchy(int x) {\n    if (x > 0) {\n        x = x + 1;\n    }\n    x = x * 2;\n    return x;\n}\n"),
]


def test_cli_writes_one_file_per_category(tmp_path, patch_iter_split, run_main):
    patch_iter_split(ezbt, RECORDS)
    prefix = str(tmp_path / "sample")
    run_main(ezbt, ["--out-prefix", prefix, "--n", "10"])

    for cat, fname in [("empty", "empty1"), ("return_only", "ret1"), ("two_loc", "two1"), ("gap", "gap1")]:
        lines = (tmp_path / f"sample_{cat}.jsonl").read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["fname"] == fname

    # branchy has a long body with a branch - matches none of the 4 categories.
    for cat in ezbt.CATEGORIES:
        recs = [json.loads(l)["fname"] for l in (tmp_path / f"sample_{cat}.jsonl").read_text().splitlines()]
        assert "branchy" not in recs


def test_cli_per_category_n_cap(tmp_path, patch_iter_split, run_main):
    recs = [_rec(f"e{i}", "int f(void) {\n}\n") for i in range(5)]
    patch_iter_split(ezbt, recs)
    prefix = str(tmp_path / "sample")
    run_main(ezbt, ["--out-prefix", prefix, "--n", "2"])

    assert len((tmp_path / "sample_empty.jsonl").read_text().splitlines()) == 2
