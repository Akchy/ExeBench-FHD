import json

from scripts.extract import extract_truncated_funcs as etf

RECORDS = [
    {"fname": "sUalrm", "func_head": "void sUalrm()", "func_def": " sUalrm(){\n    exit(0);\n}\n"},
    {"fname": "main", "func_head": "int main()", "func_def": "main()\n{\n    return 0;\n}\n"},
    {"fname": "add", "func_head": "int add(int a, int b)", "func_def": "int add(int a, int b){\n    return a+b;\n}\n"},
]


def test_cli_writes_only_truncated_records(tmp_path, patch_iter_split, run_main):
    patch_iter_split(etf, RECORDS)
    out_path = tmp_path / "out.jsonl"
    run_main(etf, ["--out", str(out_path), "--n", "10"])

    lines = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert [r["fname"] for r in lines] == ["sUalrm"]
