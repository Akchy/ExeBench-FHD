import json

from scripts.extract import extract_missing_return_type as emrt

RECORDS = [
    {"fname": "main", "func_head": "int main()", "func_def": "main()\n{\n    return 0;\n}\n"},
    {"fname": "sUalrm", "func_head": "void sUalrm()", "func_def": " sUalrm(){\n    exit(0);\n}\n"},
    {"fname": "format_sector", "func_head": "unsigned int format_sector(int cyl)",
     "func_def": "rmat_sector(unsigned int cyl){\n    return 0;\n}\n"},
    {"fname": "add", "func_head": "int add(int a, int b)", "func_def": "int add(int a, int b){\n    return a+b;\n}\n"},
]


def test_cli_routes_each_category_to_its_own_file(tmp_path, patch_iter_split, run_main):
    patch_iter_split(emrt, RECORDS)

    out_non_int = tmp_path / "non_int.jsonl"
    out_kr = tmp_path / "kr.jsonl"
    out_fname = tmp_path / "fname.jsonl"
    summary_out = tmp_path / "summary.json"

    run_main(emrt, [
        "--out-non-int", str(out_non_int), "--out-kr", str(out_kr), "--out-fname", str(out_fname),
        "--summary-out", str(summary_out),
    ])

    non_int_recs = [json.loads(l) for l in out_non_int.read_text().splitlines()]
    kr_recs = [json.loads(l) for l in out_kr.read_text().splitlines()]
    fname_recs = [json.loads(l) for l in out_fname.read_text().splitlines()]

    assert [r["fname"] for r in non_int_recs] == ["sUalrm"]
    assert [r["fname"] for r in kr_recs] == ["main"]
    assert [r["fname"] for r in fname_recs] == ["format_sector"]

    summary = json.loads(summary_out.read_text())
    assert summary["total_scanned"] == 4
    assert summary["global"]["missing_int_implicit"] == 1
    assert summary["global"]["missing_non_int"] == 1
    assert summary["global"]["fname_not_found"] == 1
    assert summary["missing_non_int_types_global"] == {"void": 1}


def test_cli_kr_cap_stops_writing_past_cap(tmp_path, patch_iter_split, run_main):
    kr_records = [
        {"fname": "main", "func_head": "int main()", "func_def": "main()\n{\n    return 0;\n}\n"}
        for _ in range(5)
    ]
    patch_iter_split(emrt, kr_records)

    out_non_int = tmp_path / "non_int.jsonl"
    out_kr = tmp_path / "kr.jsonl"
    out_fname = tmp_path / "fname.jsonl"
    summary_out = tmp_path / "summary.json"

    run_main(emrt, [
        "--out-non-int", str(out_non_int), "--out-kr", str(out_kr), "--out-fname", str(out_fname),
        "--summary-out", str(summary_out), "--kr-cap", "2",
    ])

    assert len(out_kr.read_text().splitlines()) == 2
    summary = json.loads(summary_out.read_text())
    assert summary["samples_written"]["kr_implicit_int"]["n"] == 2
    assert summary["samples_written"]["kr_implicit_int"]["capped"] is True
    assert summary["global"]["missing_int_implicit"] == 5  # counted even past the write cap
