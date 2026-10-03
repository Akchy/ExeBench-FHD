import json

from scripts.survey import survey_func_call_param_types as sfc


def test_cli_writes_survey_report(monkeypatch, tmp_path, patch_iter_split, run_main):
    records = [
        {"signature": ["int", "int", "char *"]},
        {"signature": ["int", "struct foo"]},
    ]
    patch_iter_split(sfc, records)
    out_path = tmp_path / "survey.json"

    run_main(sfc, ["--split", "test_real", "--out", str(out_path)])

    report = json.loads(out_path.read_text())
    assert report["records_scanned"] == 2
    assert report["param_type_categories"]["struct_or_union_by_value"] == 1
    assert report["param_type_categories"]["int_family"] == 1
