import json

from scripts.extract import extract_main_named_functions as emnf


def test_cli_keeps_only_main_named_records_unmodified(patch_iter_split, run_main, tmp_path, record_factory):
    main_rec = record_factory(fname="main", func_def="int main(void){return 0;}", path="o/r/a.c")
    other_rec = record_factory(fname="helper", func_def="int helper(void){return 1;}", path="o/r/b.c")
    patch_iter_split(emnf, [main_rec, other_rec])

    out_path = tmp_path / "out.jsonl"
    run_main(emnf, ["--out", str(out_path)])

    lines = out_path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == main_rec


def test_cli_missing_fname_is_not_main(patch_iter_split, run_main, tmp_path, record_factory):
    rec = record_factory(fname=None, path="o/r/c.c")
    patch_iter_split(emnf, [rec])

    out_path = tmp_path / "out.jsonl"
    run_main(emnf, ["--out", str(out_path)])

    assert out_path.read_text() == ""


def test_cli_no_matches_writes_empty_file(patch_iter_split, run_main, tmp_path, record_factory):
    patch_iter_split(emnf, [record_factory(fname="helper")])
    out_path = tmp_path / "out.jsonl"
    run_main(emnf, ["--out", str(out_path)])
    assert out_path.read_text() == ""


def test_cli_reports_summary_on_stdout(capsys, patch_iter_split, run_main, tmp_path, record_factory):
    patch_iter_split(emnf, [record_factory(fname="main"), record_factory(fname="helper")])
    out_path = tmp_path / "out.jsonl"
    run_main(emnf, ["--out", str(out_path)])

    summary = json.loads(capsys.readouterr().out)
    assert summary["scanned"] == 2
    assert summary["matched"] == 1
    assert summary["pct"] == 50.0
