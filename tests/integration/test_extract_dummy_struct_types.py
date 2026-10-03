import json

from scripts.extract import extract_dummy_struct_types as edst


def test_cli_writes_only_matching_records(monkeypatch, tmp_path, patch_iter_split, run_main, record_factory):
    matching = record_factory(fname="uses_dummy", angha_deps="struct sockaddr { int dummy; };", real_deps="",
                               path="o/r/a.c")
    non_matching = record_factory(fname="normal", angha_deps="struct point { int x; int y; };", real_deps="",
                                   path="o/r/b.c")
    patch_iter_split(edst, [matching, non_matching])

    out_path = tmp_path / "out.jsonl"
    run_main(edst, ["--out", str(out_path), "--n", "10"])

    lines = out_path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["fname"] == "uses_dummy"


def test_cli_stops_once_n_matches_collected(tmp_path, patch_iter_split, run_main, record_factory):
    recs = [record_factory(fname=f"f{i}", angha_deps="struct s { int dummy; };", real_deps="", path=f"o/r/{i}.c")
            for i in range(5)]
    patch_iter_split(edst, recs)

    out_path = tmp_path / "out.jsonl"
    run_main(edst, ["--out", str(out_path), "--n", "2"])

    assert len(out_path.read_text().splitlines()) == 2


def test_cli_all_writes_every_record_with_dummy_udt_field(tmp_path, patch_iter_split, run_main, record_factory):
    matching = record_factory(fname="bad", angha_deps="struct sockaddr { int dummy; };", real_deps="",
                               path="o/r/a.c")
    non_matching = record_factory(fname="good", angha_deps="struct point { int x; int y; };", real_deps="",
                                   path="o/r/b.c")
    patch_iter_split(edst, [matching, non_matching])

    out_path = tmp_path / "out.jsonl"
    run_main(edst, ["--out", str(out_path), "--all"])

    lines = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(lines) == 2
    assert {(l["fname"], l["dummy_udt"]) for l in lines} == {("bad", True), ("good", False)}


def test_cli_split_all_writes_one_file_per_split(tmp_path, patch_iter_split, run_main, record_factory):
    rec = record_factory(fname="bad", angha_deps="struct sockaddr { int dummy; };", real_deps="", path="o/r/a.c")
    patch_iter_split(edst, [rec])

    out_dir = tmp_path / "dummy_udt_field"
    run_main(edst, ["--split", "all", "--all", "--out-dir", str(out_dir)])

    from scripts.check_split_leakage import ALL_SPLITS
    for split in ALL_SPLITS:
        split_path = out_dir / f"{split}.jsonl"
        assert split_path.exists()
        assert json.loads(split_path.read_text().splitlines()[0])["dummy_udt"] is True
