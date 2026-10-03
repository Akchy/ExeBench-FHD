import json

from scripts.extract import extract_trivial_function as etf


def test_cli_writes_only_matching_records(tmp_path, patch_iter_split, run_main, record_factory):
    matching = record_factory(fname="bad", func_def="int bad(void){return 0;}", path="o/r/a.c")
    non_matching = record_factory(fname="good", func_def=(
        "int good(void){\n"
        "    if (cond()) {\n"
        "        return 1;\n"
        "    }\n"
        "    return 0;\n"
        "}\n"
    ), path="o/r/b.c")
    patch_iter_split(etf, [matching, non_matching])

    out_path = tmp_path / "out.jsonl"
    run_main(etf, ["--out", str(out_path), "--n", "10"])

    lines = out_path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["fname"] == "bad"


def test_cli_stops_once_n_matches_collected(tmp_path, patch_iter_split, run_main, record_factory):
    recs = [record_factory(fname=f"f{i}", func_def=f"int f{i}(void){{return {i};}}", path=f"o/r/{i}.c")
            for i in range(5)]
    patch_iter_split(etf, recs)

    out_path = tmp_path / "out.jsonl"
    run_main(etf, ["--out", str(out_path), "--n", "2"])

    assert len(out_path.read_text().splitlines()) == 2


def test_cli_all_writes_every_record_with_trivial_function_field(tmp_path, patch_iter_split, run_main,
                                                                   record_factory):
    matching = record_factory(fname="bad", func_def="int bad(void){return 0;}", path="o/r/a.c")
    non_matching = record_factory(fname="good", func_def=(
        "int good(void){\n"
        "    if (cond()) {\n"
        "        return 1;\n"
        "    }\n"
        "    return 0;\n"
        "}\n"
    ), path="o/r/b.c")
    patch_iter_split(etf, [matching, non_matching])

    out_path = tmp_path / "out.jsonl"
    run_main(etf, ["--out", str(out_path), "--all"])

    lines = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(lines) == 2
    assert {(l["fname"], l["trivial_function"]) for l in lines} == {("bad", True), ("good", False)}


def test_cli_split_all_writes_one_file_per_split(tmp_path, patch_iter_split, run_main, record_factory):
    rec = record_factory(fname="bad", func_def="int bad(void){return 0;}", path="o/r/a.c")
    patch_iter_split(etf, [rec])

    out_dir = tmp_path / "trivial_function_field"
    run_main(etf, ["--split", "all", "--all", "--out-dir", str(out_dir)])

    from scripts.check_split_leakage import ALL_SPLITS
    for split in ALL_SPLITS:
        split_path = out_dir / f"{split}.jsonl"
        assert split_path.exists()
        assert json.loads(split_path.read_text().splitlines()[0])["trivial_function"] is True
