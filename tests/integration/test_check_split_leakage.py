import json

from scripts import check_split_leakage as csl

SPLIT_RECORDS = {
    "test_real": [
        {"path": "o/shared/a.c", "func_def": "int f(void){return 0;}", "fname": "f"},
        {"path": "o/onlytest/b.c", "func_def": "int only_test(void){return 9;}", "fname": "only_test"},
    ],
    "train_real_compilable": [
        {"path": "o/shared/z.c", "func_def": "int  f(void){return  0;}", "fname": "f"},  # normalized dup of test's f (extra spaces only), same repo, same fname
        {"path": "o/othertrain/y.c", "func_def": "int h(void){return 1;}", "fname": "h"},
    ],
}


def _patch_multi_split(monkeypatch, module):
    def fake_iter_split(split, snapshot, limit=None):
        data = SPLIT_RECORDS.get(split, [])
        if limit is not None:
            data = data[:limit]
        yield from data
    monkeypatch.setattr(module, "iter_split", fake_iter_split)


def test_main_writes_report_with_expected_overlap(monkeypatch, tmp_path, run_main):
    _patch_multi_split(monkeypatch, csl)
    out_path = tmp_path / "report.json"

    run_main(csl, ["--splits", "test_real,train_real_compilable", "--out", str(out_path)])

    report = json.loads(out_path.read_text())
    assert report["splits"] == ["test_real", "train_real_compilable"]
    assert report["per_split"]["test_real"]["n"] == 2
    assert report["per_split"]["train_real_compilable"]["n"] == 2

    [pw] = report["pairwise"]
    assert pw["a"] == "test_real" and pw["b"] == "train_real_compilable"
    assert pw["exact_dup"] == 0  # bodies differ byte-for-byte (extra spaces)
    assert pw["normalized_dup"] == 1  # but match once whitespace-normalized
    assert pw["repo_overlap"] == 1  # both touch o/shared
    assert pw["fname_overlap"] == 1  # both define f

    [headline] = report["headline_eval_vs_train_union"]
    assert headline["a"] == "test_real"
    assert headline["train_splits"] == ["train_real_compilable"]
    assert headline["normalized_dup"] == 1


def test_main_rejects_unknown_split(monkeypatch, run_main):
    _patch_multi_split(monkeypatch, csl)
    try:
        run_main(csl, ["--splits", "not_a_real_split"])
        assert False, "expected SystemExit from argparse error()"
    except SystemExit as e:
        assert e.code != 0
