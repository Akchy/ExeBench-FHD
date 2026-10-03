from scripts import check_split_leakage as csl


def test_digest_deterministic_and_distinct():
    a = csl._digest("int foo(void){return 0;}")
    b = csl._digest("int foo(void){return 0;}")
    c = csl._digest("int bar(void){return 1;}")
    assert a == b
    assert a != c
    assert len(a) == 16


def test_normalize_body_collapses_whitespace():
    assert csl._normalize_body("int   foo(void)\n{\n  return 0;\n}") == "int foo(void) { return 0; }"
    assert csl._normalize_body("  a  b  ") == "a b"
    assert csl._normalize_body("") == ""


def test_normalize_body_makes_reformatted_code_match():
    a = "int foo(void) {\n    return 0;\n}\n"
    b = "int foo(void)\n{\nreturn 0;\n}"
    assert csl._digest(csl._normalize_body(a)) == csl._digest(csl._normalize_body(b))


def test_repo_of_normal_path():
    assert csl._repo_of("owner/repo/src/file.c") == "owner/repo"


def test_repo_of_edge_cases():
    assert csl._repo_of(None) is None
    assert csl._repo_of("") is None
    assert csl._repo_of("justonesegment") is None
    assert csl._repo_of("owner/repo") == "owner/repo"


def _index(name, n, exact=(), norm=(), repos=(), fnames=()):
    return {
        "split": name, "n": n,
        "exact_hashes": set(exact), "norm_hashes": set(norm),
        "repos": set(repos), "fnames": set(fnames), "repo_examples": {},
    }


def test_pair_overlap_counts_and_percentages():
    a = _index("a", 10, exact={1, 2, 3}, norm={1, 2, 3, 4}, repos={"o/r1", "o/r2"}, fnames={"f1", "f2"})
    b = _index("b", 5, exact={2, 3, 99}, norm={4}, repos={"o/r2"}, fnames={"f2", "f3"})

    result = csl.pair_overlap(a, b)

    assert result["exact_dup"] == 2
    assert result["exact_dup_pct_of_a"] == round(100 * 2 / 3, 2)
    assert result["normalized_dup"] == 1
    assert result["repo_overlap"] == 1
    assert result["repo_overlap_pct_of_a"] == 50.0
    assert result["repo_overlap_pct_of_b"] == 100.0
    assert result["fname_overlap"] == 1
    assert result["example_repos"] == ["o/r2"]


def test_pair_overlap_no_overlap():
    a = _index("a", 3, exact={1}, repos={"o/r1"})
    b = _index("b", 3, exact={2}, repos={"o/r2"})
    result = csl.pair_overlap(a, b)
    assert result["exact_dup"] == 0
    assert result["repo_overlap"] == 0
    assert result["example_repos"] == []


def test_pair_overlap_empty_sets_pct_is_zero_not_division_error():
    a = _index("a", 0)
    b = _index("b", 0)
    result = csl.pair_overlap(a, b)
    assert result["exact_dup_pct_of_a"] == 0.0
    assert result["repo_overlap_pct_of_a"] == 0.0
    assert result["repo_overlap_pct_of_b"] == 0.0


def test_union_index_merges_across_splits():
    a = _index("a", 4, exact={1, 2}, norm={1}, repos={"o/r1"}, fnames={"f1"})
    b = _index("b", 6, exact={2, 3}, norm={2}, repos={"o/r2"}, fnames={"f2"})
    u = csl.union_index({"a": a, "b": b}, ["a", "b"])
    assert u["split"] == "a+b"
    assert u["n"] == 10
    assert u["exact_hashes"] == {1, 2, 3}
    assert u["norm_hashes"] == {1, 2}
    assert u["repos"] == {"o/r1", "o/r2"}
    assert u["fnames"] == {"f1", "f2"}


def test_union_index_single_split_is_identity():
    a = _index("a", 4, exact={1, 2}, repos={"o/r1"})
    u = csl.union_index({"a": a}, ["a"])
    assert u["n"] == 4
    assert u["exact_hashes"] == {1, 2}


def test_build_index_aggregates_records(monkeypatch):
    records = [
        {"path": "o/r1/a.c", "func_def": "int f(void){return 0;}", "fname": "f"},
        {"path": "o/r1/b.c", "func_def": "int f(void){return 0;}", "fname": "f"},  # exact dup of above
        {"path": "o/r2/c.c", "func_def": "int g(void){return 1;}", "fname": "g"},
        {"path": None, "func_def": "", "fname": ""},  # missing fields shouldn't crash or count
    ]

    def fake_iter_split(split, snapshot, limit=None):
        data = records if limit is None else records[:limit]
        yield from data

    monkeypatch.setattr(csl, "iter_split", fake_iter_split)

    idx = csl.build_index("test_real", "/unused/snapshot")

    assert idx["split"] == "test_real"
    assert idx["n"] == 4
    assert len(idx["exact_hashes"]) == 2  # two distinct bodies, one duplicated
    assert idx["repos"] == {"o/r1", "o/r2"}
    assert idx["fnames"] == {"f", "g"}


def test_build_index_respects_limit(monkeypatch):
    records = [{"path": f"o/r{i}/x.c", "func_def": f"int f{i}(void){{}}", "fname": f"f{i}"} for i in range(5)]

    def fake_iter_split(split, snapshot, limit=None):
        data = records if limit is None else records[:limit]
        yield from data

    monkeypatch.setattr(csl, "iter_split", fake_iter_split)
    idx = csl.build_index("test_real", "/unused/snapshot", limit=2)
    assert idx["n"] == 2


def _three_split_indices():
    # b is a subset of a and c is disjoint, so the matrix is asymmetric and a
    # transposed result would fail.
    a = _index("a", 4, exact={1, 2, 3, 4})
    b = _index("b", 2, exact={2, 3})
    c = _index("c", 2, exact={5, 6})
    return {"a": a, "b": b, "c": c}


def test_full_containment_matrix_diagonal_is_always_100():
    indices = _three_split_indices()
    matrix = csl.full_containment_matrix(indices, ["a", "b", "c"])
    assert matrix["a"]["a"] == 100.0
    assert matrix["b"]["b"] == 100.0
    assert matrix["c"]["c"] == 100.0


def test_full_containment_matrix_directional_percentages():
    indices = _three_split_indices()
    matrix = csl.full_containment_matrix(indices, ["a", "b", "c"])
    # b is fully contained in a: 100% of b's bodies are in a...
    assert matrix["b"]["a"] == 100.0
    # ...but only half of a's bodies are in b (asymmetric on purpose).
    assert matrix["a"]["b"] == 50.0
    # c shares nothing with a or b.
    assert matrix["c"]["a"] == 0.0
    assert matrix["a"]["c"] == 0.0
    assert matrix["c"]["b"] == 0.0


def test_full_containment_matrix_empty_split_is_zero_not_division_error():
    indices = _three_split_indices()
    indices["empty"] = _index("empty", 0, exact=set())
    matrix = csl.full_containment_matrix(indices, ["a", "empty"])
    assert matrix["empty"]["a"] == 0.0
    assert matrix["empty"]["empty"] == 100.0  # diagonal special-cased before the division


def test_corpus_dedup_summary_unions_across_splits_and_counts_within_corpus_dups():
    indices = _three_split_indices()
    summary = csl.corpus_dedup_summary(indices, ["a", "b", "c"], paper_reported_total=5)
    # total_rows sums n (not unique bodies): 4 + 2 + 2 = 8.
    assert summary["total_rows"] == 8
    # effective_unique_functions = |{1,2,3,4} ∪ {2,3} ∪ {5,6}| = |{1,2,3,4,5,6}| = 6.
    assert summary["effective_unique_functions"] == 6
    # the 2 within-corpus dup rows are b's bodies (2,3), both already counted via a.
    assert summary["within_corpus_exact_dup_rows"] == 2
    assert summary["delta_vs_paper"] == 6 - 5
    assert summary["splits_covered"] == ["a", "b", "c"]


def test_corpus_dedup_summary_single_split_has_zero_within_corpus_dups():
    indices = _three_split_indices()
    summary = csl.corpus_dedup_summary(indices, ["a"], paper_reported_total=4)
    assert summary["total_rows"] == 4
    assert summary["effective_unique_functions"] == 4
    assert summary["within_corpus_exact_dup_rows"] == 0
    assert summary["delta_vs_paper"] == 0
