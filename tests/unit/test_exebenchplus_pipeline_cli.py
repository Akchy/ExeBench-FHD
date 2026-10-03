"""Mocked tests for the parallel and resume logic in scripts.exebenchplus_pipeline.cli."""
import json

import pytest

from scripts.exebenchplus_pipeline import cli as epp_cli


def test_worker_process_record_returns_index_and_result_on_success(monkeypatch):
    def fake_process_record(rec, split, work_dir, task_id, archs=None, timeout=60, cleanup=True):
        assert split == "test_real"
        assert task_id == "t1"
        return {**rec, "no_return": False}

    monkeypatch.setattr(epp_cli, "process_record", fake_process_record)
    scanned, out = epp_cli._worker_process_record(
        (7, "test_real", "t1", {"fname": "f"}, ["x86-64"], 60, True, "work"))

    assert scanned == 7
    assert out == {"fname": "f", "no_return": False}
    assert "_pipeline_error" not in out


def test_worker_process_record_wraps_exception_never_raises(monkeypatch):
    def failing_process_record(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(epp_cli, "process_record", failing_process_record)
    scanned, out = epp_cli._worker_process_record(
        (7, "test_real", "t1", {"fname": "f"}, ["x86-64"], 60, True, "work"))

    assert scanned == 7
    assert out["fname"] == "f"  # original record preserved
    assert out["_pipeline_error"] == "RuntimeError: boom"


def test_tasks_yields_one_tuple_per_record_with_unique_task_ids(monkeypatch):
    fake_records = [{"fname": "a"}, {"fname": "b"}, {"fname": "c"}]
    monkeypatch.setattr(epp_cli, "iter_split", lambda split, limit=None: iter(fake_records))

    tasks = list(epp_cli._tasks("test_real", n=20, limit=None, write_all=True,
                                 work_dir="work", archs=["x86-64"], timeout=60, cleanup=True))

    assert len(tasks) == 3
    assert [t[0] for t in tasks] == [1, 2, 3]
    task_ids = [t[2] for t in tasks]
    assert task_ids == ["test_real_1", "test_real_2", "test_real_3"]
    assert [t[3] for t in tasks] == fake_records
    assert all(t[1] == "test_real" and t[4] == ["x86-64"] and t[5] == 60 and t[6] is True
               and t[7] == "work" for t in tasks)


def test_tasks_stops_at_n_when_not_write_all(monkeypatch):
    fake_records = [{"fname": str(i)} for i in range(10)]
    monkeypatch.setattr(epp_cli, "iter_split", lambda split, limit=None: iter(fake_records))

    tasks = list(epp_cli._tasks("test_real", n=3, limit=None, write_all=False,
                                 work_dir="work", archs=["x86-64"], timeout=60, cleanup=True))

    assert len(tasks) == 3


def test_tasks_write_all_ignores_n(monkeypatch):
    fake_records = [{"fname": str(i)} for i in range(10)]
    monkeypatch.setattr(epp_cli, "iter_split", lambda split, limit=None: iter(fake_records))

    tasks = list(epp_cli._tasks("test_real", n=3, limit=None, write_all=True,
                                 work_dir="work", archs=["x86-64"], timeout=60, cleanup=True))

    assert len(tasks) == 10


def test_tasks_skip_indices_omits_tasks_but_still_scans_past_them(monkeypatch):
    fake_records = [{"fname": str(i)} for i in range(5)]
    monkeypatch.setattr(epp_cli, "iter_split", lambda split, limit=None: iter(fake_records))

    tasks = list(epp_cli._tasks("test_real", n=5, limit=None, write_all=True,
                                 work_dir="work", archs=["x86-64"], timeout=60, cleanup=True,
                                 skip_indices={2, 4}))

    assert [t[0] for t in tasks] == [1, 3, 5]


def test_main_rejects_workers_less_than_one(run_main, tmp_path):
    with pytest.raises(SystemExit):
        run_main(epp_cli, ["--workers", "0", "--work-dir", str(tmp_path / "work")])


def test_load_completed_indices_returns_empty_set_when_no_ledger(tmp_path):
    assert epp_cli._load_completed_indices(str(tmp_path / "missing.progress")) == set()


def test_load_completed_indices_reads_clean_ledger(tmp_path):
    path = tmp_path / "out.jsonl.progress"
    path.write_text("1\n3\n2\n")
    assert epp_cli._load_completed_indices(str(path)) == {1, 2, 3}


def test_load_completed_indices_drops_torn_trailing_line_and_rewrites(tmp_path):
    path = tmp_path / "out.jsonl.progress"
    path.write_text("1\n2\n3\ncorrupted-tor")  # simulates a kill mid-write, no trailing \n

    indices = epp_cli._load_completed_indices(str(path))

    assert indices == {1, 2, 3}
    assert path.read_text() == "1\n2\n3\n"


def test_load_completed_indices_raises_on_non_trailing_corruption():
    # Only the last line can be torn; earlier garbage is real corruption.
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".progress", delete=False) as f:
        f.write("1\ngarbage\n3\n")
        path = f.name
    with pytest.raises(ValueError):
        epp_cli._load_completed_indices(path)


def test_reconcile_output_file_noop_when_nothing_to_trim(tmp_path):
    path = tmp_path / "out.jsonl"
    content = '{"fname": "a"}\n{"fname": "b"}\n'
    path.write_text(content)

    epp_cli._reconcile_output_file(str(path), done_count=2)

    assert path.read_text() == content


def test_reconcile_output_file_drops_torn_trailing_line(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text('{"fname": "a"}\n{"fname": "b"}\n{"fname": "c", "trunc')

    epp_cli._reconcile_output_file(str(path), done_count=2)

    lines = path.read_text().splitlines()
    assert lines == ['{"fname": "a"}', '{"fname": "b"}']


def test_reconcile_output_file_trims_unacknowledged_tail_beyond_done_count(tmp_path):
    # Output written but ledger entry not yet: the ledger wins.
    path = tmp_path / "out.jsonl"
    path.write_text('{"fname": "a"}\n{"fname": "b"}\n{"fname": "c"}\n')

    epp_cli._reconcile_output_file(str(path), done_count=2)

    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert lines == [{"fname": "a"}, {"fname": "b"}]


def test_reconcile_output_file_missing_file_is_a_noop(tmp_path):
    epp_cli._reconcile_output_file(str(tmp_path / "missing.jsonl"), done_count=0)  # must not raise


def test_reconcile_output_file_streams_without_loading_whole_file(tmp_path):
    """Reconciling must stream the file, not load it into memory."""
    import resource

    path = tmp_path / "out.jsonl"
    n = 300_000
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"fname": f"f{i}", "pad": "x" * 500}) + "\n")

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    epp_cli._reconcile_output_file(str(path), done_count=n - 1)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ru_maxrss is a process-wide high-water mark (KB), so the bound is loose.
    assert (after - before) < 50_000

    lines = path.read_text().splitlines()
    assert len(lines) == n - 1
    assert json.loads(lines[-1])["fname"] == f"f{n - 2}"


def test_resume_state_false_deletes_stale_ledger_and_returns_write_mode(tmp_path):
    out_path = tmp_path / "out.jsonl"
    progress_path = tmp_path / "out.jsonl.progress"
    out_path.write_text('{"fname": "stale"}\n')
    progress_path.write_text("1\n")

    done_indices, mode = epp_cli._resume_state(str(out_path), resume=False)

    assert done_indices == set()
    assert mode == "w"
    assert not progress_path.exists()


def test_resume_state_true_loads_and_reconciles(tmp_path):
    out_path = tmp_path / "out.jsonl"
    progress_path = tmp_path / "out.jsonl.progress"
    out_path.write_text('{"fname": "a"}\n{"fname": "b"}\n')
    progress_path.write_text("1\n2\n")

    done_indices, mode = epp_cli._resume_state(str(out_path), resume=True)

    assert done_indices == {1, 2}
    assert mode == "a"
