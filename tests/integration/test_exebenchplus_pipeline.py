"""End-to-end CLI tests for scripts.exebenchplus_pipeline on small synthetic
records (via patch_iter_split), covering fields, cleanup, --workers and --resume.
"""
import json

import pytest

from scripts.exebenchplus_pipeline import cli as epp_cli
from scripts.extract.extract_func_call import ARCH_TOOLCHAIN
from tests.conftest import requires_arm32_gcc, requires_arm64_gcc, requires_gcc, requires_x86_32

pytestmark = [pytest.mark.slow, requires_gcc]

_ADD_ONE_DEF = "int add_one(int x) {\n    return x + 1;\n}\n"
_MUL_TWO_DEF = "int mul_two(int x) {\n    return x * 2;\n}\n"

_ALL_DOCUMENTED_FIELDS = [
    "no_return", "dummy_udt", "trivial_function", "truncated_funcs", "unused_param",
    "func_call", "func_call_errors", "func_call_skip_reason",
    "data_section", "data_section_errors",
]

# Produced only by extract_asm_regen.py, never by the pipeline.
_OUT_OF_SCOPE_FIELDS = (
    "asm", "asm_was_static", "objdump_asm", "objdump_asm_errors",
    "disassembled_asm", "disassembled_asm_errors",
)


def _two_records(record_factory):
    r1 = record_factory(fname="add_one", func_head="int add_one(int x)",
                         func_head_types="int add_one(int x)", func_def=_ADD_ONE_DEF,
                         signature=["int", "int"], path="o/r/a.c")
    r2 = record_factory(fname="mul_two", func_head="int mul_two(int x)",
                         func_head_types="int mul_two(int x)", func_def=_MUL_TWO_DEF,
                         signature=["int", "int"], path="o/r/b.c")
    return [r1, r2]


def _n_records(record_factory, n):
    records = []
    for i in range(n):
        fname = f"add_{i}"
        func_def = f"int {fname}(int x) {{\n    return x + {i};\n}}\n"
        records.append(record_factory(
            fname=fname, func_head=f"int {fname}(int x)",
            func_head_types=f"int {fname}(int x)", func_def=func_def,
            signature=["int", "int"], path=f"o/r/{fname}.c"))
    return records


def test_cli_writes_every_documented_field_for_every_record(tmp_path, patch_iter_split, run_main,
                                                              record_factory):
    patch_iter_split(epp_cli, _two_records(record_factory))
    out_path = tmp_path / "out.jsonl"

    run_main(epp_cli, ["--out", str(out_path), "--split", "train_real_compilable", "--all",
                       "--archs", "x86-64", "--work-dir", str(tmp_path / "work")])

    lines = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(lines) == 2
    assert {l["fname"] for l in lines} == {"add_one", "mul_two"}
    for line in lines:
        for field in _ALL_DOCUMENTED_FIELDS:
            assert field in line, f"missing field {field!r} in output record {line.get('fname')}"
        assert line["func_def"] in (_ADD_ONE_DEF, _MUL_TWO_DEF)
        assert line["func_call_skip_reason"] is None
        assert line["func_call"]["x86-64"]
        assert line["func_call_errors"]["x86-64"] is None
        for field in _OUT_OF_SCOPE_FIELDS:
            assert field not in line, f"{field!r} should not be in the dataset-pipeline output"


def test_cli_preserves_exebench_native_asm_field(tmp_path, patch_iter_split, run_main, record_factory):
    """ExeBench's own `asm` field (different shape) must survive untouched."""
    native_asm = {"real_gcc_x86_O0": {"func_asm": ".globl add_one\nadd_one:\n\tret\n"}}
    rec = record_factory(fname="add_one", func_head="int add_one(int x)",
                          func_head_types="int add_one(int x)", func_def=_ADD_ONE_DEF,
                          signature=["int", "int"], path="o/r/a.c", asm=native_asm)
    patch_iter_split(epp_cli, [rec])
    out_path = tmp_path / "out.jsonl"

    run_main(epp_cli, ["--out", str(out_path), "--all", "--archs", "x86-64",
                       "--work-dir", str(tmp_path / "work")])

    line = json.loads(out_path.read_text().splitlines()[0])
    assert line["asm"] == native_asm
    assert "asm_was_static" not in line


def test_cli_n_flag_caps_records_processed(tmp_path, patch_iter_split, run_main, record_factory):
    patch_iter_split(epp_cli, _two_records(record_factory))
    out_path = tmp_path / "out.jsonl"

    run_main(epp_cli, ["--out", str(out_path), "--n", "1", "--archs", "x86-64",
                       "--work-dir", str(tmp_path / "work")])

    lines = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(lines) == 1


def test_cli_default_cleanup_leaves_no_stray_task_files(tmp_path, patch_iter_split, run_main,
                                                          record_factory):
    work_dir = tmp_path / "work"
    patch_iter_split(epp_cli, _two_records(record_factory))
    out_path = tmp_path / "out.jsonl"

    run_main(epp_cli, ["--out", str(out_path), "--all", "--archs", "x86-64",
                       "--work-dir", str(work_dir)])

    c_files = work_dir / "C_files"
    # Only func_call writes into C_files/; data_section uses its own tempdir.
    leftover = list(c_files.glob("*")) if c_files.exists() else []
    assert leftover == [], f"stray files left behind: {leftover}"


def test_cli_keep_artifacts_leaves_files_on_disk(tmp_path, patch_iter_split, run_main, record_factory):
    work_dir = tmp_path / "work"
    patch_iter_split(epp_cli, _two_records(record_factory))
    out_path = tmp_path / "out.jsonl"

    run_main(epp_cli, ["--out", str(out_path), "--all", "--archs", "x86-64",
                       "--keep-artifacts", "--work-dir", str(work_dir)])

    c_files = work_dir / "C_files"
    assert c_files.exists()
    assert list(c_files.glob("*")), "expected --keep-artifacts to leave scratch files behind"


@requires_x86_32
@requires_arm32_gcc
@requires_arm64_gcc
def test_cli_cross_architecture_fields_populate_for_every_arch(tmp_path, patch_iter_split, run_main,
                                                                 record_factory):
    patch_iter_split(epp_cli, [_two_records(record_factory)[0]])
    out_path = tmp_path / "out.jsonl"

    run_main(epp_cli, ["--out", str(out_path), "--all", "--work-dir", str(tmp_path / "work")])

    line = json.loads(out_path.read_text().splitlines()[0])
    for arch in ARCH_TOOLCHAIN:
        assert line["func_call_errors"][arch] is None, line["func_call_errors"][arch]
        assert line["func_call"][arch]
        assert line["data_section_errors"][arch] is None, line["data_section_errors"][arch]


def test_cli_split_all_writes_one_file_per_split(tmp_path, patch_iter_split, run_main, record_factory):
    patch_iter_split(epp_cli, [_two_records(record_factory)[0]])
    out_dir = tmp_path / "field_out"

    run_main(epp_cli, ["--split", "all", "--all", "--out-dir", str(out_dir), "--archs", "x86-64",
                       "--work-dir", str(tmp_path / "work")])

    from scripts.check_split_leakage import ALL_SPLITS
    for split in ALL_SPLITS:
        split_path = out_dir / f"{split}.jsonl"
        assert split_path.exists()
        line = json.loads(split_path.read_text().splitlines()[0])
        assert line["fname"] == "add_one"
        for field in _ALL_DOCUMENTED_FIELDS:
            assert field in line


def test_cli_rejects_unknown_arch(run_main, tmp_path):
    with pytest.raises(SystemExit):
        run_main(epp_cli, ["--archs", "not-a-real-arch", "--work-dir", str(tmp_path / "work")])


def test_cli_rejects_workers_less_than_one(run_main, tmp_path):
    with pytest.raises(SystemExit):
        run_main(epp_cli, ["--workers", "0", "--work-dir", str(tmp_path / "work")])


def test_cli_workers_parallel_matches_sequential_output(tmp_path, patch_iter_split, run_main,
                                                          record_factory):
    """--workers 4 must produce the same records as --workers 1 (compared by
    fname, since imap_unordered does not preserve order)."""
    records = _n_records(record_factory, 8)

    patch_iter_split(epp_cli, records)
    seq_out = tmp_path / "seq.jsonl"
    run_main(epp_cli, ["--out", str(seq_out), "--all", "--archs", "x86-64", "--workers", "1",
                       "--work-dir", str(tmp_path / "work_seq")])

    patch_iter_split(epp_cli, records)
    par_out = tmp_path / "par.jsonl"
    run_main(epp_cli, ["--out", str(par_out), "--all", "--archs", "x86-64", "--workers", "4",
                       "--work-dir", str(tmp_path / "work_par")])

    seq_by_fname = {json.loads(l)["fname"]: json.loads(l) for l in seq_out.read_text().splitlines()}
    par_by_fname = {json.loads(l)["fname"]: json.loads(l) for l in par_out.read_text().splitlines()}

    assert set(seq_by_fname) == set(par_by_fname) == {r["fname"] for r in records}
    for fname in seq_by_fname:
        assert seq_by_fname[fname] == par_by_fname[fname], f"mismatch for {fname!r}"


def test_cli_workers_respects_n_cap(tmp_path, patch_iter_split, run_main, record_factory):
    records = _n_records(record_factory, 8)
    patch_iter_split(epp_cli, records)
    out_path = tmp_path / "out.jsonl"

    run_main(epp_cli, ["--out", str(out_path), "--n", "3", "--archs", "x86-64", "--workers", "4",
                       "--work-dir", str(tmp_path / "work")])

    lines = out_path.read_text().splitlines()
    assert len(lines) == 3


def test_cli_workers_default_cleanup_leaves_no_stray_files(tmp_path, patch_iter_split, run_main,
                                                            record_factory):
    work_dir = tmp_path / "work"
    records = _n_records(record_factory, 8)
    patch_iter_split(epp_cli, records)
    out_path = tmp_path / "out.jsonl"

    run_main(epp_cli, ["--out", str(out_path), "--all", "--archs", "x86-64", "--workers", "4",
                       "--work-dir", str(work_dir)])

    c_files = work_dir / "C_files"
    leftover = list(c_files.glob("*")) if c_files.exists() else []
    assert leftover == [], f"stray files left behind under concurrent workers: {leftover}"
    assert len(out_path.read_text().splitlines()) == 8


def test_cli_workers_keep_artifacts_leaves_files_from_every_worker(tmp_path, patch_iter_split,
                                                                    run_main, record_factory):
    work_dir = tmp_path / "work"
    records = _n_records(record_factory, 8)
    patch_iter_split(epp_cli, records)
    out_path = tmp_path / "out.jsonl"

    run_main(epp_cli, ["--out", str(out_path), "--all", "--archs", "x86-64", "--workers", "4",
                       "--keep-artifacts", "--work-dir", str(work_dir)])

    c_files = work_dir / "C_files"
    # One .c per record: unique task_ids keep workers from clobbering each other.
    c_sources = list(c_files.glob("*_fc.c"))
    assert len(c_sources) == 8, c_sources


def test_worker_exception_becomes_pipeline_error_not_a_crashed_run(tmp_path, patch_iter_split,
                                                                    run_main, record_factory,
                                                                    monkeypatch):
    """One record's failure must not take down a --workers run."""
    """One record's unexpected failure must not take down the whole
    --workers run - see _worker_process_record's docstring."""
    records = _n_records(record_factory, 4)
    patch_iter_split(epp_cli, records)

    real_process_record = epp_cli.process_record

    def flaky_process_record(rec, split, work_dir, task_id, **kwargs):
        if rec["fname"] == "add_2":
            raise RuntimeError("simulated failure")
        return real_process_record(rec, split, work_dir, task_id, **kwargs)

    monkeypatch.setattr(epp_cli, "process_record", flaky_process_record)
    out_path = tmp_path / "out.jsonl"

    run_main(epp_cli, ["--out", str(out_path), "--all", "--archs", "x86-64", "--workers", "4",
                       "--work-dir", str(tmp_path / "work")])

    lines = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(lines) == 4
    by_fname = {l["fname"]: l for l in lines}
    assert by_fname["add_2"]["_pipeline_error"] == "RuntimeError: simulated failure"
    for fname in ("add_0", "add_1", "add_3"):
        assert "_pipeline_error" not in by_fname[fname]


def test_resume_sequential_continues_without_recomputing_done_records(
        tmp_path, patch_iter_split, run_main, record_factory, monkeypatch):
    """--resume must not recompute records already written by an earlier run."""
    records = _n_records(record_factory, 6)
    call_count = {"n": 0}
    real_process_record = epp_cli.process_record

    def counting_process_record(rec, split, work_dir, task_id, **kwargs):
        call_count["n"] += 1
        return real_process_record(rec, split, work_dir, task_id, **kwargs)

    monkeypatch.setattr(epp_cli, "process_record", counting_process_record)
    out_path = tmp_path / "out.jsonl"
    work_dir = tmp_path / "work"

    # Stop after 3 of 6 records, then resume with --all.
    patch_iter_split(epp_cli, records)
    run_main(epp_cli, ["--out", str(out_path), "--n", "3", "--archs", "x86-64",
                       "--resume", "--work-dir", str(work_dir)])
    assert call_count["n"] == 3
    assert len(out_path.read_text().splitlines()) == 3
    progress_path = tmp_path / "out.jsonl.progress"
    assert progress_path.exists()
    assert len(progress_path.read_text().splitlines()) == 3

    patch_iter_split(epp_cli, records)
    run_main(epp_cli, ["--out", str(out_path), "--all", "--archs", "x86-64",
                       "--resume", "--work-dir", str(work_dir)])

    assert call_count["n"] == 6
    lines = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(lines) == 6
    assert {l["fname"] for l in lines} == {r["fname"] for r in records}
    assert len(lines) == len({l["fname"] for l in lines})


def test_resume_parallel_continues_without_recomputing_done_records(
        tmp_path, patch_iter_split, run_main, record_factory, monkeypatch):
    """Parallel variant. Calls are counted through a file because forked
    workers cannot update this process's memory."""
    records = _n_records(record_factory, 8)
    calls_log = tmp_path / "calls.log"
    real_process_record = epp_cli.process_record

    def counting_process_record(rec, split, work_dir, task_id, **kwargs):
        with open(calls_log, "a") as f:
            f.write(rec["fname"] + "\n")
        return real_process_record(rec, split, work_dir, task_id, **kwargs)

    monkeypatch.setattr(epp_cli, "process_record", counting_process_record)
    out_path = tmp_path / "out.jsonl"
    work_dir = tmp_path / "work"

    patch_iter_split(epp_cli, records)
    run_main(epp_cli, ["--out", str(out_path), "--n", "5", "--archs", "x86-64",
                       "--workers", "4", "--resume", "--work-dir", str(work_dir)])
    assert len(calls_log.read_text().splitlines()) == 5

    patch_iter_split(epp_cli, records)
    run_main(epp_cli, ["--out", str(out_path), "--all", "--archs", "x86-64",
                       "--workers", "4", "--resume", "--work-dir", str(work_dir)])

    assert len(calls_log.read_text().splitlines()) == 8
    lines = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(lines) == 8
    assert {l["fname"] for l in lines} == {r["fname"] for r in records}
    assert len(lines) == len({l["fname"] for l in lines})


def test_resume_recovers_from_unacknowledged_tail_record(
        tmp_path, patch_iter_split, run_main, record_factory, monkeypatch):
    """Output line written but its ledger entry lost: resume must trim the
    unacknowledged line and redo that one record, without duplicates."""
    records = _n_records(record_factory, 4)
    call_count = {"n": 0}
    real_process_record = epp_cli.process_record

    def counting_process_record(rec, split, work_dir, task_id, **kwargs):
        call_count["n"] += 1
        return real_process_record(rec, split, work_dir, task_id, **kwargs)

    monkeypatch.setattr(epp_cli, "process_record", counting_process_record)
    out_path = tmp_path / "out.jsonl"
    work_dir = tmp_path / "work"

    patch_iter_split(epp_cli, records)
    run_main(epp_cli, ["--out", str(out_path), "--all", "--archs", "x86-64",
                       "--resume", "--work-dir", str(work_dir)])
    assert call_count["n"] == 4

    # Drop the last ledger entry while keeping its output line.
    progress_path = tmp_path / "out.jsonl.progress"
    progress_lines = progress_path.read_text().splitlines()
    assert len(progress_lines) == 4
    progress_path.write_text("\n".join(progress_lines[:3]) + "\n")

    patch_iter_split(epp_cli, records)
    run_main(epp_cli, ["--out", str(out_path), "--all", "--archs", "x86-64",
                       "--resume", "--work-dir", str(work_dir)])

    assert call_count["n"] == 5
    final_lines = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(final_lines) == 4
    assert {l["fname"] for l in final_lines} == {r["fname"] for r in records}
    assert len(final_lines) == len({l["fname"] for l in final_lines})


def test_resume_off_by_default_discards_stale_progress_and_starts_fresh(
        tmp_path, patch_iter_split, run_main, record_factory):
    """Without --resume, a leftover ledger is deleted and output truncated."""
    records = _n_records(record_factory, 3)
    out_path = tmp_path / "out.jsonl"
    progress_path = tmp_path / "out.jsonl.progress"
    out_path.write_text('{"fname": "stale_leftover"}\n')
    progress_path.write_text("1\n")

    patch_iter_split(epp_cli, records)
    run_main(epp_cli, ["--out", str(out_path), "--all", "--archs", "x86-64",
                       "--work-dir", str(tmp_path / "work")])

    lines = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(lines) == 3
    assert {l["fname"] for l in lines} == {r["fname"] for r in records}
    assert "stale_leftover" not in {l["fname"] for l in lines}
    assert not progress_path.exists()


def test_resume_clean_run_never_creates_a_progress_file(
        tmp_path, patch_iter_split, run_main, record_factory):
    """No ledger file is written without --resume."""
    records = _n_records(record_factory, 3)
    out_path = tmp_path / "out.jsonl"

    patch_iter_split(epp_cli, records)
    run_main(epp_cli, ["--out", str(out_path), "--all", "--archs", "x86-64",
                       "--work-dir", str(tmp_path / "work")])

    assert not (tmp_path / "out.jsonl.progress").exists()


def test_resume_split_all_resumes_only_the_unfinished_split(
        tmp_path, patch_iter_split, run_main, record_factory):
    """Re-running a finished --split all with --resume changes nothing."""
    records = _n_records(record_factory, 2)
    out_dir = tmp_path / "field_out"

    patch_iter_split(epp_cli, records)
    run_main(epp_cli, ["--split", "all", "--all", "--out-dir", str(out_dir),
                       "--archs", "x86-64", "--resume", "--work-dir", str(tmp_path / "work")])

    from scripts.check_split_leakage import ALL_SPLITS
    for split in ALL_SPLITS:
        lines = (out_dir / f"{split}.jsonl").read_text().splitlines()
        assert len(lines) == 2

    patch_iter_split(epp_cli, records)
    run_main(epp_cli, ["--split", "all", "--all", "--out-dir", str(out_dir),
                       "--archs", "x86-64", "--resume", "--work-dir", str(tmp_path / "work")])

    for split in ALL_SPLITS:
        lines = (out_dir / f"{split}.jsonl").read_text().splitlines()
        assert len(lines) == 2
