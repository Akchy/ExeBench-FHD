import json

from scripts.preprocess import helper_func


def test_count_logs_writes_counts_and_histogram(tmp_path):
    log_file = tmp_path / "correction_prompt.txt"
    log_file.write_text(
        "--- Correction Attempt: 1_O1_5 ---\n"
        "--- Correction Attempt: 1_O1_5 ---\n"
        "--- Correction Attempt: 2_O2_3 ---\n"
    )

    helper_func.count_logs(str(tmp_path / "whatever_source.c"))

    counts_text = (tmp_path / "counts.log").read_text()
    assert "1_O1_5: 2" in counts_text
    assert "2_O2_3: 1" in counts_text

    hist_text = (tmp_path / "histogram.log").read_text()
    assert "Files with count 1: 1" in hist_text  # 2_O2_3 has count 1
    assert "Files with count 2: 1" in hist_text  # 1_O1_5 has count 2
    assert "Files with count 3: 0" in hist_text


def test_count_logs_no_correction_file_writes_nothing(tmp_path):
    helper_func.count_logs(str(tmp_path / "whatever_source.c"))
    assert not (tmp_path / "counts.log").exists()
    assert not (tmp_path / "histogram.log").exists()


def test_c_files_to_json_collects_only_dot_c_files(tmp_path):
    (tmp_path / "a.c").write_text("int a(void){return 0;}")
    (tmp_path / "b.c").write_text("int b(void){return 1;}")
    (tmp_path / "notes.txt").write_text("not a C file")

    out_path = tmp_path / "out.json"
    helper_func.c_files_to_json(str(tmp_path), str(out_path))

    data = json.loads(out_path.read_text())
    by_name = {d["file_name"]: d["code"] for d in data}
    assert set(by_name) == {"a.c", "b.c"}
    assert by_name["a.c"] == "int a(void){return 0;}"
