import json

import pytest

from scripts import exebenchplus_eval as er


def test_pick_track_explicit_is_returned_as_is():
    assert er.pick_track({}, "angha") == "angha"


def test_pick_track_auto_prefers_real():
    assert er.pick_track({"real_io_pairs": [{}], "angha_io_pairs": [{}]}) == "real"


def test_pick_track_auto_falls_back_to_angha():
    assert er.pick_track({"real_io_pairs": None, "angha_io_pairs": [{}]}) == "angha"


def test_pick_track_auto_none_when_no_pairs():
    assert er.pick_track({"real_io_pairs": [], "angha_io_pairs": None}) is None


def test_load_rows_json_list_keys_by_task_id(tmp_path):
    p = tmp_path / "rows.json"
    p.write_text(json.dumps([{"task_id": "a:1", "x": 1}, {"task_id": "a:5", "x": 2}]))
    assert er.load_rows(str(p)) == {"a:1": {"task_id": "a:1", "x": 1}, "a:5": {"task_id": "a:5", "x": 2}}


def test_load_rows_jsonl_assigns_stem_line_ids_and_filters(tmp_path):
    p = tmp_path / "valid_real.jsonl"
    p.write_text("".join(json.dumps({"x": i}) + "\n" for i in range(4)))
    rows = er.load_rows(str(p), task_ids={"valid_real:2"})
    assert rows == {"valid_real:2": {"x": 2, "task_id": "valid_real:2"}}


def test_load_rows_jsonl_without_filter_keeps_all(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("".join(json.dumps({"x": i}) + "\n" for i in range(3)))
    assert list(er.load_rows(str(p))) == ["s:0", "s:1", "s:2"]


def test_invalid_track_raises(tmp_path):
    p = tmp_path / "preds.json"
    p.write_text("[]")
    with pytest.raises(ValueError):
        er.exebenchplus_validate("m", str(p), str(p), track="synth")
