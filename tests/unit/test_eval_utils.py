import json
import os

from scripts import eval_utils


def test_writes_correct_content(tmp_path):
    p = tmp_path / "out.json"
    eval_utils.atomic_write_json(str(p), {"a": 1, "b": [1, 2, 3]})
    assert json.loads(p.read_text()) == {"a": 1, "b": [1, 2, 3]}


def test_no_leftover_tmp_file(tmp_path):
    p = tmp_path / "out.json"
    eval_utils.atomic_write_json(str(p), {"a": 1})
    assert not (tmp_path / "out.json.tmp").exists()
    assert set(os.listdir(tmp_path)) == {"out.json"}


def test_overwrites_existing_file(tmp_path):
    p = tmp_path / "out.json"
    p.write_text("stale")
    eval_utils.atomic_write_json(str(p), {"fresh": True})
    assert json.loads(p.read_text()) == {"fresh": True}


def test_custom_indent(tmp_path):
    p = tmp_path / "out.json"
    eval_utils.atomic_write_json(str(p), {"a": 1}, indent=4)
    assert p.read_text() == '{\n    "a": 1\n}'


def test_failed_write_leaves_destination_untouched(tmp_path, monkeypatch):
    p = tmp_path / "out.json"
    p.write_text('{"original": true}')

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    try:
        eval_utils.atomic_write_json(str(p), {"new": True})
    except OSError:
        pass

    assert json.loads(p.read_text()) == {"original": True}
    assert (tmp_path / "out.json.tmp").exists()


def test_empty_object(tmp_path):
    p = tmp_path / "out.json"
    eval_utils.atomic_write_json(str(p), {})
    assert json.loads(p.read_text()) == {}
