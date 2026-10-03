import json
import re
from pathlib import Path

import pytest

from scripts.release import convert_jsonl_to_parquet as cv
from scripts.release import upload_to_hf as up

CARD = Path(up.DEFAULT_CARD)
SPLITS = ["train_synth_compilable", "train_real_compilable", "train_synth_simple_io", "train_real_simple_io",
          "train_synth_rich_io", "train_not_compilable", "valid_synth", "valid_real", "test_synth", "test_real"]


def make_data_dir(tmp_path, splits=SPLITS):
    (tmp_path / "data").mkdir()
    for s in splits:
        (tmp_path / "data" / f"{s}-00000-of-00001.parquet").write_bytes(b"")
    return tmp_path


def test_card_declares_the_ten_exebench_splits():
    splits = up.card_splits(up.read_card_yaml(CARD))
    assert sorted(splits) == sorted(SPLITS)
    assert all(pattern == f"data/{s}-*.parquet" for s, pattern in splits.items())
    assert up.read_card_yaml(CARD)["license"] == "mit"


def test_card_documents_every_field_and_the_converters_json_columns():
    text = CARD.read_text()
    for field in cv.FIELDS:
        assert f"`{field}`" in text, field
    block = re.search(r"JSON_COLUMNS = (\[.*?\])", text, re.S).group(1)
    assert json.loads(block) == cv.JSON_COLUMNS


def test_render_card_fills_repo_id():
    text = up.render_card(CARD, "anon/ExeBenchPlus")
    assert "{{repo_id}}" not in text and 'load_dataset("anon/ExeBenchPlus"' in text


def test_check_passes_when_card_and_data_agree(tmp_path):
    assert up.check_card_against_data(CARD, make_data_dir(tmp_path)) == []


def test_check_reports_missing_split_and_orphan_shard(tmp_path):
    d = make_data_dir(tmp_path, [s for s in SPLITS if s != "test_real"])
    (d / "data" / "stray-00000-of-00001.parquet").write_bytes(b"")
    problems = up.check_card_against_data(CARD, d)
    assert any("'test_real': no shard" in p for p in problems)
    assert any("stray-00000-of-00001.parquet is not covered" in p for p in problems)


def test_check_reports_splits_json_mismatch_and_unfinished_run(tmp_path):
    d = make_data_dir(tmp_path)
    (d / "splits.json").write_text(json.dumps({s: {} for s in SPLITS[1:] + ["extra"]}))
    (d / ".tmp_test_real").mkdir()
    problems = up.check_card_against_data(CARD, d)
    assert any(f"'{SPLITS[0]}' is only in the card" in p for p in problems)
    assert any("'extra' is only in splits.json" in p for p in problems)
    assert any("unfinished conversion" in p for p in problems)


class FakeApi:
    calls = []

    def __getattr__(self, name):
        return lambda *a, **kw: FakeApi.calls.append((name, a, kw))


@pytest.fixture
def fake_api(monkeypatch):
    import huggingface_hub
    FakeApi.calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    return FakeApi.calls


def test_main_uploads_rendered_card_and_shards(tmp_path, fake_api):
    d = make_data_dir(tmp_path)
    up.main(["--repo-id", "anon/EBP", "--data-dir", str(d), "--private"])
    names = [c[0] for c in fake_api]
    assert names == ["create_repo", "upload_large_folder"]
    assert fake_api[0][2]["private"] is True and fake_api[0][2]["repo_type"] == "dataset"
    assert fake_api[1][2]["allow_patterns"] == ["README.md", "data/*.parquet"]
    assert 'load_dataset("anon/EBP"' in (d / "README.md").read_text()


def test_main_refuses_to_upload_inconsistent_data(tmp_path, fake_api):
    d = make_data_dir(tmp_path, SPLITS[:3])
    with pytest.raises(SystemExit, match="card and data disagree"):
        up.main(["--repo-id", "anon/EBP", "--data-dir", str(d)])
    assert fake_api == []


def test_card_only_uploads_just_the_card(fake_api):
    up.main(["--repo-id", "anon/EBP", "--card-only"])
    assert [c[0] for c in fake_api] == ["create_repo", "upload_file"]
    upload = fake_api[1][2]
    assert upload["path_in_repo"] == "README.md" and b'load_dataset("anon/EBP"' in upload["path_or_fileobj"]
