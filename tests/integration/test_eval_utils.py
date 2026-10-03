import json

from scripts import eval_utils


def test_real_round_trip_large_nested_object(tmp_path):
    p = tmp_path / "state.json"
    payload = {
        "records": [{"id": i, "vals": [i, i * 2], "ok": i % 2 == 0} for i in range(200)],
        "meta": {"nested": {"deep": [1, 2, {"x": None}]}},
    }
    eval_utils.atomic_write_json(str(p), payload)
    with open(p) as f:
        reloaded = json.load(f)
    assert reloaded == payload

    payload["meta"]["nested"]["deep"].append(3)
    eval_utils.atomic_write_json(str(p), payload)
    with open(p) as f:
        reloaded2 = json.load(f)
    assert reloaded2 == payload
