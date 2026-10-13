import json

import pytest

from scripts import exebench_fhd_eval as er
from tests.conftest import requires_gxx

pytestmark = [pytest.mark.slow, requires_gxx]

_WRAPPER = """\
extern "C" {
#include "placeholder.c"
}

#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

int main(int argc, char** argv) {
    std::ifstream ifs(argv[1]);
    json input_json = json::parse(ifs);
    int x = input_json["x"];
    int result = add_one(x);
    json output_json;
    output_json["result"] = result;
    std::ofstream out_str(argv[2]);
    out_str << output_json;
    return 0;
}
"""

_ADD_ONE = "int func0(int x) {\n    return x + 1;\n}\n"


def _pair(x, result, dummy_funcs=""):
    return {"input": {"x": x}, "output": {"result": result}, "dummy_funcs": dummy_funcs, "dummy_funcs_seed": 0}


def _row(real_io_pairs=None, angha_io_pairs=None, real_deps="", angha_deps=""):
    return {
        "task_id": "t1",
        "fname": "add_one",
        "real_deps": real_deps,
        "angha_deps": angha_deps,
        "real_exe_wrapper": _WRAPPER,
        "angha_exe_wrapper": _WRAPPER,
        "real_io_pairs": real_io_pairs,
        "angha_io_pairs": angha_io_pairs,
    }


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return str(p)


def _run(tmp_path, row, pred_func, **kw):
    item = {"task_id": "t1", "predicted_c_funcs": [{"pred_func": pred_func}]}
    dataset_path = _write(tmp_path, "dataset.json", [item])
    exeb_path = _write(tmp_path, "exeb.json", [row])
    er.exebench_fhd_validate("mymodel", dataset_path, exeb_path, parallel_workers=1, resume=False, **kw)
    return json.loads(open(dataset_path).read())[0]


@pytest.fixture(autouse=True)
def _in_tmp(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)


def test_correct_prediction_passes(tmp_path):
    pred = _run(tmp_path, _row(real_io_pairs=[_pair(3, 4), _pair(-1, 0)]), _ADD_ONE)["predicted_c_funcs"][0]
    assert pred["re-compile"] is True
    assert pred["re-execute"] is True


def test_wrong_output_fails_execute(tmp_path):
    pred = _run(tmp_path, _row(real_io_pairs=[_pair(3, 4)]),
                "int func0(int x) {\n    return x + 2;\n}\n")["predicted_c_funcs"][0]
    assert pred["re-compile"] is True
    assert pred["re-execute"] is False


def test_broken_c_fails_compile(tmp_path):
    pred = _run(tmp_path, _row(real_io_pairs=[_pair(3, 4)]), "int func0(int x) { return x +\n")["predicted_c_funcs"][0]
    assert pred["re-compile"] is False
    assert pred["re-execute"] is False


def test_unparseable_prediction_fails_compile(tmp_path):
    pred = _run(tmp_path, _row(real_io_pairs=[_pair(3, 4)]), "not a function at all")["predicted_c_funcs"][0]
    assert pred["re-compile"] is False
    assert pred["re-execute"] is False


def test_harness_error_after_build_keeps_recompile_true(tmp_path):
    # A malformed io pair (no "input") is a harness problem, not the model's:
    # the prediction did compile, so re-compile must stay True.
    pred = _run(tmp_path, _row(real_io_pairs=[{"output": {"result": 4}, "dummy_funcs": ""}]),
                _ADD_ONE)["predicted_c_funcs"][0]
    assert pred["re-compile"] is True
    assert pred["re-execute"] is False


def test_no_oracle_marks_item_and_skips(tmp_path):
    result = _run(tmp_path, _row(), _ADD_ONE)
    assert result["no_oracle"] is True
    assert result["predicted_c_funcs"][0]["re-compile"] is None
    assert result["predicted_c_funcs"][0]["re-execute"] is None


def test_auto_track_falls_back_to_angha(tmp_path):
    pred = _run(tmp_path, _row(angha_io_pairs=[_pair(3, 4)]), _ADD_ONE)["predicted_c_funcs"][0]
    assert pred["re-execute"] is True


def test_explicit_real_track_without_real_pairs_is_no_oracle(tmp_path):
    result = _run(tmp_path, _row(angha_io_pairs=[_pair(3, 4)]), _ADD_ONE, track="real")
    assert result["no_oracle"] is True


def test_per_pair_dummy_funcs_are_compiled_in(tmp_path):
    # Each pair stubs the external ext() with a different return value, so it
    # only passes if every pair runs against a binary built with ITS stubs.
    row = _row(
        angha_deps="int ext(void);\n",
        angha_io_pairs=[
            _pair(3, 4, dummy_funcs="int ext(void) { return 1; }\n"),
            _pair(3, 13, dummy_funcs="int ext(void) { return 10; }\n"),
        ],
    )
    pred = _run(tmp_path, row, "int func0(int x) {\n    return x + ext();\n}\n")["predicted_c_funcs"][0]
    assert pred["re-compile"] is True
    assert pred["re-execute"] is True


def test_jsonl_dataset_uses_stem_and_line_number_task_ids(tmp_path):
    rows = [_row(real_io_pairs=[_pair(0, 0)]), _row(real_io_pairs=[_pair(3, 4)])]
    for r in rows:
        del r["task_id"]
    (tmp_path / "test_real.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    item = {"task_id": "test_real:1", "predicted_c_funcs": [{"pred_func": _ADD_ONE}]}
    dataset_path = _write(tmp_path, "dataset.json", [item])

    er.exebench_fhd_validate("mymodel", dataset_path, str(tmp_path / "test_real.jsonl"),
                             parallel_workers=1, resume=False)

    pred = json.loads(open(dataset_path).read())[0]["predicted_c_funcs"][0]
    assert pred["re-execute"] is True  # row 0 expects 0 -> 0, which add_one would fail


def test_resume_skips_already_validated_predictions(tmp_path):
    item = {"task_id": "t1", "predicted_c_funcs": [{"pred_func": "garbage", "re-compile": False, "re-execute": False}]}
    dataset_path = _write(tmp_path, "dataset.json", [item])
    exeb_path = _write(tmp_path, "exeb.json", [_row(real_io_pairs=[_pair(3, 4)])])

    er.exebench_fhd_validate("mymodel", dataset_path, exeb_path, parallel_workers=1, resume=True)

    pred = json.loads(open(dataset_path).read())[0]["predicted_c_funcs"][0]
    assert pred["re-compile"] is False
