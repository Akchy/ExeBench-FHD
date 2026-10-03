import json

import pytest

from scripts import exebench_eval as ee
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


def _row(fname="add_one", real_deps="", real_io_pairs=None):
    return {
        "task_id": "t1",
        "fname": fname,
        "real_deps": real_deps,
        "real_exe_wrapper": _WRAPPER,
        "real_io_pairs": real_io_pairs,
    }


def _io_pairs():
    return {
        "input": [{"var": ["x"], "value": ["3"]}],
        "output": [{"var": ["result"], "value": ["4"]}],
    }


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return str(p)


def test_exebench_validate_new_correct_prediction_passes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    row = _row(real_io_pairs=_io_pairs())
    item = {"task_id": "t1", "predicted_c_funcs": [{"pred_func": "int func0(int x) {\n    return x + 1;\n}\n"}]}

    dataset_path = _write(tmp_path, "dataset.json", [item])
    exeb_path = _write(tmp_path, "exeb.json", [row])

    ee.exebench_validate_new("mymodel", dataset_path, exeb_path, parallel_workers=1, resume=False)

    result = json.loads(open(dataset_path).read())
    pred = result[0]["predicted_c_funcs"][0]
    assert pred["re-compile"] is True
    assert pred["re-execute"] is True


def test_exebench_validate_new_wrong_output_fails_execute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    row = _row(real_io_pairs=_io_pairs())
    item = {"task_id": "t1", "predicted_c_funcs": [{"pred_func": "int func0(int x) {\n    return x + 2;\n}\n"}]}

    dataset_path = _write(tmp_path, "dataset.json", [item])
    exeb_path = _write(tmp_path, "exeb.json", [row])

    ee.exebench_validate_new("mymodel", dataset_path, exeb_path, parallel_workers=1, resume=False)

    pred = json.loads(open(dataset_path).read())[0]["predicted_c_funcs"][0]
    assert pred["re-compile"] is True
    assert pred["re-execute"] is False


def test_exebench_validate_new_broken_c_fails_compile(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    row = _row(real_io_pairs=_io_pairs())
    item = {"task_id": "t1", "predicted_c_funcs": [{"pred_func": "int func0(int x) { return x +\n"}]}

    dataset_path = _write(tmp_path, "dataset.json", [item])
    exeb_path = _write(tmp_path, "exeb.json", [row])

    ee.exebench_validate_new("mymodel", dataset_path, exeb_path, parallel_workers=1, resume=False)

    pred = json.loads(open(dataset_path).read())[0]["predicted_c_funcs"][0]
    assert pred["re-compile"] is False
    assert pred["re-execute"] is False


def test_exebench_validate_new_unparseable_prediction_skips_recompile(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    row = _row(real_io_pairs=_io_pairs())
    item = {"task_id": "t1", "predicted_c_funcs": [{"pred_func": "not a function at all"}]}

    dataset_path = _write(tmp_path, "dataset.json", [item])
    exeb_path = _write(tmp_path, "exeb.json", [row])

    ee.exebench_validate_new("mymodel", dataset_path, exeb_path, parallel_workers=1, resume=False)

    pred = json.loads(open(dataset_path).read())[0]["predicted_c_funcs"][0]
    assert pred["re-compile"] is False
    assert pred["re-execute"] is False


def test_exebench_validate_new_no_oracle_marks_item_and_skips(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    row = _row(real_io_pairs=None)
    item = {"task_id": "t1", "predicted_c_funcs": [{"pred_func": "int func0(int x) {\n    return x + 1;\n}\n"}]}

    dataset_path = _write(tmp_path, "dataset.json", [item])
    exeb_path = _write(tmp_path, "exeb.json", [row])

    ee.exebench_validate_new("mymodel", dataset_path, exeb_path, parallel_workers=1, resume=False)

    result = json.loads(open(dataset_path).read())[0]
    assert result["no_oracle"] is True
    assert result["predicted_c_funcs"][0]["re-compile"] is None
    assert result["predicted_c_funcs"][0]["re-execute"] is None


def test_exebench_validate_new_resume_skips_already_validated_predictions(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    row = _row(real_io_pairs=_io_pairs())
    item = {"task_id": "t1", "predicted_c_funcs": [
        {"pred_func": "garbage", "re-compile": False, "re-execute": False},
    ]}

    dataset_path = _write(tmp_path, "dataset.json", [item])
    exeb_path = _write(tmp_path, "exeb.json", [row])

    # Nothing is queued; the file must come back unchanged.
    ee.exebench_validate_new("mymodel", dataset_path, exeb_path, parallel_workers=1, resume=True)

    pred = json.loads(open(dataset_path).read())[0]["predicted_c_funcs"][0]
    assert pred["re-compile"] is False
