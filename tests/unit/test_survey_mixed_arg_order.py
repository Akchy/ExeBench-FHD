import json

from scripts.survey import survey_mixed_arg_order as smao

# fmt: off
# R1  int + float                         -> mixed (int_family & float_family)
# R2  float + float                       -> not mixed (float only)
# R3  int + int                           -> not mixed (int only)
# R4  double + int*                       -> mixed (float_family & other_typed_pointer)
# R5  ["void"] only (len<3)               -> not mixed (too short, no params)
# R6  ["int","int"] single param (len<3)  -> not mixed (only 1 real param)
# R7  char* + struct foo*                 -> not mixed (both pointer/int-like, no float)
# R8  double + double + double            -> not mixed (float only, 3 params)
# R9  long double + unsigned int          -> mixed (float_family & int_family)
# R10 __m128 + int                        -> not mixed (simd_or_complex isn't float_family)
_SIGS = [
    ["int", "int", "float"],
    ["void", "float", "float"],
    ["int", "int", "int"],
    ["float", "double", "int *"],
    ["void"],
    ["int", "int"],
    ["void", "char *", "struct foo *"],
    ["double", "double", "double"],
    ["int", "long double", "unsigned int"],
    ["void", "__m128", "int"],
]
_EXPECTED_MIXED = [True, False, False, True, False, False, False, False, True, False]
# fmt: on


def test_has_mixed_arg_order_hand_computed():
    for sig, expected in zip(_SIGS, _EXPECTED_MIXED):
        assert smao.has_mixed_arg_order(sig) is expected, sig


def test_has_mixed_arg_order_empty_signature():
    assert smao.has_mixed_arg_order(None) is False
    assert smao.has_mixed_arg_order([]) is False


def test_survey_hand_computed_prevalence(patch_iter_split):
    patch_iter_split(smao, [dict(signature=sig) for sig in _SIGS])

    result = smao.survey("fixture_split")

    assert result["records_scanned"] == 10
    assert result["records_with_signature"] == 10
    # R1, R4, R9 are the 3 mixed-arg-order records.
    assert result["mixed_arg_order_count"] == 3
    assert result["mixed_arg_order_pct"] == 30.0


def test_survey_skips_records_without_signature(patch_iter_split):
    patch_iter_split(smao, [dict(signature=["int", "int", "float"]), dict(signature=None), dict(signature=[])])

    result = smao.survey("fixture_split")

    assert result["records_scanned"] == 3
    assert result["records_with_signature"] == 1
    assert result["mixed_arg_order_count"] == 1
    assert result["mixed_arg_order_pct"] == 100.0


def test_cli_writes_output_file(patch_iter_split, tmp_path, run_main):
    patch_iter_split(smao, [dict(signature=sig) for sig in _SIGS])
    out_path = tmp_path / "mixed_arg_order.json"

    run_main(smao, ["--split", "test_real", "--out", str(out_path)])

    report = json.loads(out_path.read_text())
    assert report["per_split"]["test_real"]["mixed_arg_order_count"] == 3
    assert report["total_mixed_arg_order_count"] == 3
    assert report["total_mixed_arg_order_pct"] == 30.0
