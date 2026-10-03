import pytest

from scripts import exebench as eb
from scripts.extract import adapt_unused_param_wrapper as auw
from tests.conftest import requires_gxx

pytestmark = [pytest.mark.slow, requires_gxx]

_ORIGINAL_WRAPPER = """\
extern "C" {
#include "placeholder.c"
}

#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

int main(int argc, char** argv) {
    std::ifstream ifs(argv[1]);
    json input_json = json::parse(ifs);
    int a = input_json["a"];
    int b = input_json["b"];
    int result = add(a, b);
    json output_json;
    output_json["result"] = result;
    std::ofstream out_str(argv[2]);
    out_str << output_json;
    return 0;
}
"""

# A correct decompilation that drops the unused parameter b.
_PREDICTED_ADD_ONE_PARAM = "int add(int a) { return a; }\n"


def test_original_wrapper_fails_to_compile_against_reduced_prediction():
    with pytest.raises(eb.CompileError):
        eb.Exebench_Wrapper(
            c_deps=_PREDICTED_ADD_ONE_PARAM,
            func_c_signature="int add(int a)",
            cpp_wrapper=_ORIGINAL_WRAPPER,
        )


def test_adapted_wrapper_compiles_and_runs_against_reduced_prediction():
    adapted = auw.adapt_exe_wrapper(_ORIGINAL_WRAPPER, "add", ["b"])
    assert 'input_json["b"]' not in adapted

    wrapper = eb.Exebench_Wrapper(
        c_deps=_PREDICTED_ADD_ONE_PARAM,
        func_c_signature="int add(int a)",
        cpp_wrapper=adapted,
    )
    out = wrapper({"a": 21})
    assert out == {"result": 21}
