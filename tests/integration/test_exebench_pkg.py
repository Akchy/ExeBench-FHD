import pytest

from scripts import exebench as eb
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
    int result = double_it(x);
    json output_json;
    output_json["result"] = result;
    std::ofstream out_str(argv[2]);
    out_str << output_json;
    return 0;
}
"""


def test_exebench_wrapper_compiles_and_calls_real_function():
    wrapper = eb.Exebench_Wrapper(
        c_deps="int double_it(int x) { return x * 2; }\n",
        func_c_signature="int double_it(int x)",
        cpp_wrapper=_WRAPPER,
    )
    out = wrapper({"x": 21})
    assert out == {"result": 42}


def test_exebench_wrapper_bad_deps_raises_compile_error():
    with pytest.raises(eb.CompileError):
        eb.Exebench_Wrapper(
            c_deps="this is not valid C {{{",
            func_c_signature="int double_it(int x)",
            cpp_wrapper=_WRAPPER,
        )


def test_exebench_wrapper_return_stdout_and_stderr():
    wrapper = eb.Exebench_Wrapper(
        c_deps="int double_it(int x) { return x * 2; }\n",
        func_c_signature="int double_it(int x)",
        cpp_wrapper=_WRAPPER,
    )
    out, stdout, stderr = wrapper({"x": 5}, return_stdout_and_stderr=True)
    assert out == {"result": 10}
