"""Shared fixtures for the ExeBench-FHD test suite.

Two schemas recur across scripts/ and get their own factory fixture here:

  * `make_record()` - an ExeBench dataset record, as read by `iter_split()`
    and consumed by every `extract_*`/`survey_*` script. Fields match what
    scripts/*.py actually reads: fname, func_head, func_def, signature,
    real_deps, angha_deps, real_io_pairs, angha_io_pairs, path.

  * `make_dict_db()` - the small dict `scripts/preprocess/get_asm.py`
    expects: type (opt level), task_id, c_func, fname.

`patch_iter_split` is the standard way to full-code-test a script's CLI
without the real ExeBench snapshot on disk: it monkeypatches the `iter_split`
name already imported into the target module's own namespace (every script
does `from scripts.check_split_leakage import iter_split`, so the patch
target is `<module>.iter_split`, not the original definition site).
"""
import shutil
import sys
import copy

import pytest

HAVE_GCC = shutil.which("gcc") is not None
HAVE_GXX = shutil.which("g++") is not None
HAVE_OBJDUMP = shutil.which("objdump") is not None
HAVE_ARM32_GCC = shutil.which("arm-linux-gnueabihf-gcc") is not None
HAVE_ARM64_GCC = shutil.which("aarch64-linux-gnu-gcc") is not None


def _x86_32_link_works():
    if not HAVE_GCC:
        return False
    import subprocess
    import tempfile
    import os
    src = "int main(void){return 0;}\n"
    with tempfile.TemporaryDirectory() as d:
        c_path = os.path.join(d, "t.c")
        out_path = os.path.join(d, "t.out")
        with open(c_path, "w") as f:
            f.write(src)
        proc = subprocess.run(["gcc", "-m32", c_path, "-o", out_path],
                               capture_output=True, text=True)
        return proc.returncode == 0


HAVE_X86_32 = _x86_32_link_works()

requires_gcc = pytest.mark.skipif(not HAVE_GCC, reason="gcc not on PATH")
requires_gxx = pytest.mark.skipif(not HAVE_GXX, reason="g++ not on PATH")
requires_objdump = pytest.mark.skipif(not HAVE_OBJDUMP, reason="objdump not on PATH")
requires_arm32_gcc = pytest.mark.skipif(not HAVE_ARM32_GCC, reason="arm-linux-gnueabihf-gcc not on PATH")
requires_arm64_gcc = pytest.mark.skipif(not HAVE_ARM64_GCC, reason="aarch64-linux-gnu-gcc not on PATH")
requires_x86_32 = pytest.mark.skipif(not HAVE_X86_32, reason="gcc -m32 cannot link on this machine")

_DEFAULT_RECORD = {
    "fname": "add_one",
    "func_head": "int add_one(int x)",
    "func_def": "int add_one(int x) {\n    return x + 1;\n}\n",
    "signature": ["int", "int"],
    "real_deps": "",
    "angha_deps": "",
    "real_io_pairs": {"var": [], "value": []},
    "angha_io_pairs": {"var": [], "value": []},
    "path": "someowner/somerepo/src/file.c",
    "ref": "abc123",
}


def make_record(**overrides):
    rec = copy.deepcopy(_DEFAULT_RECORD)
    rec.update(overrides)
    return rec


@pytest.fixture
def record_factory():
    return make_record


_DEFAULT_DICT_DB = {
    "type": "O0",
    "task_id": "1",
    "c_func": "int func0(int x) {\n    return x + 1;\n}\n",
    "fname": "func0",
}


def make_dict_db(**overrides):
    d = copy.deepcopy(_DEFAULT_DICT_DB)
    d.update(overrides)
    return d


@pytest.fixture
def dict_db_factory():
    return make_dict_db


@pytest.fixture
def patch_iter_split(monkeypatch):
    """patch_iter_split(module, records) - make module.iter_split(split, ...)
    yield exactly `records`, ignoring split/snapshot/limit args (unless
    limit is given, which truncates - matching the real generator's
    contract closely enough for CLI-level tests)."""

    def _patch(module, records):
        def fake_iter_split(split, *args, limit=None, **kwargs):
            data = records if limit is None else records[:limit]
            for r in data:
                yield r
        monkeypatch.setattr(module, "iter_split", fake_iter_split)
    return _patch


@pytest.fixture
def run_main(monkeypatch):
    """run_main(module, argv) - run module.main() with sys.argv patched,
    the way the script's own `if __name__ == "__main__":` block would."""

    def _run(module, argv):
        monkeypatch.setattr(sys, "argv", ["prog"] + argv)
        return module.main()
    return _run
