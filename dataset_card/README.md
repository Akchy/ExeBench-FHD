---
license: mit
pretty_name: ExeBenchPlus
language:
- code
task_categories:
- text-generation
tags:
- decompilation
- binary-analysis
- reverse-engineering
- assembly
- c
size_categories:
- 1M<n<10M
configs:
- config_name: default
  data_files:
  - split: train_synth_compilable
    path: data/train_synth_compilable-*.parquet
  - split: train_real_compilable
    path: data/train_real_compilable-*.parquet
  - split: train_synth_simple_io
    path: data/train_synth_simple_io-*.parquet
  - split: train_real_simple_io
    path: data/train_real_simple_io-*.parquet
  - split: train_synth_rich_io
    path: data/train_synth_rich_io-*.parquet
  - split: train_not_compilable
    path: data/train_not_compilable-*.parquet
  - split: valid_synth
    path: data/valid_synth-*.parquet
  - split: valid_real
    path: data/valid_real-*.parquet
  - split: test_synth
    path: data/test_synth-*.parquet
  - split: test_real
    path: data/test_real-*.parquet
---

# ExeBenchPlus

ExeBenchPlus is an audited and extended version of
[ExeBench](https://huggingface.co/datasets/jordiae/exebench), a dataset of
executable C functions used to train and evaluate neural decompilers. Auditing
all 6.56M functions in ExeBench's ten splits turned up nine data-quality
problems. In three of them, the function's C code cannot be recovered from its
assembly alone:

- P1, parameter order: x86-64, ARM32 and ARM64 pass integer/pointer and
  floating-point parameters in separate registers, so the callee's assembly
  does not show how mixed parameters are ordered.
- P2, unused parameters: from `-O1` up, an unused parameter leaves no trace in
  the callee's assembly.
- P3, data sections: constants live in `.rodata`, `.data` or `.bss`, and the
  function's code only refers to them by address.

ExeBenchPlus keeps all 23 ExeBench fields unchanged and adds 16: the assembly
of a call to the function (`func_call`, P1), wrappers that test a
reduced-arity decompilation (`opt_*_exe_wrapper`, P2), data-section contents
(`data_section`, P3), disassembly of the linked binary on four architectures
at five optimization levels (`objdump_asm`), and five warning flags.
Problematic records are flagged rather than removed, so ExeBenchPlus works as a
drop-in replacement for ExeBench and users choose what to filter.

## Splits

| Split | Rows |
|---|---:|
| `train_synth_compilable` | 4,159,373 |
| `train_real_compilable` | 885,074 |
| `train_not_compilable` | 715,700 |
| `train_synth_simple_io` | 640,116 |
| `train_synth_rich_io` | 107,250 |
| `train_real_simple_io` | 42,769 |
| `valid_synth` | 5,000 |
| `valid_real` | 2,133 |
| `test_synth` | 5,000 |
| `test_real` | 2,134 |
| **Total** | **6,564,549** |

The splits and their rows are ExeBench's own; ExeBenchPlus only adds columns.

## Loading

```python
import json
from datasets import load_dataset

test = load_dataset("{{repo_id}}", split="test_real")
# The training splits are large - stream them rather than downloading:
train = load_dataset("{{repo_id}}", split="train_real_compilable", streaming=True)
```

Flat fields (strings and the boolean flags) are ordinary columns. The nested
fields below are stored as JSON strings, because several have no single
fixed type (e.g. a `func_call_errors[arch]` entry is a string, `null`, or a
per-level dict). Decode them with `json.loads`:

```python
JSON_COLUMNS = ["asm", "angha_io_pairs", "real_io_pairs", "angha_iospec", "real_iospec",
                "objdump_asm", "objdump_asm_errors", "func_call", "func_call_errors",
                "data_section", "data_section_errors"]

def decode(row):
    return {k: json.loads(v) if k in JSON_COLUMNS and v is not None else v for k, v in row.items()}

row = decode(test[0])
row["objdump_asm"]["x86-64"]["O2"]     # disassembly of the function, x86-64, -O2
row["data_section"]["ARM64"]["O2"]     # data the function reads, ARM64, -O2
```

Decode per row as above rather than with `Dataset.map`, which would have to fit
the mixed-type fields back into a fixed Arrow type.

The warning flags need no decoding. For example, the rows with none of the four
data-quality flags set (63.85% of the corpus):

```python
clean = test.filter(lambda r: not (r["dummy_udt"] or r["truncated_funcs"]
                                   or r["no_return"] or r["trivial_function"]))
```

## Fields

Architectures are `x86`, `x86-64`, `ARM32`, `ARM64`; optimization levels are
`O0`, `O1`, `O2`, `O3`, `Os`. Every compiled field has a companion `*_errors`
field explaining why an entry is empty.

### Added by ExeBenchPlus (16)

| Field | Type | Description |
|---|---|---|
| `objdump_asm` | JSON `{arch: {level: str}}` | objdump disassembly of the function from the linked binary (`static`/`inline` stripped before compiling, so optimized builds are not empty). |
| `objdump_asm_errors` | JSON `{arch: {level: str\|null}}` | Why an `objdump_asm` entry is empty; `null` = no error. |
| `func_call` | JSON `{arch: {level: str}}` | Assembly of the call site in a driver `main` that calls the function (renamed `func0`) with arguments derived from `argc`/`argv`, so the compiler cannot fold them away. Shows how each argument is passed (P1). Some rows hold the earlier single-build shape `{arch: str}` (one `-O3` build); the snippet is then the same for every level. |
| `func_call_errors` | JSON | Per arch: an error string, `null` (no error), or `{level: error}` when levels differ. |
| `func_call_skip_reason` | str \| null | Why no call site was generated: e.g. `unsupported_param_type:<category>` (struct/union by value, SIMD, `_Complex`), `fname_is_main`, `all_architectures_failed`. `null` = attempted. |
| `data_section` | JSON `{arch: {level: str}}` | Contents of the data sections (`.rodata`, `.data`, `.bss`, …) the function references, from the same binary as `objdump_asm` (P3). Empty when the function reads no such data. |
| `data_section_errors` | JSON `{arch: {level: str\|null}}` | Why a `data_section` entry could not be produced. |
| `unused_param` | bool | The function has a parameter it never uses, which optimized assembly drops (P2). |
| `opt_real_exe_wrapper` | str \| null | `real_exe_wrapper` rewritten to call the function without its unused parameters, so a reduced-arity decompilation can be executed and tested. Set when `unused_param` is true. |
| `opt_real_exe_wrapper_error` | str \| null | Why the wrapper could not be adapted. |
| `opt_angha_exe_wrapper` | str \| null | Same, for `angha_exe_wrapper` (synthetic I/O). |
| `opt_angha_exe_wrapper_error` | str \| null | Why the wrapper could not be adapted. |
| `no_return` | bool | Non-`void` return type, but no `return` statement. |
| `dummy_udt` | bool | A dependency replaces a real user-defined type with a `{int dummy;}` placeholder. |
| `trivial_function` | bool | Trivial body: empty, a bare `return`, or at most two non-blank lines. |
| `truncated_funcs` | bool | `func_def` is truncated upstream: its return type (or part of its name) is missing although `func_head` has it. K&R implicit-`int` definitions are not flagged. |

### Original ExeBench fields (23, unchanged)

| Field | Type | Description |
|---|---|---|
| `path` | str | `owner/repo/path/to/file.c` of the function's GitHub source. |
| `ref` | str | Git ref (branch) of that repository, e.g. `master`. |
| `fname` | str | Function name. |
| `func_def` | str | C definition of the function. |
| `func_head` | str | Function header as extracted from the source. |
| `func_head_types` | str | Function header with each parameter's type and name. |
| `signature` | list[str] | Return type, then the parameter types. |
| `doc` | str \| null | Documentation comment. |
| `asm` | JSON | Compiler-generated assembly per target (`{real\|angha}_gcc_{x86\|arm}_{O0\|O3\|Os}`), each with `pre_asm`, `func_asm`, `post_asm`, `target`. |
| `real_deps` / `angha_deps` | str | Dependencies (types, globals, prototypes) from the real repository / synthesized. |
| `real_exe_wrapper` / `angha_exe_wrapper` | str \| null | C++ harness that runs the function on I/O examples. |
| `real_io_pairs` / `angha_io_pairs` | JSON | Input/output examples. |
| `real_iospec` / `angha_iospec` | JSON | I/O specification used to build the harness. |
| `real_error` / `angha_error` | str \| null | Compilation error, if any. |
| `real_io_error` / `angha_io_error` | str \| null | I/O generation error, if any. |
| `real_io_pairs_are_trivial` / `angha_io_pairs_are_trivial` | bool | All outputs are identical. |

`real_*` fields use the dependencies from the original repository; `angha_*`
fields use synthesized ones (AnghaBench-style).

## Licensing

ExeBenchPlus follows ExeBench's licensing. All C functions keep the original
license of their GitHub repository (identified by `path` and `ref`). The fields
ExeBenchPlus adds are released under the MIT license, as are ExeBench's own
contributions. The code that produces them is released under Apache-2.0.

## Citation

ExeBenchPlus is under anonymous review; its citation will be added here.
Please also cite ExeBench:

```bibtex
@inproceedings{armengol2022exebench,
  title     = {ExeBench: an ML-scale dataset of executable C functions},
  author    = {Armengol-Estap{\'e}, Jordi and Woodruff, Jackson and Brauckmann, Alexander and
               de Souza Magalh{\~a}es, Jos{\'e} Wesley and O'Boyle, Michael F. P.},
  booktitle = {Proceedings of the 6th ACM SIGPLAN International Symposium on Machine Programming},
  pages     = {50--59},
  year      = {2022},
  doi       = {10.1145/3520312.3534867}
}
```
