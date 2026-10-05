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
    path: train_synth_compilable-*.parquet
  - split: train_real_compilable
    path: train_real_compilable-*.parquet
  - split: train_synth_simple_io
    path: train_synth_simple_io-*.parquet
  - split: train_real_simple_io
    path: train_real_simple_io-*.parquet
  - split: train_synth_rich_io
    path: train_synth_rich_io-*.parquet
  - split: train_not_compilable
    path: train_not_compilable-*.parquet
  - split: valid_synth
    path: valid_synth-*.parquet
  - split: valid_real
    path: valid_real-*.parquet
  - split: test_synth
    path: test_synth-*.parquet
  - split: test_real
    path: test_real-*.parquet
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

ExeBenchPlus keeps the 16 columns that `load_dataset("jordiae/exebench")`
returns, with the same names, types and values, and adds 16: the assembly
of a call to the function (`func_call`, P1), wrappers that test a
reduced-arity decompilation (`opt_*_exe_wrapper`, P2), data-section contents
(`data_section`, P3), disassembly of the linked binary on four architectures
at five optimization levels (`objdump_asm`), and five warning flags.
Problematic records are flagged rather than removed, so ExeBenchPlus works as a
drop-in replacement for ExeBench: code written against `jordiae/exebench`
runs unchanged, and users choose what to filter.

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
from datasets import load_dataset

# Pass data_files so only that split's shards are downloaded
# (split= alone downloads every split before selecting one).
test = load_dataset("{{repo_id}}", data_files="test_real-*.parquet", split="train")
# The training splits are large - stream them rather than downloading:
train = load_dataset("{{repo_id}}", split="train_real_compilable", streaming=True)
```

All columns are typed Arrow columns; no decoding is needed for the
ExeBenchPlus fields:

```python
row = test[0]
row["objdump_asm"]["x86-64"]["O2"]     # disassembly of the function, x86-64, -O2
row["data_section"]["ARM64"]["O2"]     # data the function reads, ARM64, -O2
row["func_call"]["ARM32"]["O3"]        # call site of the function, ARM32, -O3
```

As in ExeBench, `real_iospec` / `synth_iospec` and each I/O-pair `value` are
JSON strings (`json.loads` them).

The warning flags are plain booleans. For example, the rows with none of the four
data-quality flags set (63.85% of the corpus):

```python
clean = test.filter(lambda r: not (r["dummy_udt"] or r["truncated_funcs"]
                                   or r["no_return"] or r["trivial_function"]))
```

## Fields

Architectures are `x86`, `x86-64`, `ARM32`, `ARM64`; optimization levels are
`O0`, `O1`, `O2`, `O3`, `Os`. Per-architecture fields are structs
`{arch: {level: str}}` with all 20 entries present; an entry is `null`
when it does not apply or could not be produced, and the companion `*_errors`
field explains why. `func_call` and `func_call_errors` are `null` as a whole
when no call site was generated (see `func_call_skip_reason`).

### Added by ExeBenchPlus (16)

| Field | Type | Description |
|---|---|---|
| `objdump_asm` | `{arch: {level: str}}` | objdump disassembly of the function from the linked binary (`static`/`inline` stripped before compiling, so optimized builds are not empty). |
| `objdump_asm_errors` | `{arch: {level: str}}` | Why an `objdump_asm` entry is empty; `null` = no error. |
| `func_call` | `{arch: {level: str}}` | Assembly of the call site in a driver `main` that calls the function (renamed `func0`) with arguments derived from `argc`/`argv`, so the compiler cannot fold them away. Shows how each argument is passed (P1). |
| `func_call_errors` | `{arch: {level: str}}` | Why a `func_call` entry is empty; `null` = no error. |
| `func_call_skip_reason` | str \| null | Why no call site was generated: e.g. `unsupported_param_type:<category>` (struct/union by value, SIMD, `_Complex`), `fname_is_main`, `all_architectures_failed`. `null` = attempted. |
| `data_section` | `{arch: {level: str}}` | Contents of the data sections (`.rodata`, `.data`, `.bss`, …) the function references, from the same binary as `objdump_asm` (P3). Empty when the function reads no such data. |
| `data_section_errors` | `{arch: {level: str}}` | Why a `data_section` entry could not be produced; `null` = no error. |
| `unused_param` | bool | The function has a parameter it never uses, which optimized assembly drops (P2). |
| `opt_real_exe_wrapper` | str \| null | `real_exe_wrapper` rewritten to call the function without its unused parameters, so a reduced-arity decompilation can be executed and tested. Set when `unused_param` is true. |
| `opt_real_exe_wrapper_error` | str \| null | Why the wrapper could not be adapted. |
| `opt_angha_exe_wrapper` | str \| null | Same, for `synth_exe_wrapper` (synthetic I/O). |
| `opt_angha_exe_wrapper_error` | str \| null | Why the wrapper could not be adapted. |
| `no_return` | bool | Non-`void` return type, but no `return` statement. |
| `dummy_udt` | bool | A dependency replaces a real user-defined type with a `{int dummy;}` placeholder. |
| `trivial_function` | bool | Trivial body: empty, a bare `return`, or at most two non-blank lines. |
| `truncated_funcs` | bool | `func_def` is truncated upstream: its return type (or part of its name) is missing although `func_head` has it. K&R implicit-`int` definitions are not flagged. |

### ExeBench fields (16, unchanged)

These are the columns of `load_dataset("jordiae/exebench")`, in the same order
and with the same types.

| Field | Type | Description |
|---|---|---|
| `path` | str | `owner/repo/path/to/file.c` of the function's GitHub source. |
| `func_def` | str | C definition of the function. |
| `func_head` | str | Function header as extracted from the source. |
| `func_head_types` | str | Function header with each parameter's type and name. |
| `fname` | str | Function name. |
| `signature` | list[str] | Return type, then the parameter types. |
| `asm` | list[{`target`, `code`}] | Compiler-generated assembly of the function per target (`{real\|angha}_gcc_{x86\|arm}_{O0\|O3\|Os}`). |
| `synth_deps` / `real_deps` | str | Dependencies (types, globals, prototypes): synthesized (AnghaBench-style) / from the original repository. |
| `synth_io_pairs` / `real_io_pairs` | list[{`input`, `output`, `dummy_funcs`, `dummy_funcs_seed`}] | Input/output examples; `input` and `output` are lists of {`var`, `value`} with `value` JSON-encoded. |
| `synth_exe_wrapper` / `real_exe_wrapper` | str \| null | C++ harness that runs the function on I/O examples. |
| `ref` | str | Git ref (branch) of that repository, e.g. `master`. |
| `synth_iospec` / `real_iospec` | str (JSON) | I/O specification used to build the harness. |

`real_*` fields use the dependencies from the original repository; `synth_*`
fields use synthesized ones. (`opt_angha_exe_wrapper*` keeps the raw ExeBench
`angha` prefix but pairs with `synth_exe_wrapper`.)

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
