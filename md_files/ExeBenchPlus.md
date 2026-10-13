# ExeBenchPlus fields: design and validation

This document describes how each field ExeBenchPlus adds to ExeBench is
computed, why it is computed that way, and how it was checked. The problems
the fields address are measured in [Exebench_issues.md](Exebench_issues.md).

| Field(s) | Type | Script |
|---|---|---|
| `objdump_asm`, `objdump_asm_errors` | per arch, per opt level | `scripts/extract/extract_asm_regen.py` |
| `func_call`, `func_call_errors`, `func_call_skip_reason` | per arch | `scripts/extract/extract_func_call.py` |
| `data_section`, `data_section_errors` | per arch | `scripts/extract/extract_data_sections.py` |
| `unused_param` | bool | `scripts/extract/extract_unused_param.py` |
| `opt_real_exe_wrapper`, `opt_angha_exe_wrapper` (+ `_error`) | str | `scripts/extract/adapt_unused_param_wrapper.py` |
| `no_return` | bool | `scripts/extract/extract_non_void_no_return.py` |
| `dummy_udt` | bool | `scripts/extract/extract_dummy_struct_types.py` |
| `trivial_function` | bool | `scripts/extract/extract_trivial_function.py` |
| `truncated_funcs` | bool | `scripts/extract/extract_truncated_funcs.py` |

Every extractor reads records from the raw ExeBench snapshot through
`scripts/check_split_leakage.py::iter_split` and can be run on its own (each
has `--help`). `scripts/exebenchplus_pipeline` runs all of them except
`extract_asm_regen.py` on each record and writes one merged record; see the
README for usage.

The boolean flags are kept on every record rather than used to drop records,
so that users can filter, cap or weight as suits their task. `dummy_udt` and
`truncated_funcs` mark records whose ground truth is unreliable, and we
recommend filtering on them for type or signature recovery.

## Shared machinery

### Architectures and toolchain

The compiled fields cover four targets: x86 (32-bit), x86-64, ARM32 and ARM64
(`ARCH_TOOLCHAIN` in `extract_func_call.py`). On Debian/Ubuntu, `gcc-multilib`
cannot be installed alongside the ARM cross compilers: it reinstalls a
multilib `gcc-13` that conflicts with the cross `gcc-13` packages through the
alternatives system, and apt silently removes whichever was installed first.
`libc6-dev-i386` alone provides what 32-bit x86 linking needs (crt objects and
32-bit libc) without that conflict:

```bash
sudo apt-get install -y libc6-dev-i386 gcc-arm-linux-gnueabihf gcc-aarch64-linux-gnu
```

With these packages all four targets build in a single pass.

### Building a translation unit

ExeBench stores one function per record, together with dependency text
(`real_deps` or `angha_deps`). To compile a record we concatenate the
dependencies and `func_def`, rename the target to `func0`, and append a small
driver `main` so that the program links.

**Choosing dependencies.** Every non-null `real_deps` value ends in a dangling
`# 1` GCC line marker (100% of a 20K-record sample on each of
`train_real_compilable` and `train_synth_compilable`). A `real_deps` with no
other content is therefore a non-empty string, and the obvious
`rec.get("real_deps") or rec.get("angha_deps")` picks it over a populated
`angha_deps`, dropping typedefs and prototypes the function needs. On
`train_real_compilable`, 31.81% of records have a content-empty `real_deps`,
and falling back to `angha_deps` recovers real content for 14.04% of the split
(124,225 records). The figure is 3.00% for `train_synth_compilable` (124,581
records) and 1.03-1.31% for `test_real`/`valid_real`.
`scripts/regen_asm.py::pick_deps` strips the trailing marker before the
emptiness check and falls back to `angha_deps` only if nothing remains. It
never concatenates the two, since that causes conflicting libc prototypes.

**Functions named `main`.** 15.5% of `train_real_compilable` (138,220 records)
are functions literally named `main`; in `test_real` there are none. Appending
a driver `main` to such a record is a compile-time redefinition error, so a
switch to `gcc -c` or `gcc -S` does not help. `scripts/preprocess/get_asm.py`
renames every target to `func0` before appending the driver, `main` included,
and after extraction renames `func0` back to `main` in the assembly of these
records (the label and any recursive calls). Running the production functions
over all 138,220 records:

| | before | after |
|---|---|---|
| redefinition of `main` | 138,144 (99.95%) | 0 |
| compiles | 76 (0.05%) | 68,011 (49.2%) |

The 76 records that compiled before the fix have a `func_def` truncated
upstream (e.g. `"in(){"`), so they never defined `main` in the first place.
49.2% matches the corpus-wide link success rate. On 500 `main`-named and 500
other records, end-to-end extraction (compile, objdump, find label) went from
0% to about 50-53% for both groups. The 0% for ordinary records was a separate
bug found in the same investigation: before the fix nothing renamed the
compiled symbol to `func0`, so the label search in `get_processed_asm` never
matched. `scripts/verify/` contains the scripts used for these checks.

The rename is a regex substitution over the translation unit, not a C parse.
A call-shaped occurrence of the name inside a string literal or comment would
also be renamed. We have not seen this in the corpus.

## `objdump_asm`: assembly with `static`/`inline` removed

**Problem.** ExeBench compiles each function on its own, with no caller in the
translation unit. A `static` function, or a C99 `inline` function (which emits
no external definition), is then unreachable, and GCC deletes it at `-O1` and
above. About 22-24% of the Os/O3 assembly in every evaluation split contains
only directives, and 96.6% of these come from `static` functions. ExeBench's
extractor still adds a `.globl` header, so the empty entries look well-formed.

**Fix.** `extract_asm_regen.py` strips leading `static`/`inline` specifiers
(`regen_asm.py::strip_static`), so the symbol is externally visible and must be
kept, and wraps the target in `__attribute__((noinline))` so that a call site
inside the translation unit (a recursive call, or a caller in the
dependencies) cannot inline the body away. For the common case with no
in-unit caller the `noinline` attribute produced byte-identical output, so it
is applied unconditionally. Each record is compiled and linked for all four
targets at `-O0`, `-O1`, `-O2`, `-O3` and `-Os`, and the target's block is cut
from `objdump -d` output with addresses and opcode bytes kept. A `main`-named
target is disassembled under its own name with no driver appended.

**Cost.** Measured on 10 `static` records from `train_real_compilable`:

| | bytes per record | full corpus (6.5M rows) |
|---|---:|---:|
| original ExeBench fields | 1,390 | 9.0 GB |
| + single-arch `gcc -S` text (`asm`) | 4,455 | 29.0 GB |
| + `objdump_asm` (4 archs x 5 levels) | 19,707 | 128.1 GB |

For this reason `objdump_asm` is opt-in (`--with-objdump-asm`) and is not
computed by the unified pipeline. Without the flag, the script still writes a
regenerated single-architecture `asm[O0..O3]` (`gcc -S` text) and
`asm_was_static`. That `asm` key has the same name as ExeBench's own `asm`
field but a different shape, so drop it before merging this output into an
ExeBench record.

**Checks.** For `static int dsu(int a, int b){ int x = a*(b-1); return x; }`,
the regenerated `asm` has real instructions at every level (13 at O0, 4 at
O1-O3), where ExeBench's pipeline produces only directives above O0, and
`objdump_asm` has addressed disassembly on all four targets at all five
levels. On 10 `static` records from `train_real_compilable`, all 10 produced
non-empty `asm` at every level.

## `func_call`: the call site

**Problem.** On x86-64, floating-point arguments are passed in `xmm`
registers and integer and pointer arguments in general-purpose registers. When
the two are mixed, or when an argument is unused, the function body alone does
not show the declared parameter order. The caller's argument setup does, but
ExeBench records have no caller.

**Why a naive driver does not work.** Take

```c
int dsu(int a, int b) { int x; x = a*(b-1); return x; }
int main() { int l = 5, m = 6, p; p = dsu(l, m); return 0; }
```

At `-O0` the disassembly of `main` contains `call dsu`. At `-O1` and above
`main` reduces to `mov $0x0,%eax; ret`, for two reasons:

1. Dead code elimination: `p` is never read and `dsu` has no side effects, so
   the call is removed.
2. Inlining and constant folding: even if `p` is printed, `l` and `m` are
   constants, so GCC inlines `dsu` and replaces the call with `25`.

Both mechanisms have to be blocked. `noinline` alone still lets GCC evaluate
the call at compile time; runtime-unknown inputs alone still let it inline the
body. With `__attribute__((noinline))` on the callee and arguments derived from
`argc`/`argv`, `call dsu` is present from `-O0` to `-O3`.

**Construction.** `extract_func_call.py` writes a driver that calls the
renamed target with `__attribute__((noipa))` (falling back to `noinline` where
unsupported) and arguments read from `argv`: integer and floating-point
scalars via `strtoll`/`strtod` and a cast, and any pointer type as
`(T)(void*)argv[i]`. The driver is only compiled and disassembled, never run,
so pointers need not be valid. Variadic functions are called with their fixed
arguments only. The recorded snippet is the call-site window of the driver's
disassembly. In the released dataset it is stored per architecture and
optimization level; the script's `--levels` option produces that shape, and
without it a single `-O3` build is used.

**Coverage and skips.** A survey of parameter types
(`scripts/survey/survey_func_call_param_types.py`) found that scalars and
pointers cover 98.88% of parameters in `train_real_compilable`. Functions with
any parameter outside these classes (struct or union by value, SIMD and
`_Complex` types, unresolvable typedefs; about 0.50%) are skipped as a whole,
as are `main`-named functions, records without a signature, and records where
every requested architecture fails. `func_call_skip_reason` records why.
`func_call[arch]` is `""` when that architecture was attempted and failed, and
`func_call_errors[arch]` gives the reason.

On a 10-record sample of `train_real_compilable`, 6 succeeded on all four
targets, 3 failed on all four because they call a sibling function that
ExeBench did not include in the record, and 1 (`outsl`, x86 port I/O inline
assembly) failed on ARM only, as expected.

## `data_section`: referenced data

**Problem.** The instructions in `func_asm` refer to values stored elsewhere:
string literals, globals, `switch` lookup tables. ExeBench does not record
these bytes, so a symbol reference in the assembly cannot be resolved to a
value.

**Where data ends up.** We compiled small test programs with `gcc`,
`arm-linux-gnueabihf-gcc` and `aarch64-linux-gnu-gcc` at `-O0` and `-O2`.
Placement was the same on all four targets:

| C construct | Section |
|---|---|
| initialized global (`float g = 3.14159f;`) | `.data` |
| string literal bytes | `.rodata` |
| pointer to a string literal (`char *s = "..."`) | `.data.rel.local` (needs a load-time relocation) |
| uninitialized or zero-initialized global | `.bss` (no bytes on disk) |
| `const int t[4] = {...}` | `.rodata` |
| `const` table of function pointers | `.data.rel.ro` |
| `__thread int x = 42;` / `__thread int y;` | `.tdata` / `.tbss` |
| `switch` compiled to a lookup table | `.rodata` |

Two consequences. String literals live in `.rodata`, not `.data`. And a float
constant need not be in memory at all: for `74454.235f` (`0x47916b1e`), x86-64
and x86 load the value from `.rodata`, while ARM32 (`movw`/`movt`) and ARM64
(`mov`/`movk`) build it in a register from immediates.

**Why the linked binary.** A first version took data directives from
`gcc -S` output. That is wrong for any value that needs relocation: the `.s`
text holds a symbolic reference (`.quad .LC0`) and an unlinked `.o` holds
zeros plus a relocation record. Only the linked executable contains the
resolved value. In a test build, the 8 bytes of `global_str` on x86-64 were
`04 20 00 00 00 00 00 00`, i.e. `0x2004`, the address of the string in that
binary's `.rodata`; the other three targets resolved likewise.

`extract_data_sections.py` therefore, per record and architecture:

1. compiles to an object file and links it with `-Wl,-Map=...` to get a GNU ld
   link map;
2. reads from the map every byte range that the target's own object file (not
   crt or libc) contributed to a data-like output section (`.data*`,
   `.rodata*`, `.bss*`, `.tdata`, `.tbss`); ld moves the address and size to a
   separate line when the section name is long, e.g. `.data.rel.local`, and
   the parser accepts both forms;
3. runs `objdump -s` once and slices those ranges from the hex dump. A section
   without contents in that output is NOBITS (`.bss`/`.tbss`) and is reported
   as `"N bytes, zero-initialized"`.

`data_section[arch]` is `""` both when a function references no data
(`data_section_errors[arch]` is `None`) and when the build failed (the error
field gives the reason), so check the error field to tell them apart.

**Failure rate.** `gcc -S` never resolves external symbols, but a link does,
so any function calling an uncaptured sibling now fails to link. On 100
`train_real_compilable` records the non-empty rate was 23-33% (38-51% for the
`.s`-based version) and the build-failure rate 32-48% (1-21% before). Every
new failure we inspected was an `undefined reference` at link time.

**Non-determinism.** Each build runs in a fresh temporary directory. In
functions that call `assert()`, `__FILE__` expands to the driver's path, which
then appears as bytes in `.rodata`. For these few records `data_section`
differs between runs.

## `unused_param` and `opt_*_exe_wrapper`

**Problem.** A parameter the body never uses leaves no trace in optimized
assembly, especially an unused leading `double`/`float` argument, which leaves
no gap in the integer registers. A correct decompilation from such assembly
has fewer parameters than the original.

**Why trimming test inputs is not enough.** A first attempt removed the unused
parameter's value from every I/O pair (`opt_real_io_pairs`). When run through
ExeBench's harness this failed: `real_exe_wrapper` is C++ generated from the
original function and calls it with a fixed argument list, so a two-argument
prediction for a three-argument function fails to compile (`too many arguments
to function`) before any I/O pair is read. The wrapper itself has to change.
Once it does, the I/O pairs can stay as they are: the adapted wrapper never
reads the dropped key from the input JSON, and `scripts.exebench.diff_io`
compares only keys present in the observed output. We checked this end to end
(compile, run, `diff_io`) on a real record with an unused scalar input and an
unused pointer that the wrapper echoes to its output, using the unmodified
`real_io_pairs`.

**Detection** (`extract_unused_param.py`). Parameter names are parsed from
`func_head_types` (falling back to `func_head`) with a small, parenthesis-aware
declarator splitter. `signature` is not used: it has types only, and on a 100K
sample of `train_real_compilable` it drops or merges parameters for 1.87% of
records (function-pointer parameters, multidimensional or VLA arrays, a
leading `__attribute__` or `__typeof__`). A parameter counts as used if its
name occurs as a whole word anywhere in the body. This errs towards "used",
since a name in a comment or a shadowing local still counts. Keeping an unused
parameter is harmless, while removing a used one would break execution.
Records with `fname == "main"` are never flagged. Two parser bugs that caused
false positives were fixed: a comment inside a declarator was read as the
parameter name, and an integer suffix such as the `L` in `16L` inside an
array dimension was read as an identifier.

**Wrapper adaptation** (`adapt_unused_param_wrapper.py`). The script rewrites
the call site, the code that reads each argument from the input JSON, and the
`write_output` helper, which echoes pointer and array arguments back into the
output. Supported shapes, all found by running
`scripts/survey/survey_unused_param_wrappers.py` over the corpus: plain
scalars, `.get<T>()` casts, strings via `strdup`, arrays read in a loop into a
`std::vector`, struct pointers and structs by value built from per-field keys,
and the corresponding echo blocks. When `fname` clashes with a libc/libm name
(`pow`, `puts`, `mktime`, ...), ExeBench's generator calls `<fname>__bench`,
and this is handled too. A final check makes sure no removed name is still
referenced in `write_output`. Unrecognized shapes raise `WrapperAdaptError`;
`opt_*_exe_wrapper` is then `""` and `opt_*_exe_wrapper_error` holds the
reason. The original `real_exe_wrapper`/`angha_exe_wrapper` are never modified.
`unused_param` and the adapted wrappers are produced on every split.

**Results.** On 100 records from each of the 10 splits, 8.2% were flagged
(4-13% per split). The wrapper survey reached 1,000/1,000 on
`train_real_compilable` and 999/1,000 on `train_synth_compilable`. On the full
evaluation splits:

| Split | Records | `unused_param` | `opt_real_exe_wrapper` adapted | `opt_angha_exe_wrapper` adapted |
|---|---:|---:|---:|---:|
| `test_real` | 2,134 | 158 | 158/158 | 7/7 |
| `valid_real` | 2,133 | 156 | 156/156 | 7/7 |
| `test_synth` | 5,000 | 296 | n/a | 296/296 |
| `valid_synth` | 5,000 | 312 | n/a | 311/312 |

Two known failures are left as reported errors: `get_field_string`, whose
wrapper renamed its `_buffer` parameter to `static_buffer` in the JSON keys,
and `i8086UncondJumpDS`, whose by-value struct parameter contains an array
field.

**Use in evaluation.** `scripts/exebench_eval.py::select_exe_wrapper(row,
opt_level, track)` returns `opt_{track}_exe_wrapper` when `unused_param` is
true, the optimization level is not `O0` and the adapted wrapper is
non-empty, and the original wrapper otherwise. A missing level is treated as
`O0`. `exebench_validate_new` takes an `opt_level` argument or reads a
per-item `opt` field; with neither, it behaves as before.
`scripts/exebenchplus_eval.py` does the same for the released ExeBenchPlus record format.
`python -m scripts.examples.run_sample_eval` runs both paths on a two-record
example.

## `no_return`

True when `signature[0]` is non-empty and not `void`, and `func_def` contains
no `return` keyword. Calling such a function and using its result is
undefined behaviour, so there is no well-defined ground truth. `main` is never
flagged, because falling off the end of `main` returns 0 (C99 5.1.2.2.3).

Over all 6,564,549 records, 195,297 (2.98%) were flagged, from 0.19-0.42% in
the evaluation splits to 4.55% in `train_real_compilable` and 10.50% in
`train_not_compilable` (per-split counts in
[Exebench_issues.md](Exebench_issues.md)). This count was taken before the
`main` exception was added and so slightly overcounts. It has not been rerun
at full scale; `scripts/survey/survey_non_void_no_return.py` reproduces it.

## `dummy_udt`

When Psyche-C cannot resolve a type, ExeBench's dependency text contains a
placeholder with a single field, e.g. `struct sockaddr {int dummy; };`. This
also happens for standard types such as `sockaddr` and `pollfd`. Such records
have no real type definition to recover. The detector matches a struct or
union whose body is exactly `{ int dummy; }`, so a real struct that happens to
have a field called `dummy` is not flagged. It checks both `real_deps` and
`angha_deps`, although in a 200K-record scan across the main splits every
occurrence was in `angha_deps`, including on real-track records with a
normal `real_deps`.

Matches when scanning each split until 100 were found (or the split ended):

| split | scanned | matched | rate |
|---|---:|---:|---:|
| `train_not_compilable` | 715,700 (all) | 0 | 0% |
| `train_synth_compilable` | 992 | 100 | ~10.1% |
| `train_real_compilable` | 2,032 | 100 | ~4.9% |
| `train_synth_simple_io` | 681 | 100 | ~14.7% |
| `train_real_simple_io` | 15,554 | 100 | ~0.64% |
| `train_synth_rich_io` | 1,440 | 100 | ~6.9% |
| `valid_synth` | 2,026 | 100 | ~4.9% |
| `valid_real` | 2,133 (all) | 9 | 0.42% |
| `test_synth` | 1,366 | 100 | ~7.3% |
| `test_real` | 2,134 (all) | 11 | 0.52% |

`train_not_compilable` has no dependency text at all, so it cannot contain a
placeholder. The field was originally named `dummy_struct`; the detection
function keeps that name.

## `trivial_function`

True when the body is empty, a bare `return` statement, or at most two
non-blank lines. These form nested sets:
empty ⊆ return-only ⊆ ≤2 lines ⊆ zero-branch. The flag covers the first three.
Straight-line functions longer than two lines are not flagged (see the open
issues in [Exebench_issues.md](Exebench_issues.md)). Detection reuses
`extract_zero_branch_triviality.py::classify`, a regex analysis of the
brace-matched body with comments removed.

On the whole of `test_real`, 585 of 2,134 records (27.41%) are flagged; on a
5,000-record sample of `train_real_simple_io`, 3,192 (63.84%). We have not run
a full-corpus count.

## `truncated_funcs`

Some `func_def` values have lost their return type, or part of the name,
compared with `func_head` and `signature`.
`extract_missing_return_type.py::classify` locates the function name followed
by `(` in `func_def`, also allowing redundant parentheses around the name
(`int (fname)(...)`) and GCC line markers between the type, the name and the
parameter list (seen in glibc and Linux tracepoint code). If nothing but
storage-class specifiers and attributes precedes the name, the record falls in
one of three classes:

- `missing_int_implicit`: valid C89 implicit `int`, including K&R definitions
  (`main()\n{...}`). Not an error.
- `missing_non_int`: the real type is something implicit `int` cannot produce.
  For example, `func_head` is `void sUalrm()` while `func_def` starts
  `" sUalrm(){"`, with a space where `void` was.
- `fname_not_found`: the name itself is cut, e.g. `format_sector` stored as
  `rmat_sector(...`.

`truncated_funcs` is true for the last two classes: 41,099 records, 0.6260% of
the corpus. None are in the evaluation splits. Both are unusable as ground
truth. `missing_non_int` records compile at the normal rate, but GCC treats
the function as returning `int` (`warning: return type defaults to 'int'`), so
the assembly does not match the declared signature. `fname_not_found` records
compile in 36-50% of samples, but extraction always fails, because the
truncated name also defeats the rename to `func0`. Per-split counts are in
[Exebench_issues.md](Exebench_issues.md).
