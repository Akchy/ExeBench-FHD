# ExeBench data quality audit

ExeBench (Armengol-Estapé et al., MAPS 2022) contains real (GitHub) and
synthetic (AnghaBench-style) C functions for ML-based decompilation. Its
pipeline reconstructs missing type and dependency definitions with Psyche-C
(falling back to placeholder `dummy` structs), compiles each function at
several optimization levels, and, where a test harness can be built, runs the
function on generated inputs to record I/O pairs. The training data is split
by how far a function got through this pipeline (`train_not_compilable`,
`train_*_compilable`, `train_*_simple_io`, `train_synth_rich_io`), each
divided into real and synthetic. The evaluation splits `test_real`,
`test_synth`, `valid_real` and `valid_synth` contain only functions with I/O
pairs.

This document collects issues reported in the literature, then reports our
measurements on the full dataset. The fields added to address them are
described in [ExeBench-FHD.md](ExeBench-FHD.md).

## The nine audited issues (P1-P9)

Prevalence over all 6,564,549 functions and over `test_real` (2,134
functions). P3 counts functions with a non-empty data section at `-O2` on at
least one architecture. P4 excludes `main`, which is well defined without a
`return`.

| ID | Issue | Corpus count | Corpus % | `test_real` count | `test_real` % | Addressed by |
|---|---|---:|---:|---:|---:|---|
| P1 | Mixed int/float parameters | 95,573 | 1.46 | 245 | 11.48 | `func_call` |
| P2 | Unused parameter | 496,244 | 7.56 | 158 | 7.40 | `unused_param`, `opt_*_exe_wrapper` |
| P3 | Reads a data section (`-O2`, any arch) | 1,062,170 | 16.18 | 348 | 16.31 | `data_section` |
| P4 | Non-void without return | 113,650 | 1.73 | 4 | 0.19 | `no_return` |
| P5 | Dummy UDT dependency | 608,573 | 9.27 | 11 | 0.52 | `dummy_udt` |
| P6 | `static`/`inline` definition | 2,383,779 | 36.31 | 613 | 28.73 | `objdump_asm` |
| P7 | Function named `main` | 413,822 | 6.30 | 0 | 0.00 | regenerated assembly |
| P8 | Trivial body | 1,827,693 | 27.84 | 585 | 27.41 | `trivial_function` |
| P9 | Truncated definition | 41,099 | 0.63 | 0 | 0.00 | `truncated_funcs` |
| | Any of P1-P9 | 4,659,378 | 70.98 | 1,307 | 61.25 | |

Sections below are tagged with these IDs. Findings that are not one of
P1-P9 (split sizes, non-standard C, train/test overlap) follow them.

## Issues reported in prior work

| Issue | Reported by |
|---|---|
| I/O pairs could be generated for only ~15% of functions, and the test and validation sets are drawn from that subset (selection bias) | ExeBench; Idioms |
| Compilation settings are not published; compiling in a standard way works for about half the samples (443K of 797K train, 2,621 of 5,000 test) | Prior work |
| Much of the code is non-standard and needs `g++ -fpermissive` | QiMeng |
| 32-bit builds conflict with 64-bit assumptions, e.g. `typedef unsigned long size_t;` | CodeInverter |
| The executable subset is simple (mean cyclomatic complexity 2.1, Halstead difficulty 10.5) | ExeBench; Decompile-Bench |
| Many programs are very short | QiMeng |
| Missing header files and unexpanded macros cause compilation errors | ExeBench |
| User-defined types are lost after compilation | Prior work |
| Synthetic type definitions are placeholders with a single `dummy` field | Idioms |
| The Hugging Face viewer is disabled because loading needs a custom script | Hugging Face |

## Data

We streamed every record of every split from the `jordiae/exebench` snapshot
on Hugging Face (v1.01): 6,564,549 functions, without sampling, unless a
section says otherwise. Note that the split on the hub is called
`train_not_compilable` while its archive is named `train_no_compilable`.

### Split sizes do not match the dataset card

| split | records in the snapshot | dataset card | ratio |
|---|---:|---:|---:|
| `train_real_compilable` | 885,074 | 675,074 | 131% |
| `train_synth_compilable` | 4,159,373 | 2,308,373 | 180% |
| `train_real_simple_io` | 42,769 | 43,769 | 98% |
| `train_synth_simple_io` | 640,116 | 550,116 | 116% |
| `train_synth_rich_io` | 107,250 | 97,250 | 110% |
| `train_not_compilable` | 715,700 | 2,357,000 | 30% |

We confirmed the counts in two ways (shard count times 10,000 rows per shard,
and identical totals with single- and multi-threaded decompression). The
differences go in both directions, so they are not explained by a uniform
re-scrape. Anyone quoting split sizes should count the current snapshot.

## Full-corpus metrics

| split | n | has `return` % | non-void, no `return` % | return-only % | empty % | zero-branch % | K&R % | dummy struct %* | GCC ext. % | error % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_real` | 2,134 | 42.50 | 0.19 | 4.69 | 1.27 | 50.84 | 0.00 | 0.52 | 5.20 | 0.09 |
| `test_synth` | 5,000 | 34.56 | 0.30 | 1.18 | 0.00 | 45.20 | 0.02 | 6.22 | 15.46 | 72.60 |
| `valid_real` | 2,133 | 42.01 | 0.28 | 3.56 | 1.27 | 50.35 | 0.00 | 0.42 | 6.09 | 0.09 |
| `valid_synth` | 5,000 | 33.60 | 0.42 | 0.88 | 0.02 | 46.24 | 0.08 | 6.04 | 16.62 | 71.88 |
| `train_real_compilable` | 885,074 | 62.23 | 4.55 | 13.39 | 4.52 | 46.05 | 0.55 | 4.10 | 6.41 | 1.30 |
| `train_synth_compilable` | 4,159,373 | 65.61 | 1.72 | 9.31 | 0.45 | 35.22 | 0.07 | 11.22 | 16.59 | 65.05 |
| `train_real_simple_io` | 42,769 | 71.77 | 0.51 | 32.03 | 19.19 | 63.80 | 0.01 | 0.72 | 3.96 | 0.60 |
| `train_synth_simple_io` | 640,116 | 60.81 | 1.22 | 18.49 | 1.76 | 57.86 | 0.06 | 15.33 | 23.18 | 51.64 |
| `train_synth_rich_io` | 107,250 | 34.18 | 0.36 | 1.10 | 0.04 | 45.74 | 0.04 | 5.95 | 16.47 | 71.71 |
| `train_not_compilable` | 715,700 | 62.91 | 10.50 | 3.23 | 1.85 | 24.86 | 2.73 | n/a | 9.10 | 99.98 |
| **total** | **6,564,549** | **63.85** | **2.98** | **10.09** | **1.39** | **38.14** | **0.42** | | **14.95** | **58.61** |

\* Of records with non-empty `angha_deps` or `real_deps`. "Error" means the
record carries an error string in `real_error`/`angha_error`.

Body length, from an exact per-function line count:

| split | single-line body % | median lines | mean lines |
|---|---:|---:|---:|
| `test_real` | 15.42 | 5 | 9.74 |
| `test_synth` | 21.34 | 5 | 10.11 |
| `valid_real` | 13.92 | 5 | 10.03 |
| `valid_synth` | 21.18 | 5 | 8.90 |
| `train_real_compilable` | 24.02 | 5 | 13.06 |
| `train_synth_compilable` | 17.30 | 8 | 15.78 |
| `train_real_simple_io` | 37.15 | 1 | 4.55 |
| `train_synth_simple_io` | 31.90 | 3 | 6.66 |
| `train_synth_rich_io` | 20.89 | 5 | 9.17 |
| `train_not_compilable` | 11.89 | 15 | 49.95 |

### P8: Trivial functions

About half of the evaluation functions contain no branching keyword. The
problem is worst in `train_real_simple_io`: median body length 1 line, 32%
return-only and 19% empty bodies, presumably because I/O generation succeeds
most easily on trivial functions. `train_not_compilable` is the most complex
split, and its mean is more than three times its median, i.e. it has a long
tail of very large functions. In the synthetic splits, most single-line bodies
are a call or an assignment rather than a bare `return` (`test_synth`: 21.34%
single-line, 1.18% return-only). Flagged by `trivial_function`.

### P4: Non-void functions without `return`

195,297 functions (2.98%) declare a non-`void` return type but contain no
`return` statement. 63.85% of all functions contain a `return`; most of the
rest are `void`.

| split | count | rate |
|---|---:|---:|
| `test_real` | 4 | 0.19% |
| `test_synth` | 15 | 0.30% |
| `valid_real` | 6 | 0.28% |
| `valid_synth` | 21 | 0.42% |
| `train_real_compilable` | 40,281 | 4.55% |
| `train_synth_compilable` | 71,382 | 1.72% |
| `train_real_simple_io` | 217 | 0.51% |
| `train_synth_simple_io` | 7,826 | 1.22% |
| `train_synth_rich_io` | 381 | 0.36% |
| `train_not_compilable` | 75,164 | 10.50% |

These counts include some `main` functions, which are well defined without a
`return`. The `no_return` field excludes them, leaving the 113,650 functions
(1.73%; 4 in `test_real`) counted as P4 above.

### P5: Placeholder types and missing dependencies

4-16% of records with dependency text contain a `{int dummy;}` placeholder in
place of a real type, including standard types (`struct sockaddr`, `in_addr`,
`servent`, `pollfd`). Flagged by `dummy_udt`. `train_not_compilable` has no
dependency text at all and a 99.98% error rate; the most common errors are
missing standard typedefs (`size_t`, `bool`, `u32`) and missing
project-specific structs.

### Non-standard C

GCC-specific constructs (`asm`, `typeof`, statement expressions,
`__extension__`) occur in 5-10% of real code, mostly in kernel and SIMD code.
The higher rate in synthetic splits comes largely from AnghaBench's own
`__attribute__((used))`. K&R definitions are 0.42% overall, absent from every
synthetic split and from the real evaluation splits, and most frequent in
`train_not_compilable` (2.73%). On a 154K-function sample of all splits, no
split has a consistent brace style (20-32% same-line, 55-73% next-line),
and tabs appear in 15-24% of synthetic code and almost never in real code.

### P9: Truncated `func_def`

Classification described in [ExeBench-FHD.md](ExeBench-FHD.md#truncated_funcs):

| class | records | % of corpus |
|---|---:|---:|
| `missing_int_implicit` (valid C89, not flagged) | 72,844 | 1.1097% |
| `missing_non_int` | 2,352 | 0.0358% |
| `fname_not_found` | 38,747 | 0.5902% |

| split | `missing_int_implicit` | `missing_non_int` | `fname_not_found` |
|---|---:|---:|---:|
| `train_not_compilable` | 38,041 (5.3152%) | 1,729 (0.2416%) | 37,270 (5.2075%) |
| `train_synth_compilable` | 19,659 (0.4726%) | 457 (0.011%) | 980 (0.0236%) |
| `train_real_compilable` | 15,144 (1.711%) | 166 (0.0188%) | 494 (0.0558%) |
| `train_synth_simple_io` | 0 | 0 | 2 |
| `train_real_simple_io` | 0 | 0 | 1 |
| other splits | 0 | 0 | 0 |

The problem is almost entirely in `train_not_compilable` (96.2% of
`fname_not_found`, 73.5% of `missing_non_int`) and does not affect the
evaluation splits. Of the 108 distinct missing types, `void` accounts for
1,812 of the 2,352 `missing_non_int` records, followed by `char *` (85),
`Bool` (43), `PNode *` and `double` (32 each).

## Parameters, data and optimized assembly (P1-P3, P6, P7)

### P1: Mixed int/float parameters

On x86-64, float and integer arguments use separate register files, so
parameter order cannot always be recovered from the callee. 95,573 functions
(1.46%) mix the two; they are 11.48% of `test_real`. Addressed by `func_call`.

### P2: Unused parameters

Unused parameters disappear from optimized assembly, so a correct
decompilation can have fewer parameters than the reference, and ExeBench's
fixed-arity test wrappers reject it. 496,244 functions (7.56%; 7.40% of
`test_real`). Addressed by `unused_param` and `opt_*_exe_wrapper`. The
ExeBench-FHD wrapper accepts a prediction that passes either the reduced
(`opt_*_exe_wrapper`) or the full-signature (`*_exe_wrapper`) calling form.

### P3: Referenced data

String literals, globals and lookup tables used by the function are not in
the record. 1,062,170 functions (16.18%; 16.31% of `test_real`) read a
non-empty data section at `-O2` on at least one architecture. Addressed by
`data_section`.

### P6: `static`/`inline` definitions and empty optimized assembly

2,383,779 functions (36.31%; 28.73% of `test_real`) are `static` or `inline`.
About 22-24% of the Os/O3 assembly in every evaluation split contains only
directives; 96.6% of these come from `static` functions, which GCC removes
because nothing calls them. Addressed by `objdump_asm`.

### P7: Functions named `main`

413,822 functions (6.30%) are named `main`, including 15.5% of
`train_real_compilable`; none are in `test_real`. A function named `main`
collides with any driver `main` used to build it.

### Other compilation notes

- **Linking.** Building records as complete programs loses about 48-60% of the
  real-code splits, mainly to undefined references to sibling functions that
  ExeBench did not capture, and partly to the `main` collision above.
- **Dependency selection.** A content-empty `real_deps` still ends in a `#
  1` line marker, which makes naive fallbacks to `angha_deps` fail (see
  [ExeBench-FHD.md](ExeBench-FHD.md#building-a-translation-unit)).

## Train/test overlap

`scripts/check_split_leakage.py` compares every pair of splits by exact
`func_def`, whitespace-normalized `func_def`, repository (`owner/repo` from
`path`) and function name. The name comparison is a weak signal, since common
names recur across unrelated repositories. Each evaluation split against the
union of all six training splits:

| eval split | exact dup. | normalized dup. | repo overlap | shared names |
|---|---:|---:|---:|---:|
| `test_real` | 0 | 0 | 1,130/1,146 (98.60%) | 860 |
| `valid_real` | 0 | 0 | 1,117/1,125 (99.29%) | 826 |
| `test_synth` | 0 | 0 | 2,166/2,168 (99.91%) | 1,928 |
| `valid_synth` | 0 | 1 (0.02%) | 2,131/2,134 (99.86%) | 1,848 |

There is essentially no function-level duplication between training and
evaluation data, and within-split duplication is negligible (378 of
6,564,549 records). Almost every repository in an evaluation split also
contributes to training, however. Synthetic and real splits draw on the same
GitHub repositories; `train_synth_compilable` alone (78,379 repositories)
covers 97-99% of each evaluation split's repositories. For example,
`zimmski/tirion` contributes `mmapClose` and `shmGet` to `test_real` and
`mmapOpen` and `shmSet`, from the same files, to `train_real_compilable`.

The splits also overlap with each other: `train_real_simple_io` is entirely
contained in `train_real_compilable`, `train_synth_simple_io` is 99.98%
contained in `train_synth_compilable`, and `train_synth_compilable` and
`train_real_compilable` share 590,014 identical function bodies.

### Test vs. train complexity

On sampled training shards, evaluation functions have fewer branches than
training functions: a mean of 2.22 branch keywords in `test_real` against
2.84 in `train_real_compilable`, and 2.09 in `test_synth` against 3.69 in
`train_synth_compilable`. Reported evaluation results may therefore
overstate performance on typical code.

## Open issues

These are not addressed by any ExeBench-FHD field.

- **Repository-level overlap.** The evaluation splits test generalization to
  unseen functions, not unseen codebases. Fixing this needs a repository-aware
  re-split. We did not measure near-duplicate (sibling function) similarity:
  in the single-function decompilation setting the model sees no repository
  context, and exact and normalized duplicates are already near zero. This
  should be revisited for tasks that use file or repository context.
- **Straight-line functions.** `trivial_function` covers bodies of at most two
  lines. Longer functions without any branches need no control-flow recovery
  either, and are not flagged.
- **Complexity gap.** Nothing matches the complexity of the evaluation and
  training distributions; the underlying cause, selection by I/O generation
  success, is unchanged.
- **User-defined types in general.** `dummy_udt` marks placeholder types only.
  Recovering arbitrary user-defined types from assembly remains open.
- **Incomplete counts.** There is no full-corpus count for `trivial_function`
  or `dummy_udt`; the `no_return` total above predates the `main` exception.
