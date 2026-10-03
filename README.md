# ExeBenchPlus

Code for ExeBenchPlus, an audited and extended version of
[ExeBench](https://github.com/jordiae/exebench), a dataset of executable C
functions for training and evaluating neural decompilers.

We audited all 6,564,549 functions in ExeBench's ten splits. Three of the
problems we found mean that a function's C source cannot be fully recovered
from its assembly: parameter order is ambiguous when integer and
floating-point parameters are mixed, unused parameters vanish from optimized
code, and constants referenced by the function are not part of the record.
ExeBenchPlus keeps the 23 original ExeBench fields unchanged and adds 16
fields that supply the missing information or flag unreliable records. Records
are flagged, not removed, so the dataset can replace ExeBench directly.

This repository contains the extraction pipeline that computes the new fields,
the audit scripts, the evaluation harness, the conversion and upload scripts
for the Hugging Face release, and the LLM experiment from the paper's
appendix. The dataset card, including loading instructions, is in
[`dataset_card/README.md`](dataset_card/README.md).

## Added fields

| Field | Content |
|---|---|
| `objdump_asm` (+ `_errors`) | Disassembly of the function from a linked binary for x86, x86-64, ARM32 and ARM64 at O0, O1, O2, O3 and Os, with `static`/`inline` removed so optimized builds are not empty |
| `func_call` (+ `_errors`, `_skip_reason`) | Disassembly of a call to the function with runtime-unknown arguments, showing how each argument is passed |
| `data_section` (+ `_errors`) | Contents of the data sections (`.rodata`, `.data`, `.bss`, ...) the function contributes to the linked binary |
| `unused_param` | The function has a parameter its body never uses |
| `opt_real_exe_wrapper`, `opt_angha_exe_wrapper` (+ `_error`) | ExeBench's test wrappers rewritten without the unused parameters, so a reduced-arity decompilation can be tested |
| `no_return` | Non-`void` return type but no `return` statement |
| `dummy_udt` | A dependency replaces a real type with a `{int dummy;}` placeholder |
| `trivial_function` | Empty body, bare `return`, or at most two lines |
| `truncated_funcs` | `func_def` has lost its return type or part of its name upstream |

- [`md_files/ExeBenchPlus.md`](md_files/ExeBenchPlus.md): how each field is
  computed, design decisions, and validation results.
- [`md_files/Exebench_issues.md`](md_files/Exebench_issues.md): the audit of
  ExeBench, with per-split statistics, train/test overlap, and open issues.

## Requirements

- Linux and Python 3 (tested with 3.12).
- `gcc`, `g++`, `objdump` and `zstd` on `PATH`.
- For the four-architecture fields (`objdump_asm`, `func_call`,
  `data_section`):

  ```bash
  sudo apt-get install -y libc6-dev-i386 gcc-arm-linux-gnueabihf gcc-aarch64-linux-gnu
  ```

  Do not install `gcc-multilib`; on Debian/Ubuntu it conflicts with the ARM
  cross compilers (see [`md_files/ExeBenchPlus.md`](md_files/ExeBenchPlus.md#architectures-and-toolchain)).
- Python packages: the extraction, audit and evaluation code uses only the
  standard library. Tests need `pytest`; `scripts/release` needs `pyarrow`,
  `pyyaml` and `huggingface_hub`; `scripts/survey` uses `pyarrow` for reading
  Arrow shards; `scripts/experiments` needs `openai`.
- The raw ExeBench snapshot (`jordiae/exebench` on Hugging Face). Scripts look
  for it in the Hugging Face hub cache (`HF_HUB_CACHE`, else
  `$HF_HOME/hub`, else `~/.cache/huggingface/hub`); set `EXEBENCH_SNAPSHOT`
  to point at a different directory.

All commands below are run from the repository root.

## Repository layout

```
.
├── README.md
├── pytest.ini
├── dataset_card/README.md   Hugging Face dataset card
├── md_files/                design notes and audit (see above)
├── scripts/
│   ├── exebenchplus_pipeline/   runs all per-record extractors and merges the fields
│   ├── extract/                 one extractor per field, each usable on its own
│   ├── preprocess/              shared compile, disassemble and asm-cleaning helpers
│   ├── exebench/                ExeBench's evaluation harness (Exebench_Wrapper, diff_io)
│   ├── survey/                  corpus-wide statistics behind the audit
│   ├── verify/                  large-scale checks of individual fixes
│   ├── release/                 JSONL to Parquet conversion and Hugging Face upload
│   ├── experiments/             LLM decompilation experiment (paper appendix A.6)
│   ├── examples/                small runnable evaluation example with sample data
│   ├── check_split_leakage.py   split reader (iter_split) and train/test overlap audit
│   ├── regen_asm.py             static/inline stripping and dependency selection
│   ├── exebench_eval.py         evaluation for the Hugging Face ExeBench format
│   ├── exebench_eval_raw.py     evaluation for the raw/release record format
│   └── eval_utils.py
└── tests/
    ├── unit/
    └── integration/             compile with the real toolchains
```

## Computing the fields

`scripts.exebenchplus_pipeline` adds `no_return`, `dummy_udt`,
`trivial_function`, `truncated_funcs`, `unused_param`, `opt_*_exe_wrapper`,
`func_call` and `data_section` to every record of a split:

```bash
# 20 records of one split
python -m scripts.exebenchplus_pipeline --split test_real --n 20 \
    --out output_files/test_real_sample.jsonl

# a whole split, 32 worker processes, resumable
python -m scripts.exebenchplus_pipeline --split test_real --all --workers 32 --resume \
    --out output_files/test_real.jsonl

# every split, one file per split in --out-dir
python -m scripts.exebenchplus_pipeline --split all --all --workers 32 --resume \
    --out-dir output_files/exebenchplus
```

Main options (`--help` lists all):

| Option | Default | Meaning |
|---|---|---|
| `--split` | `train_real_compilable` | a split name, or `all` |
| `--n` / `--all` | `20` | process the first N records, or the whole split |
| `--limit` | none | stop after scanning this many records per split |
| `--archs` | all four | subset of `x86,x86-64,ARM32,ARM64` |
| `--workers` | `1` | worker processes; records are independent |
| `--resume` | off | continue an interrupted run with the same `--out` |
| `--work-dir` | `output_files/exebenchplus_pipeline_work` | scratch directory for build files |
| `--keep-artifacts` | off | keep the generated `.c` and binary files |

Each record takes about 8 compiler and objdump invocations, roughly 0.8 s when
run sequentially. With `--workers` we measured speedups of 7.8x (100 records,
32 workers) to 9.4x (20 records, 16 workers); output order is not preserved.
With `--resume`, completed record indices are written to `<out>.progress`
after each output line. Rerunning the same command trims any partly written
record and skips completed ones; keep `--n`, `--all`, `--limit` and `--archs`
unchanged between runs. With `--split all`, resume works per split.

`objdump_asm` is not part of the pipeline because it is large (about 11 times
the size of the original record). It is produced by:

```bash
python -m scripts.extract.extract_asm_regen --split test_real --n 20 \
    --with-objdump-asm --out output_files/test_real_objdump.jsonl
```

This script also writes its own `asm` key, which has a different shape from
ExeBench's `asm` field; drop it before merging with an ExeBench record.

Each script in `scripts/extract/` can also be run alone to compute one field
(`python -m scripts.extract.<name> --help`).

## Evaluation

`scripts/exebench_eval.py` re-compiles and re-executes predicted C functions
against ExeBench's I/O pairs. `select_exe_wrapper` uses `opt_*_exe_wrapper`
for records with `unused_param` set when the prediction comes from optimized
assembly. A self-contained example with two records:

```bash
python -m scripts.examples.run_sample_eval
```

For predictions on rows in the raw/release JSONL format:

```bash
python -m scripts.exebench_eval_raw --predictions preds.json \
    --dataset test_real.jsonl --model-name my_model --opt O2 --workers 16
```

The predictions file is updated in place with re-compile and re-execute
results; logs go to `output_files/<model-name>/`.

## Audit

```bash
python -m scripts.check_split_leakage --out output_files/leakage_report.json
```

compares all ten splits by exact and normalized function body, repository and
function name. The other corpus statistics come from the scripts in
`scripts/survey/` and from the extractors' sample modes.

## Release and experiment

```bash
# JSONL splits -> Parquet shards
python -m scripts.release.convert_jsonl_to_parquet \
    --input-dir datasets_jsonl_v2 --output-dir datasets_parquet

# upload shards and dataset card (requires `hf auth login`)
python -m scripts.release.upload_to_hf --repo-id <org>/<name> --data-dir datasets_parquet
```

The appendix A.6 experiment asks an LLM to decompile `objdump_asm` with and
without `func_call` and `data_section` as extra context. It reads release
splits from `datasets_jsonl_v2/<split>.jsonl` (or `--dataset`) and needs
`OPENAI_API_KEY`:

```bash
python -m scripts.experiments.a6_filter_subsets --split test_real --arch x86-64 --opt O2
python -m scripts.experiments.a6_decompile_eval --split test_real --arch x86-64 --opt O2 --limit 4
python -m scripts.experiments.a6_build_tables
```

## Tests

```bash
python -m pytest
```

Integration tests compile real code. Tests that need a missing toolchain are
skipped (markers `requires_gcc`, `requires_cross_gcc`, `slow` in
`pytest.ini`); select fast tests with `python -m pytest tests/unit`.

## Known limitations

- Nearly every repository in the evaluation splits also contributes to
  training (98.6-99.9%), although no function is duplicated. The evaluation
  splits therefore measure generalization to new functions, not new
  codebases.
- `data_section` requires a successful link, so functions that call siblings
  missing from the record have no data section (32-48% build failures on a
  `train_real_compilable` sample). For functions using `assert()`, the bytes
  include a temporary file path and differ between runs.
- `func_call` skips functions with struct-by-value, SIMD or `_Complex`
  parameters (about 0.5%) and functions named `main`.
- Wrapper adaptation fails on two known records (`get_field_string`,
  `i8086UncondJumpDS`); the reason is in `opt_*_exe_wrapper_error`.
- `trivial_function` does not flag straight-line functions longer than two
  lines.

Further details are in [`md_files/Exebench_issues.md`](md_files/Exebench_issues.md#open-issues).

## License

The code is released under the Apache-2.0 license (see `LICENSE`). The
ExeBenchPlus fields are released under the MIT license; the C functions keep
the licenses of the repositories they come from, as in ExeBench.

## Citation

The ExeBenchPlus citation will be added after review. Please also cite
ExeBench:

> Jordi Armengol-Estapé, Jackson Woodruff, Alexander Brauckmann, José Wesley
> de Souza Magalhães, and Michael F. P. O'Boyle. 2022. ExeBench: an ML-scale
> dataset of executable C functions. In *Proceedings of the 6th ACM SIGPLAN
> International Symposium on Machine Programming* (MAPS 2022), 50-59.
> https://doi.org/10.1145/3520312.3534867

ExeBench: [paper](https://dl.acm.org/doi/10.1145/3520312.3534867),
[Hugging Face](https://huggingface.co/datasets/jordiae/exebench),
[GitHub](https://github.com/jordiae/exebench).
