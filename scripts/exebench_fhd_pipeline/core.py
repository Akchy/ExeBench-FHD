#!/usr/bin/env python3
"""process_record(): run every ExeBench-FHD field extractor on one record.

Adds no_return, dummy_udt, trivial_function, truncated_funcs, unused_param,
opt_{real,angha}_exe_wrapper, func_call* and data_section* by calling each
scripts/extract module directly. The extractors only read original ExeBench
fields, so the cheap text-only ones run first.

asm, asm_was_static, objdump_asm and disassembled_asm are deliberately left
out: ExeBench records already have a differently shaped `asm` field that
extract_asm_regen would overwrite, and the objdump fields are expensive. Run
scripts/extract/extract_asm_regen.py standalone if you need them.

Each record costs about 8 gcc/objdump calls (4 archs x 2 extractors).
"""
from scripts.extract import (
    extract_data_sections,
    extract_dummy_struct_types,
    extract_func_call,
    extract_non_void_no_return,
    extract_trivial_function,
    extract_unused_param,
)
from scripts.extract.extract_truncated_funcs import truncated_funcs

GCC_TIMEOUT = 60


def process_record(rec, split, work_dir, task_id, archs=None, timeout=GCC_TIMEOUT,
                    cleanup=True):
    """Return a copy of `rec` with every in-scope ExeBench-FHD field added.

    `work_dir`/`task_id` name the scratch directory for func_call's temp files.
    `archs` defaults to all four (x86, x86-64, ARM32, ARM64).
    """
    out = dict(rec)

    out = extract_non_void_no_return.process_record(out)
    out = extract_dummy_struct_types.process_record(out)
    out = extract_trivial_function.process_record(out)
    out["truncated_funcs"] = truncated_funcs(out)
    out = extract_unused_param.process_record(out, split)

    out, _func_call_skip_reason = extract_func_call.process_record(
        out, work_dir, f"{task_id}_fc", archs=archs, timeout=timeout, cleanup=cleanup)

    data_section, data_section_errors = extract_data_sections.process_record(
        out, archs=archs, timeout=timeout)
    out["data_section"] = data_section
    out["data_section_errors"] = data_section_errors

    return out
