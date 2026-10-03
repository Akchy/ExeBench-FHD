#!/usr/bin/env python3
"""Tally ExeBench parameter types by ABI-relevant category.

`categorize()` is also used by extract_func_call.py to decide which parameter
types its driver can synthesize arguments for. `signature` is
[return_type, param_type, ...]; a lone '' param means `(void)`.

    python3 scripts/survey/survey_func_call_param_types.py --split train_real_compilable --limit 500000
    python3 scripts/survey/survey_func_call_param_types.py --split all --out-dir output_files
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

from scripts.check_split_leakage import ALL_SPLITS, iter_split

# Typedefs of integer scalars. The second group holds the most frequent
# otherwise-uncategorized names in the corpus; names that by convention are
# handles/pointers (HANDLE, yyscan_t, tree, ...) or vectors (vec3_t,
# int32x4_t, ...) are deliberately left out.
_INT_TYPEDEF_ALIASES = {
    "size_t", "ssize_t", "off_t", "uid_t", "gid_t", "pid_t", "mode_t",
    "time_t", "socklen_t", "uint", "u_int", "u_short", "u_long", "u_char",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "wchar_t",

    "u8", "u16", "u32", "u64", "U32",
    "s8", "s16", "s32", "s64",
    "uint8", "uint16", "uint32", "uint64", "UINT32",
    "int8", "int16", "int32", "sint32",
    "ulong", "word", "Word", "byte", "integer",
    "BOOL", "Bool", "boolean", "qboolean", "svn_boolean_t",
    "UINT", "INT", "DWORD", "ULONG", "LONGLONG", "BYTE", "WORD",
    "GLenum", "GLint", "GLuint", "GLsizei", "GLshort", "GLbyte", "GLubyte", "GLushort",
    "UChar", "UInt", "Int", "Oid", "ftnlen", "gfp_t", "loff_t", "dbref",
    "IRTemp", "vm_offset_t", "dma_addr_t", "int_t",
}

_FLOAT_TYPEDEF_ALIASES = {
    "GLfloat", "GLdouble", "f32", "f64", "real", "float_t", "double_t",
}

# First match wins.
_CATEGORY_PATTERNS = [
    ("void_placeholder", re.compile(r"^(void)?$")),
    ("varargs", re.compile(r"^\.\.\.$")),
    ("char_or_void_pointer", re.compile(r"^(char|void)\s*\*+$")),
    ("function_pointer", re.compile(r"\(\s*\*+\s*\)\s*\(")),
    ("simd_or_complex", re.compile(
        r"^__m\d*[a-z]*$|^__v\d+[a-z]+\d*$|^__float128$|"
        r"^(unsigned\s+)?__int128$|^_Complex\s")),
    ("struct_or_union_pointer", re.compile(
        r"^(struct|union|enum)\s+\w+\s*\*+$")),
    ("struct_or_union_by_value", re.compile(
        r"^(struct|union|enum)\s+\w+$")),
    ("other_typed_pointer", re.compile(r"\*")),
    ("float_family", re.compile(r"^(float|double|long\s+double)$")),
    ("int_family", re.compile(
        r"^(unsigned\s+|signed\s+)?"
        r"(short|long\s+long|long|int|char|_Bool|bool)(\s+int)?$")),
]


def categorize(param_type: str) -> str:
    """Map a C parameter type string to a category name such as 'int_family'."""
    t = " ".join(param_type.strip().split())
    t = re.sub(r"\b(const|volatile|restrict)\b", "", t)
    t = " ".join(t.split())
    if t in _INT_TYPEDEF_ALIASES:
        return "int_family"
    if t in _FLOAT_TYPEDEF_ALIASES:
        return "float_family"
    for label, pat in _CATEGORY_PATTERNS:
        if pat.search(t):
            return label
    return "uncategorized"


def survey(split, limit=None):
    category_counts = Counter()
    uncategorized_examples = Counter()
    return_type_counts = Counter()
    records_scanned = 0
    records_with_signature = 0

    for rec in iter_split(split, limit=limit):
        records_scanned += 1
        sig = rec.get("signature")
        if not sig:
            continue
        records_with_signature += 1
        return_type_counts[" ".join(sig[0].strip().split())] += 1

        for param_type in sig[1:]:
            label = categorize(param_type)
            category_counts[label] += 1
            if label == "uncategorized":
                uncategorized_examples[param_type.strip()] += 1

        if records_scanned % 200_000 == 0:
            print(f"  ... scanned {records_scanned}", file=sys.stderr, flush=True)

    return {
        "split": split,
        "records_scanned": records_scanned,
        "records_with_signature": records_with_signature,
        "param_type_categories": dict(category_counts.most_common()),
        "top_return_types": dict(return_type_counts.most_common(20)),
        "uncategorized_examples": dict(uncategorized_examples.most_common(30)),
    }


# --split all runs the small eval splits first.
_ALL_SPLITS_EVAL_FIRST = [
    "test_real", "test_synth", "valid_real", "valid_synth",
    "train_real_compilable", "train_synth_compilable",
    "train_real_simple_io", "train_synth_simple_io", "train_synth_rich_io",
    "train_not_compilable",
]
assert set(_ALL_SPLITS_EVAL_FIRST) == set(ALL_SPLITS)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train_real_compilable", choices=ALL_SPLITS + ["all"])
    ap.add_argument("--limit", type=int, default=None,
                     help="cap records scanned per split (omit for the full split)")
    ap.add_argument("--out", default="output_files/func_call_param_type_survey.json",
                     help="output path for a single --split run (ignored with --split all)")
    ap.add_argument("--out-dir", default="output_files",
                     help="directory for per-split + consolidated output with --split all")
    args = ap.parse_args()

    if args.split != "all":
        report = survey(args.split, limit=args.limit)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        return

    os.makedirs(args.out_dir, exist_ok=True)
    per_split = {}
    for split in _ALL_SPLITS_EVAL_FIRST:
        print(f"=== {split} ===", file=sys.stderr, flush=True)
        report = survey(split, limit=args.limit)
        per_split[split] = report
        out_path = f"{args.out_dir}/func_call_param_type_survey_{split}.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  -> {out_path}", file=sys.stderr, flush=True)

    consolidated_path = f"{args.out_dir}/func_call_param_type_survey_all.json"
    with open(consolidated_path, "w") as f:
        json.dump(per_split, f, indent=2)
    print(f"consolidated -> {consolidated_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
