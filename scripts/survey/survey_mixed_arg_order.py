#!/usr/bin/env python3
"""Count records whose parameters mix float and int/pointer types.

Under the x86-64 SysV ABI these go in separate register sequences, so the
callee's asm alone doesn't reveal their relative order in the signature.
Uses the same categorize() as survey_func_call_param_types.

    python3 -m scripts.survey.survey_mixed_arg_order --split all --out output_files/mixed_arg_order_survey.json
"""
import argparse
import json
import sys

from scripts.check_split_leakage import ALL_SPLITS, iter_split
from scripts.survey.survey_func_call_param_types import categorize

_INT_LIKE_CATEGORIES = {
    "int_family",
    "char_or_void_pointer",
    "struct_or_union_pointer",
    "other_typed_pointer",
}


def has_mixed_arg_order(sig):
    """True if `sig` ([return_type, param_type, ...]) has both a float and an int/pointer parameter."""
    if not sig or len(sig) < 3:
        return False
    cats = {categorize(t) for t in sig[1:] if t.strip() != ""}
    return "float_family" in cats and bool(cats & _INT_LIKE_CATEGORIES)


def survey(split, limit=None):
    scanned = 0
    with_signature = 0
    mixed = 0
    for row in iter_split(split, limit=limit):
        scanned += 1
        sig = row.get("signature")
        if not sig:
            continue
        with_signature += 1
        if has_mixed_arg_order(sig):
            mixed += 1
        if scanned % 500_000 == 0:
            print(f"  [{split}] ... scanned {scanned}, mixed {mixed}", file=sys.stderr, flush=True)
    return {
        "split": split,
        "records_scanned": scanned,
        "records_with_signature": with_signature,
        "mixed_arg_order_count": mixed,
        "mixed_arg_order_pct": (100.0 * mixed / with_signature) if with_signature else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="all", choices=ALL_SPLITS + ["all"])
    ap.add_argument("--limit", type=int, default=None, help="cap records scanned per split (debugging)")
    ap.add_argument("--out", default="output_files/mixed_arg_order_survey.json")
    args = ap.parse_args()

    splits = ALL_SPLITS if args.split == "all" else [args.split]
    per_split = {}
    for split in splits:
        print(f"=== {split} ===", file=sys.stderr, flush=True)
        per_split[split] = survey(split, limit=args.limit)

    total_scanned = sum(r["records_scanned"] for r in per_split.values())
    total_with_sig = sum(r["records_with_signature"] for r in per_split.values())
    total_mixed = sum(r["mixed_arg_order_count"] for r in per_split.values())

    report = {
        "per_split": per_split,
        "total_records_scanned": total_scanned,
        "total_records_with_signature": total_with_sig,
        "total_mixed_arg_order_count": total_mixed,
        "total_mixed_arg_order_pct": (100.0 * total_mixed / total_with_sig) if total_with_sig else 0.0,
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
