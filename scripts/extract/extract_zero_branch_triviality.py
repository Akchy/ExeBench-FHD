#!/usr/bin/env python3
"""Sample records by body triviality. The categories nest:

    empty < return_only < two_loc (<= 2 lines) < gap (no branches at all)

and each record goes in the narrowest one. Writes up to --n records per
category to <out-prefix>_<category>.jsonl. Text heuristics, not a C parser.

Usage:
    python3 -m scripts.extract.extract_zero_branch_triviality --split train_real_compilable \
        --n 10 --out-prefix output_files/zero_branch_triviality_sample
"""
import argparse
import json
import re
import sys

from scripts.check_split_leakage import iter_split

_COMMENT_RE = re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)
_WS_RE = re.compile(r"\s+")
_RETURN_ONLY_RE = re.compile(r"^return[^;]*;$")
_BRANCH_RE = re.compile(r"\b(if|else|for|while|do|switch|case|goto)\b|\?")

CATEGORIES = ("empty", "return_only", "two_loc", "gap")


def extract_body(func_def):
    """Text between the outermost matched braces of func_def, or None."""
    i = func_def.find("{")
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(func_def)):
        c = func_def[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return func_def[i + 1:j]
    return None


def classify(rec):
    """The narrowest of CATEGORIES the body falls in, or None (has branches,
    or no brace-matched body)."""
    body = extract_body(rec.get("func_def") or "")
    if body is None:
        return None

    clean = _COMMENT_RE.sub("", body)
    if clean.strip() == "":
        return "empty"
    if _RETURN_ONLY_RE.match(_WS_RE.sub(" ", clean).strip()):
        return "return_only"
    if len([l for l in clean.strip("\n").split("\n") if l.strip()]) <= 2:
        return "two_loc"
    if not _BRANCH_RE.search(clean):
        return "gap"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train_real_compilable")
    ap.add_argument("--n", type=int, default=10, help="samples to collect per category")
    ap.add_argument("--out-prefix", default="output_files/zero_branch_triviality_sample")
    ap.add_argument("--limit", type=int, default=None, help="cap records scanned (debugging)")
    args = ap.parse_args()

    files = {cat: open(f"{args.out_prefix}_{cat}.jsonl", "w") for cat in CATEGORIES}
    counts = {cat: 0 for cat in CATEGORIES}
    scanned = 0

    try:
        for rec in iter_split(args.split, limit=args.limit):
            scanned += 1
            cat = classify(rec)
            if cat is not None and counts[cat] < args.n:
                files[cat].write(json.dumps(rec) + "\n")
                counts[cat] += 1
            if scanned % 200_000 == 0:
                print(f"  scanned {scanned}, counts {counts}", file=sys.stderr, flush=True)
            if all(counts[cat] >= args.n for cat in CATEGORIES):
                break
    finally:
        for fh in files.values():
            fh.close()

    print(json.dumps({
        "split": args.split,
        "out_prefix": args.out_prefix,
        "scanned": scanned,
        "counts": counts,
    }, indent=2))


if __name__ == "__main__":
    main()
