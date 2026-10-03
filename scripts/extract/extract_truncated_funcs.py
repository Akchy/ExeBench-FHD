#!/usr/bin/env python3
"""Extract records with `truncated_funcs` set: func_def lost its return type
(or part of its name) upstream. Implicit-int K&R definitions are valid C and
are not flagged; see extract_missing_return_type.classify.

Usage:
    python3 -m scripts.extract.extract_truncated_funcs --split train_real_compilable \
        --n 100 --out output_files/truncated_funcs_sample.jsonl
"""
import argparse
import json
import sys

from scripts.check_split_leakage import iter_split
from scripts.extract.extract_missing_return_type import classify


def truncated_funcs(rec):
    """The `truncated_funcs` field."""
    return classify(rec) in ("missing_non_int", "fname_not_found")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train_real_compilable")
    ap.add_argument("--n", type=int, default=100, help="number of matching rows to collect")
    ap.add_argument("--out", default="output_files/truncated_funcs_sample.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="cap records scanned (debugging)")
    args = ap.parse_args()

    scanned = 0
    matched = 0
    with open(args.out, "w") as out:
        for rec in iter_split(args.split, limit=args.limit):
            scanned += 1
            if truncated_funcs(rec):
                out.write(json.dumps(rec) + "\n")
                matched += 1
                if matched >= args.n:
                    break
            if scanned % 200_000 == 0:
                print(f"  scanned {scanned}, matched {matched}", file=sys.stderr, flush=True)

    print(json.dumps({
        "split": args.split,
        "out": args.out,
        "scanned": scanned,
        "matched": matched,
    }, indent=2))


if __name__ == "__main__":
    main()
