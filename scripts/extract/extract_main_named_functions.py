#!/usr/bin/env python3
"""Extract every record whose function is named `main`. These break any
driver that appends its own main() (redefinition of 'main').

Usage:
    python3 -m scripts.extract.extract_main_named_functions --split train_real_compilable \
        --out output_files/main_named_functions.jsonl
"""
import argparse
import json
import sys

from scripts.check_split_leakage import iter_split


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train_real_compilable")
    ap.add_argument("--out", default="output_files/main_named_functions.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="cap records scanned (debugging)")
    args = ap.parse_args()

    scanned = 0
    matched = 0
    with open(args.out, "w") as out:
        for rec in iter_split(args.split, limit=args.limit):
            scanned += 1
            if (rec.get("fname") or "") == "main":
                out.write(json.dumps(rec) + "\n")
                matched += 1
            if scanned % 200_000 == 0:
                print(f"  scanned {scanned}, matched {matched}", file=sys.stderr, flush=True)

    print(json.dumps({
        "split": args.split,
        "out": args.out,
        "scanned": scanned,
        "matched": matched,
        "pct": round(100 * matched / scanned, 2) if scanned else 0.0,
    }, indent=2))


if __name__ == "__main__":
    main()
