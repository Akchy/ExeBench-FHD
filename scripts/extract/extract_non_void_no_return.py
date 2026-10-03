#!/usr/bin/env python3
"""Flag functions with a non-void return type and no `return` statement
anywhere in func_def (`no_return`). The return type is signature[0].

By default writes the first --n matching records unchanged; --all writes every
record with `no_return` added. --split all writes one file per split to --out-dir.

Usage:
    python3 -m scripts.extract.extract_non_void_no_return --split train_real_compilable --all \
        --out output_files/train_real_compilable_no_return.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

from scripts.check_split_leakage import ALL_SPLITS, iter_split

_RETURN_RE = re.compile(r"\breturn\b")


def is_non_void_no_return(rec):
    # Falling off the end of main is defined as return 0, so main is never flagged.
    if rec.get("fname") == "main":
        return False
    sig = rec.get("signature") or []
    ret_type = (sig[0] if sig else "").strip()
    if not ret_type or ret_type == "void":
        return False
    func_def = rec.get("func_def") or ""
    return not _RETURN_RE.search(func_def)


def process_record(rec):
    """Return `rec` with the `no_return` field added."""
    out = dict(rec)
    out["no_return"] = is_non_void_no_return(rec)
    return out


def _run_split(split, out_path, n, limit, write_all):
    scanned = 0
    matched = 0
    with open(out_path, "w") as out:
        for rec in iter_split(split, limit=limit):
            scanned += 1
            flagged = is_non_void_no_return(rec)
            if flagged:
                matched += 1
            if write_all:
                out.write(json.dumps(process_record(rec)) + "\n")
            elif flagged:
                out.write(json.dumps(rec) + "\n")
                if matched >= n:
                    break
            if scanned % 200_000 == 0:
                print(f"  [{split}] scanned {scanned}, matched {matched}",
                      file=sys.stderr, flush=True)
    return {"split": split, "out": str(out_path), "scanned": scanned, "matched": matched}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train_real_compilable", choices=ALL_SPLITS + ["all"],
                     help="one official split, or 'all' for every split in ALL_SPLITS")
    ap.add_argument("--n", type=int, default=100,
                     help="number of matching rows to collect (ignored with --all)")
    ap.add_argument("--all", action="store_true",
                     help="write every record with no_return added, instead of only the "
                          "first --n matches (written unchanged)")
    ap.add_argument("--out", default="output_files/non_void_no_return_sample.jsonl",
                     help="output path, used when --split is a single split")
    ap.add_argument("--out-dir", default="output_files/no_return_field",
                     help="output directory for --split all (one <split>.jsonl each)")
    ap.add_argument("--limit", type=int, default=None,
                     help="cap records scanned per split")
    args = ap.parse_args()

    if args.split == "all":
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for split in ALL_SPLITS:
            print(f"=== {split} ===", file=sys.stderr, flush=True)
            out_path = out_dir / f"{split}.jsonl"
            results.append(_run_split(split, out_path, args.n, args.limit, args.all))
        summary = {
            "splits": ALL_SPLITS,
            "per_split": results,
            "total_scanned": sum(r["scanned"] for r in results),
            "total_matched": sum(r["matched"] for r in results),
        }
        print(json.dumps(summary, indent=2))
    else:
        result = _run_split(args.split, args.out, args.n, args.limit, args.all)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
