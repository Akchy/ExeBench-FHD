#!/usr/bin/env python3
"""Flag functions whose body is empty, a single return, or at most two lines
(`trivial_function`), using extract_zero_branch_triviality.classify.

By default writes the first --n matching records unchanged; --all writes every
record with `trivial_function` added. --split all writes one file per split to
--out-dir.

Usage:
    python3 -m scripts.extract.extract_trivial_function --split train_real_compilable --all \
        --out output_files/train_real_compilable_trivial_function.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

from scripts.check_split_leakage import ALL_SPLITS, iter_split
from scripts.extract.extract_zero_branch_triviality import classify

_TRIVIAL_CATEGORIES = frozenset({"empty", "return_only", "two_loc"})


def is_trivial_function(rec):
    """True for empty, return-only or <=2-line bodies (not the wider "gap" tier)."""
    return classify(rec) in _TRIVIAL_CATEGORIES


def process_record(rec):
    """Return `rec` with the `trivial_function` field added."""
    out = dict(rec)
    out["trivial_function"] = is_trivial_function(rec)
    return out


def _run_split(split, out_path, n, limit, write_all):
    scanned = 0
    matched = 0
    with open(out_path, "w") as out:
        for rec in iter_split(split, limit=limit):
            scanned += 1
            flagged = is_trivial_function(rec)
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
                     help="write every record with trivial_function added, instead of only "
                          "the first --n matches (written unchanged)")
    ap.add_argument("--out", default="output_files/trivial_function_sample.jsonl",
                     help="output path, used when --split is a single split")
    ap.add_argument("--out-dir", default="output_files/trivial_function_field",
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
