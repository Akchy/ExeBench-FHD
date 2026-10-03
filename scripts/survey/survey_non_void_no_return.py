#!/usr/bin/env python3
"""Count non-void functions with no `return` statement in every ExeBench split.

Uses extract_non_void_no_return.is_non_void_no_return, writes per-split and
total counts to --out, and up to --n matching records per split to --sample-dir.

    python3 scripts/survey/survey_non_void_no_return.py --splits train_real_compilable --limit 50000
"""
import argparse
import json
import sys
from pathlib import Path

from scripts.check_split_leakage import ALL_SPLITS, iter_split
from scripts.extract.extract_non_void_no_return import is_non_void_no_return


def survey_split(split, limit=None, sample_n=20, sample_path=None):
    scanned = 0
    matched = 0

    sample_f = open(sample_path, "w") if sample_path else None
    sampled = 0
    try:
        for rec in iter_split(split, limit=limit):
            scanned += 1
            if is_non_void_no_return(rec):
                matched += 1
                if sample_f is not None and sampled < sample_n:
                    sample_f.write(json.dumps(rec) + "\n")
                    sampled += 1

            if scanned % 200_000 == 0:
                print(f"  ... [{split}] scanned {scanned}, matched {matched}",
                      file=sys.stderr, flush=True)
    finally:
        if sample_f is not None:
            sample_f.close()

    return {
        "split": split,
        "scanned": scanned,
        "non_void_no_return_true": matched,
        "non_void_no_return_rate": matched / scanned if scanned else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=ALL_SPLITS, choices=ALL_SPLITS)
    ap.add_argument("--limit", type=int, default=None, help="cap records scanned per split (debugging)")
    ap.add_argument("--n", type=int, default=20, help="matched-record sample size per split")
    ap.add_argument("--out", default="output_files/non_void_no_return_survey.json")
    ap.add_argument("--sample-dir", default="output_files/non_void_no_return_samples")
    args = ap.parse_args()

    sample_dir = Path(args.sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    per_split = {}
    for split in args.splits:
        print(f"=== {split} ===", file=sys.stderr, flush=True)
        sample_path = sample_dir / f"{split}.jsonl"
        per_split[split] = survey_split(
            split, limit=args.limit, sample_n=args.n, sample_path=str(sample_path),
        )
        print(json.dumps(per_split[split], indent=2), file=sys.stderr, flush=True)

    totals = {
        "scanned": sum(r["scanned"] for r in per_split.values()),
        "non_void_no_return_true": sum(r["non_void_no_return_true"] for r in per_split.values()),
    }
    totals["non_void_no_return_rate"] = (
        totals["non_void_no_return_true"] / totals["scanned"] if totals["scanned"] else 0.0
    )

    survey = {"per_split": per_split, "totals": totals}
    with open(args.out, "w") as f:
        json.dump(survey, f, indent=2)

    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
