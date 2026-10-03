#!/usr/bin/env python3
"""Count `unused_param` records in every ExeBench split.

For each split reports the match rate, a histogram of unused-parameter counts,
how many matches have io pairs, and how often their exe wrappers could be
adapted into opt_*_exe_wrapper. Writes up to --n processed matches per split
to --sample-dir. `main`-named functions are never flagged.

    python3 scripts/survey/survey_unused_param.py --splits train_real_compilable --limit 50000
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from scripts.check_split_leakage import ALL_SPLITS, EVAL_SPLITS, iter_split
from scripts.extract.extract_unused_param import find_unused_params, process_record


def survey_split(split, limit=None, sample_n=20, sample_path=None):
    scanned = 0
    main_skipped = 0
    unused_true = 0
    hist = Counter()
    has_real_io = 0
    has_angha_io = 0
    opt_real_wrapper_ok = 0
    opt_real_wrapper_failed = 0
    opt_angha_wrapper_ok = 0
    opt_angha_wrapper_failed = 0

    sample_f = open(sample_path, "w") if sample_path else None
    sampled = 0
    try:
        for rec in iter_split(split, limit=limit):
            scanned += 1
            if rec.get("fname") == "main":
                main_skipped += 1

            unused = find_unused_params(rec)
            if unused:
                unused_true += 1
                hist[len(unused)] += 1
                if rec.get("real_io_pairs"):
                    has_real_io += 1
                if rec.get("angha_io_pairs"):
                    has_angha_io += 1

                # process_record is cheap, so run it on every match to count
                # opt_* wrappers, not just the sampled ones.
                out = process_record(rec, split)
                if out.get("opt_real_exe_wrapper"):
                    opt_real_wrapper_ok += 1
                elif out.get("opt_real_exe_wrapper_error"):
                    opt_real_wrapper_failed += 1
                if out.get("opt_angha_exe_wrapper"):
                    opt_angha_wrapper_ok += 1
                elif out.get("opt_angha_exe_wrapper_error"):
                    opt_angha_wrapper_failed += 1
                if sample_f is not None and sampled < sample_n:
                    sample_f.write(json.dumps(out) + "\n")
                    sampled += 1

            if scanned % 200_000 == 0:
                print(f"  ... [{split}] scanned {scanned}, unused_param_true {unused_true}",
                      file=sys.stderr, flush=True)
    finally:
        if sample_f is not None:
            sample_f.close()

    return {
        "split": split,
        "is_eval_split": split in EVAL_SPLITS,
        "scanned": scanned,
        "main_skipped": main_skipped,
        "unused_param_true": unused_true,
        "unused_param_rate": unused_true / scanned if scanned else 0.0,
        "unused_param_count_hist": dict(sorted(hist.items())),
        "unused_true_with_real_io_pairs": has_real_io,
        "unused_true_with_angha_io_pairs": has_angha_io,
        "opt_real_exe_wrapper_ok": opt_real_wrapper_ok,
        "opt_real_exe_wrapper_failed": opt_real_wrapper_failed,
        "opt_angha_exe_wrapper_ok": opt_angha_wrapper_ok,
        "opt_angha_exe_wrapper_failed": opt_angha_wrapper_failed,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=ALL_SPLITS, choices=ALL_SPLITS)
    ap.add_argument("--limit", type=int, default=None, help="cap records scanned per split (debugging)")
    ap.add_argument("--n", type=int, default=20, help="matched-record sample size per split")
    ap.add_argument("--out", default="output_files/unused_param_survey.json")
    ap.add_argument("--sample-dir", default="output_files/unused_param_samples")
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
        "main_skipped": sum(r["main_skipped"] for r in per_split.values()),
        "unused_param_true": sum(r["unused_param_true"] for r in per_split.values()),
    }
    totals["unused_param_rate"] = (
        totals["unused_param_true"] / totals["scanned"] if totals["scanned"] else 0.0
    )

    survey = {"per_split": per_split, "totals": totals}
    with open(args.out, "w") as f:
        json.dump(survey, f, indent=2)

    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
