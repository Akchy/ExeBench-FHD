#!/usr/bin/env python3
"""Check adapt_exe_wrapper() against real ExeBench wrappers of unused_param records.

Collects up to --n records that have an unused parameter and an exe wrapper,
runs the text-only adaptation on each wrapper, and buckets successes, failures
and the JSON-read shapes matched. Failing records go to --sample-dir.

    python3 -m scripts.survey.survey_unused_param_wrappers --split train_real_compilable --n 1000
"""
import argparse
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

from scripts.check_split_leakage import iter_split
from scripts.extract.adapt_unused_param_wrapper import WrapperAdaptError, adapt_exe_wrapper
from scripts.extract.extract_unused_param import find_unused_params

_WRAPPER_FIELDS = ("real_exe_wrapper", "angha_exe_wrapper")


def try_adapt(rec, unused_names, shape_counter=None):
    """Adapt each non-empty wrapper field of `rec`; returns {field: outcome}.

    outcome is "ok", "adapt_error:<stage>:<msg>" or "unexpected:<ExcType>:<msg>".
    If given, `shape_counter` is updated with the shapes matched by successful
    adaptations.
    """
    outcomes = {}
    for field in _WRAPPER_FIELDS:
        wrapper_src = rec.get(field)
        if not wrapper_src:
            continue
        shape_log = [] if shape_counter is not None else None
        try:
            adapt_exe_wrapper(wrapper_src, rec["fname"], unused_names, shape_log=shape_log)
            outcomes[field] = "ok"
            if shape_counter is not None:
                shape_counter.update(shape_log)
        except WrapperAdaptError as e:
            outcomes[field] = f"adapt_error:{e.stage}:{e}"
        except Exception as e:
            outcomes[field] = f"unexpected:{type(e).__name__}:{e}"
    return outcomes


def survey(split, n, limit, sample_dir):
    scanned = 0
    matched = 0
    field_attempted = Counter()
    field_ok = Counter()
    failure_reason_counts = Counter()
    shape_counter = Counter()
    edge_case_samples = []

    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    edge_cases_path = sample_dir / f"{split}_edge_cases.jsonl"
    all_matched_path = sample_dir / f"{split}_all_matched.jsonl"

    with open(edge_cases_path, "w") as edge_f, open(all_matched_path, "w") as all_f:
        for rec in iter_split(split, limit=limit):
            scanned += 1
            if scanned % 200_000 == 0:
                print(f"  ... scanned {scanned}, matched {matched}", file=sys.stderr, flush=True)

            if not (rec.get("real_exe_wrapper") or rec.get("angha_exe_wrapper")):
                continue
            unused = find_unused_params(rec)
            if not unused:
                continue

            matched += 1
            unused_names = [p["name"] for p in unused]

            try:
                outcomes = try_adapt(rec, unused_names, shape_counter=shape_counter)
            except Exception as e:
                outcomes = {"__survey_error__": f"unexpected:{type(e).__name__}:{e}"}
                print(f"SURVEY ERROR on {rec.get('fname')}: {traceback.format_exc()}", file=sys.stderr)

            row = {
                "fname": rec["fname"],
                "path": rec.get("path"),
                "unused_names": unused_names,
                "header": rec.get("func_head_types") or rec.get("func_head"),
                "outcomes": outcomes,
            }
            all_f.write(json.dumps(row) + "\n")

            any_failure = False
            for field, outcome in outcomes.items():
                field_attempted[field] += 1
                if outcome == "ok":
                    field_ok[field] += 1
                else:
                    any_failure = True
                    reason_key = outcome.split(":", 2)[0] + ":" + outcome.split(":", 2)[1] if ":" in outcome else outcome
                    failure_reason_counts[f"{field}:{reason_key}"] += 1

            if any_failure and len(edge_case_samples) < 200:
                full_row = dict(row)
                full_row["real_exe_wrapper"] = rec.get("real_exe_wrapper")
                full_row["angha_exe_wrapper"] = rec.get("angha_exe_wrapper")
                full_row["func_def"] = rec.get("func_def")
                edge_case_samples.append(full_row)
                edge_f.write(json.dumps(full_row) + "\n")

            if matched >= n:
                break

    return {
        "split": split,
        "scanned": scanned,
        "matched_unused_param_with_wrapper": matched,
        "field_attempted": dict(field_attempted),
        "field_ok": dict(field_ok),
        "field_failure_rate": {
            field: 1 - (field_ok[field] / field_attempted[field]) if field_attempted[field] else None
            for field in _WRAPPER_FIELDS
        },
        "failure_reason_counts": dict(failure_reason_counts.most_common()),
        "shape_distribution": dict(shape_counter.most_common()),
        "shape_distribution_pct": (
            {shape: round(100 * c / sum(shape_counter.values()), 2)
             for shape, c in shape_counter.most_common()}
            if shape_counter else {}
        ),
        "edge_cases_path": str(edge_cases_path),
        "all_matched_path": str(all_matched_path),
        "edge_case_count_saved": len(edge_case_samples),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train_real_compilable")
    ap.add_argument("--n", type=int, default=1000, help="number of unused_param+wrapper matches to collect")
    ap.add_argument("--limit", type=int, default=None, help="cap records scanned (debugging)")
    ap.add_argument("--out", default="output_files/unused_param_wrapper_survey.json")
    ap.add_argument("--sample-dir", default="output_files/unused_param_wrapper_edge_cases")
    args = ap.parse_args()

    result = survey(args.split, args.n, args.limit, args.sample_dir)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
