#!/usr/bin/env python3
"""Find records whose func_def is missing the return type shown in func_head.

classify() separates three cases:
  missing_int_implicit  K&R / C89 implicit int, valid C
  missing_non_int       a non-int return type was cut off (upstream truncation)
  fname_not_found       truncation reached into the function name
Only an empty prefix counts as missing; prefix and func_head types are not
compared, since func_head canonicalizes typedefs (e.g. uint32_t -> int).

Usage:
    python3 -m scripts.extract.extract_missing_return_type --split all \
        --summary-out output_files/missing_return_type_survey.json
"""
import argparse
import json
import re
import sys
from collections import Counter

from scripts.check_split_leakage import ALL_SPLITS, iter_split

_STORAGE_QUALS = re.compile(
    r"^\s*(static|inline|extern|register|__inline__|__inline|_Noreturn|"
    r"__attribute__\s*\(\([^)]*\)\)|__attribute__\s*\(\(\([^)]*\)\)\))\s*",
    re.IGNORECASE)


def _strip_quals(prefix):
    """Strip leading storage-class keywords and attributes, repeatedly."""
    prev = None
    p = prefix
    while p != prev:
        prev = p
        p = _STORAGE_QUALS.sub("", p, count=1)
    return p


def _head_ret_type(rec):
    """The return type as `func_head` spells it, e.g. "void", "int", "char *"."""
    head = rec.get("func_head") or ""
    fname = rec.get("fname") or ""
    if not head or not fname:
        return None
    m = re.search(re.escape(fname) + r"\s*\(", head)
    if not m:
        return None
    return head[:m.start()].strip()


def _find_def_split(fdef, fname):
    """Text of `fdef` before the function name, or None if the name isn't found.

    Besides `fname(`, accepts parenthesized names (`int ((f))(...)`) and
    whitespace, `# 301 "file.h"` line markers or `)` between the name and
    its parameter list.
    """
    m = re.search(r"\b" + re.escape(fname) + r"\b\s*\(", fdef)
    if m:
        return fdef[:m.start()]
    m = re.search(r"\(+\s*" + re.escape(fname) + r"\s*\)+\s*\(", fdef)
    if m:
        return fdef[:m.start()]
    m = re.search(r"\b" + re.escape(fname) + r"\b(?:\s|#[^\n]*|\))*\(", fdef)
    if m:
        return fdef[:m.start()]
    return None


def classify(rec):
    """Return "missing_int_implicit", "missing_non_int", "fname_not_found", or None
    when func_def has a return type prefix."""
    fdef = rec.get("func_def") or ""
    fname = rec.get("fname") or ""
    if not fdef or not fname:
        return None
    dp = _find_def_split(fdef, fname)
    if dp is None:
        return "fname_not_found"
    if _strip_quals(dp).strip() == "":
        ht = _head_ret_type(rec)
        return "missing_int_implicit" if ht == "int" else "missing_non_int"
    return None


def _pct(n, total):
    return round(100 * n / total, 4) if total else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train_real_compilable",
                     choices=ALL_SPLITS + ["all"],
                     help="one official split, or 'all' for every split in ALL_SPLITS")
    ap.add_argument("--n", type=int, default=None,
                     help="stop each split after this many matches (all categories)")
    ap.add_argument("--out-non-int", default="output_files/missing_return_type_non_int_sample.jsonl",
                     help="all missing_non_int records (uncapped)")
    ap.add_argument("--out-kr", default="output_files/missing_return_type_kr_implicit_int_sample.jsonl",
                     help="up to --kr-cap missing_int_implicit records")
    ap.add_argument("--out-fname", default="output_files/missing_return_type_fname_corrupted_sample.jsonl",
                     help="up to --fname-cap fname_not_found records")
    ap.add_argument("--kr-cap", type=int, default=200)
    ap.add_argument("--fname-cap", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="cap records scanned per split (debugging)")
    ap.add_argument("--summary-out", default="output_files/missing_return_type_survey.json")
    args = ap.parse_args()

    splits = ALL_SPLITS if args.split == "all" else [args.split]

    per_split = {}
    global_counts = Counter()
    global_non_int_types = Counter()
    global_scanned = 0

    kr_written = 0
    fname_written = 0
    non_int_written = 0

    with open(args.out_non_int, "w") as f_non_int, \
         open(args.out_kr, "w") as f_kr, \
         open(args.out_fname, "w") as f_fname:

        for split in splits:
            scanned = 0
            counts = Counter()
            non_int_types = Counter()

            print(f"== {split} ==", file=sys.stderr, flush=True)
            for rec in iter_split(split, limit=args.limit):
                scanned += 1
                cat = classify(rec)
                if cat is not None:
                    counts[cat] += 1
                    global_counts[cat] += 1

                    if cat == "missing_non_int":
                        ht = _head_ret_type(rec) or "<unknown>"
                        non_int_types[ht] += 1
                        global_non_int_types[ht] += 1
                        f_non_int.write(json.dumps(rec) + "\n")
                        non_int_written += 1
                    elif cat == "missing_int_implicit" and kr_written < args.kr_cap:
                        f_kr.write(json.dumps(rec) + "\n")
                        kr_written += 1
                    elif cat == "fname_not_found" and fname_written < args.fname_cap:
                        f_fname.write(json.dumps(rec) + "\n")
                        fname_written += 1

                if scanned % 200_000 == 0:
                    print(f"  [{split}] scanned {scanned}, "
                          f"missing_int_implicit={counts['missing_int_implicit']} "
                          f"missing_non_int={counts['missing_non_int']} "
                          f"fname_not_found={counts['fname_not_found']}",
                          file=sys.stderr, flush=True)

                if args.n is not None and sum(counts.values()) >= args.n:
                    break

            global_scanned += scanned
            per_split[split] = {
                "scanned": scanned,
                "missing_int_implicit": counts["missing_int_implicit"],
                "missing_non_int": counts["missing_non_int"],
                "fname_not_found": counts["fname_not_found"],
                "missing_int_implicit_pct": _pct(counts["missing_int_implicit"], scanned),
                "missing_non_int_pct": _pct(counts["missing_non_int"], scanned),
                "fname_not_found_pct": _pct(counts["fname_not_found"], scanned),
                "missing_non_int_types": dict(non_int_types.most_common()),
            }

    summary = {
        "splits": splits,
        "total_scanned": global_scanned,
        "global": {
            "missing_int_implicit": global_counts["missing_int_implicit"],
            "missing_non_int": global_counts["missing_non_int"],
            "fname_not_found": global_counts["fname_not_found"],
            "missing_int_implicit_pct": _pct(global_counts["missing_int_implicit"], global_scanned),
            "missing_non_int_pct": _pct(global_counts["missing_non_int"], global_scanned),
            "fname_not_found_pct": _pct(global_counts["fname_not_found"], global_scanned),
        },
        "missing_non_int_types_global": dict(global_non_int_types.most_common()),
        "per_split": per_split,
        "samples_written": {
            "non_int": {"path": args.out_non_int, "n": non_int_written, "capped": False},
            "kr_implicit_int": {"path": args.out_kr, "n": kr_written, "capped": kr_written >= args.kr_cap},
            "fname_corrupted": {"path": args.out_fname, "n": fname_written, "capped": fname_written >= args.fname_cap},
        },
    }

    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
