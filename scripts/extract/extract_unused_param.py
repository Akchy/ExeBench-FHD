#!/usr/bin/env python3
"""Flag functions with parameters their body never uses (`unused_param`).

When a parameter is unused, also add opt_real_exe_wrapper/opt_angha_exe_wrapper:
the exe wrapper rewritten to call the function without those arguments, since
a decompiled function from optimized asm usually drops them. Originals are
kept. Functions named main are never flagged.

Parameter names come from func_head_types (or func_head), not `signature`,
which carries no names and mis-splits some declarators. A parameter counts as
used if its name appears anywhere in the body; this errs toward "used".

Usage:
    python3 -m scripts.extract.extract_unused_param --split test_real --n 100 \
        --out output_files/unused_param_test_sample.jsonl
    python3 -m scripts.extract.extract_unused_param --split train_real_compilable \
        --limit 100000 --analyze-only
"""
import argparse
import json
import re
import sys

from scripts.check_split_leakage import iter_split

# Words in a declarator that are never the parameter's name.
_TYPE_KEYWORDS = {
    "void", "char", "short", "int", "long", "float", "double", "signed",
    "unsigned", "struct", "union", "enum", "const", "volatile", "restrict",
    "static", "register", "_Bool", "bool", "complex", "_Complex",
    "__restrict", "__restrict__", "__const", "__volatile__", "__attribute__",
}

# Function-pointer name: `int (*fn)(const char *file, int flag)` -> fn, not flag.
_FNPTR_NAME_RE = re.compile(r"\(\s*\*+\s*([A-Za-z_]\w*)\s*\)")
# (?<!\w) keeps literal suffixes like the L in `long A[16L]` from matching.
_IDENT_RE = re.compile(r"(?<!\w)[A-Za-z_]\w*")


def split_top_level(s, sep=","):
    """Split `s` on `sep`, ignoring separators nested inside () or []."""
    parts = []
    depth = 0
    cur = []
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def extract_param_list(header):
    """Text of the last top-level (...) group in `header`, or None.

    The last, because a leading __attribute__((...)) or __typeof__(...) is
    also a top-level group.
    """
    groups = []
    depth = 0
    start = None
    for i, ch in enumerate(header):
        if ch == "(":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                groups.append(header[start:i])
                start = None
    return groups[-1] if groups else None


def param_name(decl):
    """Name declared by one parameter declarator, or None (void, ..., unnamed)."""
    decl = decl.strip()
    if decl in ("", "void", "..."):
        return None
    m = _FNPTR_NAME_RE.search(decl)
    if m:
        return m.group(1)
    idents = [t for t in _IDENT_RE.findall(decl) if t not in _TYPE_KEYWORDS]
    return idents[-1] if idents else None


_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")


def _strip_comments(text):
    """Remove C comments, so `int nMs /* stop time */` isn't named "time"."""
    return _LINE_COMMENT_RE.sub("", _BLOCK_COMMENT_RE.sub("", text))


def parse_params(rec):
    """Ordered [{"index", "name", "type"}] parsed from func_head_types
    (which spells K&R parameters as `type name`) or func_head. `name` may be
    None. Returns [] for no-argument functions or an unparseable header.
    """
    header = rec.get("func_head_types") or rec.get("func_head") or ""
    header = _strip_comments(header)
    plist = extract_param_list(header)
    if plist is None or not plist.strip():
        return []
    decls = split_top_level(plist)
    if len(decls) == 1 and decls[0].strip() in ("void", "...", ""):
        return []
    params = []
    for i, decl in enumerate(decls):
        if decl.strip() == "...":
            continue
        params.append({"index": i, "name": param_name(decl), "type": decl.strip()})
    return params


_BODY_START_RE = re.compile(r"\{")


def function_body(func_def):
    """func_def from the first `{`, skipping the parameter list and K&R declarations."""
    m = _BODY_START_RE.search(func_def or "")
    return func_def[m.start():] if m else (func_def or "")


def find_unused_params(rec):
    """Named parameters (as in parse_params) whose name never appears in the
    body. Always [] for main."""
    if rec.get("fname") == "main":
        return []
    params = parse_params(rec)
    if not params:
        return []
    body = function_body(rec.get("func_def") or "")
    unused = []
    for p in params:
        if p["name"] is None:
            continue
        if not re.search(r"\b" + re.escape(p["name"]) + r"\b", body):
            unused.append(p)
    return unused


def unused_param(rec):
    """The `unused_param` field."""
    return bool(find_unused_params(rec))


_WRAPPER_FIELDS = {
    "real_exe_wrapper": "opt_real_exe_wrapper",
    "angha_exe_wrapper": "opt_angha_exe_wrapper",
}


def process_record(rec, split):
    """Add `unused_param`, and when True an opt_<x>_exe_wrapper plus
    opt_<x>_exe_wrapper_error for each non-empty <x>_exe_wrapper ("" and the
    error message if adaptation fails). `split` is unused; it keeps the
    signature shared with the pipeline.
    """
    # Imported here because adapt_unused_param_wrapper imports this module.
    from scripts.extract.adapt_unused_param_wrapper import WrapperAdaptError, adapt_exe_wrapper

    unused = find_unused_params(rec)
    out = dict(rec)
    out["unused_param"] = bool(unused)
    if unused:
        names = [p["name"] for p in unused]
        for field, opt_field in _WRAPPER_FIELDS.items():
            wrapper_src = rec.get(field)
            if not wrapper_src:
                continue
            try:
                out[opt_field] = adapt_exe_wrapper(wrapper_src, rec["fname"], names)
                out[f"{opt_field}_error"] = None
            except WrapperAdaptError as e:
                out[opt_field] = ""
                out[f"{opt_field}_error"] = str(e)
    return out


def analyze_split(split, limit):
    """Compare parse_params() counts with `signature` counts (diagnostic only;
    most mismatches are signature's own parsing gaps)."""
    scanned = match = mismatch = 0
    mismatch_samples = []
    for rec in iter_split(split, limit=limit):
        scanned += 1
        sig = rec.get("signature") or []
        expected = len([t for t in sig[1:] if t.strip() != ""])
        actual = len(parse_params(rec))
        if actual == expected:
            match += 1
        else:
            mismatch += 1
            if len(mismatch_samples) < 10:
                mismatch_samples.append({
                    "fname": rec.get("fname"),
                    "header": rec.get("func_head_types") or rec.get("func_head"),
                    "signature": sig,
                    "parsed_param_count": actual,
                })
    return {
        "split": split,
        "scanned": scanned,
        "signature_param_count_match": match,
        "signature_param_count_mismatch": mismatch,
        "mismatch_rate": mismatch / scanned if scanned else 0.0,
        "mismatch_samples": mismatch_samples,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train_real_compilable")
    ap.add_argument("--n", type=int, default=100, help="number of matching (unused_param=True) rows to collect (ignored with --all)")
    ap.add_argument("--out", default="output_files/unused_param_sample.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="cap records scanned (debugging)")
    ap.add_argument("--all", action="store_true",
                     help="write every record (unused_param=False included) instead of "
                          "only the first --n matches")
    ap.add_argument("--analyze-only", action="store_true",
                     help="only compare parameter counts against signature; writes nothing")
    args = ap.parse_args()

    if args.analyze_only:
        print(json.dumps(analyze_split(args.split, args.limit), indent=2))
        return

    scanned = 0
    matched = 0
    opt_real_exe_wrapper_ok = 0
    opt_angha_exe_wrapper_ok = 0
    opt_real_exe_wrapper_failed = 0
    opt_angha_exe_wrapper_failed = 0
    with open(args.out, "w") as out:
        for rec in iter_split(args.split, limit=args.limit):
            scanned += 1
            unused = find_unused_params(rec)
            if unused or args.all:
                rec_out = process_record(rec, args.split)
                out.write(json.dumps(rec_out) + "\n")
                if unused:
                    matched += 1
                    if rec_out.get("opt_real_exe_wrapper"):
                        opt_real_exe_wrapper_ok += 1
                    elif rec_out.get("opt_real_exe_wrapper_error"):
                        opt_real_exe_wrapper_failed += 1
                    if rec_out.get("opt_angha_exe_wrapper"):
                        opt_angha_exe_wrapper_ok += 1
                    elif rec_out.get("opt_angha_exe_wrapper_error"):
                        opt_angha_exe_wrapper_failed += 1
                    if not args.all and matched >= args.n:
                        break
            if scanned % 200_000 == 0:
                print(f"  scanned {scanned}, matched {matched}", file=sys.stderr, flush=True)

    print(json.dumps({
        "split": args.split,
        "out": args.out,
        "scanned": scanned,
        "matched": matched,
        "opt_real_exe_wrapper_ok": opt_real_exe_wrapper_ok,
        "opt_real_exe_wrapper_failed": opt_real_exe_wrapper_failed,
        "opt_angha_exe_wrapper_ok": opt_angha_exe_wrapper_ok,
        "opt_angha_exe_wrapper_failed": opt_angha_exe_wrapper_failed,
    }, indent=2))


if __name__ == "__main__":
    main()
