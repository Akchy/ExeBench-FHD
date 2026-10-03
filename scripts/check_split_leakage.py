#!/usr/bin/env python3
"""Measure duplicate and repo overlap between every pair of ExeBench splits.

Signals, strongest first: byte-identical `func_def`, whitespace-normalized
`func_def`, shared `owner/repo` (from `path`), and shared function name (weak:
common names collide across unrelated repos). Reads the raw snapshot
(tar.gz -> zstd -> jsonl); `iter_split` here is the shared reader for other scripts.

    python3 scripts/check_split_leakage.py --splits test_real,valid_real,train_real_compilable
    python3 scripts/check_split_leakage.py --limit-per-split 5000 --out /tmp/leakage.json
"""
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import threading
from itertools import combinations

# jordiae/exebench in the HF hub cache, resolved like huggingface_hub does.
_HF_HUB_CACHE = os.environ.get(
    "HF_HUB_CACHE",
    os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"),
)
SNAPSHOT = os.environ.get(
    "EXEBENCH_SNAPSHOT",
    os.path.join(_HF_HUB_CACHE, "datasets--jordiae--exebench", "snapshots",
                 "093085f8558cfd53de8e2c8f4ccc7b9e73dc22ae"),
)

# Official split name -> tar.gz stem in the snapshot (only "not" vs "no" differs).
SPLIT_TO_TARSTEM = {
    "train_not_compilable": "train_no_compilable",
    "train_synth_compilable": "train_synth_compilable",
    "train_real_compilable": "train_real_compilable",
    "train_synth_simple_io": "train_synth_simple_io",
    "train_real_simple_io": "train_real_simple_io",
    "train_synth_rich_io": "train_synth_rich_io",
    "valid_synth": "valid_synth",
    "valid_real": "valid_real",
    "test_synth": "test_synth",
    "test_real": "test_real",
}
ALL_SPLITS = list(SPLIT_TO_TARSTEM)
EVAL_SPLITS = {"test_real", "test_synth", "valid_real", "valid_synth"}


def iter_split(split, snapshot=SNAPSHOT, limit=None):
    """Yield the record dicts of `split`, streamed from its tar.gz of zstd'd jsonl."""
    path = os.path.join(snapshot, SPLIT_TO_TARSTEM[split] + ".tar.gz")
    tf = tarfile.open(path, "r:gz")
    seen = 0
    for member in tf:
        if not member.isfile() or not member.name.endswith(".zst"):
            continue
        fh = tf.extractfile(member)
        proc = subprocess.Popen(["zstd", "-dc", "-q"], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        # Feed zstd from a thread so reading its stdout can't deadlock on a full pipe.
        def feed():
            try:
                while True:
                    chunk = fh.read(1 << 20)
                    if not chunk:
                        break
                    proc.stdin.write(chunk)
            except BrokenPipeError:
                pass
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

        threading.Thread(target=feed, daemon=True).start()
        for line in io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)["text"]
            except Exception:
                continue
            yield rec
            seen += 1
            if limit is not None and seen >= limit:
                proc.stdout.close()
                proc.kill()
                tf.close()
                return
        proc.stdout.close()
        proc.wait()
    tf.close()


def _digest(s):
    return hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=16).digest()


def _normalize_body(s):
    return " ".join(s.split())


def _repo_of(path):
    if not path:
        return None
    parts = path.split("/")
    if len(parts) < 2:
        return None
    return "/".join(parts[:2])


def build_index(split, snapshot, limit=None, progress_every=200_000):
    """Collect body digests, repos and fnames for one split.

    Bodies are stored as 16-byte digests so all splits fit in memory at once.
    """
    n = 0
    exact_hashes = set()
    norm_hashes = set()
    repos = set()
    fnames = set()
    repo_examples = {}  # repo -> one example path, for spot-checking a report

    for rec in iter_split(split, snapshot, limit=limit):
        n += 1
        path = rec.get("path") or ""
        func_def = rec.get("func_def") or ""
        fname = rec.get("fname") or ""

        if func_def:
            exact_hashes.add(_digest(func_def))
            norm_hashes.add(_digest(_normalize_body(func_def)))
        repo = _repo_of(path)
        if repo:
            repos.add(repo)
            repo_examples.setdefault(repo, path)
        if fname:
            fnames.add(fname)

        if n % progress_every == 0:
            print(f"  [{split}] {n} records...", file=sys.stderr, flush=True)

    return {
        "split": split,
        "n": n,
        "exact_hashes": exact_hashes,
        "norm_hashes": norm_hashes,
        "repos": repos,
        "fnames": fnames,
        "repo_examples": repo_examples,
    }


def pair_overlap(a, b):
    exact = a["exact_hashes"] & b["exact_hashes"]
    norm = a["norm_hashes"] & b["norm_hashes"]
    repo = a["repos"] & b["repos"]
    fname = a["fnames"] & b["fnames"]

    def pct(part, whole):
        return round(100 * len(part) / len(whole), 2) if whole else 0.0

    return {
        "a": a["split"], "b": b["split"],
        "n_a": a["n"], "n_b": b["n"],
        "exact_dup": len(exact),
        "exact_dup_pct_of_a": pct(exact, a["exact_hashes"]),
        "exact_dup_pct_of_b": pct(exact, b["exact_hashes"]),
        "normalized_dup": len(norm),
        "normalized_dup_pct_of_a": pct(norm, a["norm_hashes"]),
        "normalized_dup_pct_of_b": pct(norm, b["norm_hashes"]),
        "repo_overlap": len(repo),
        "repo_overlap_pct_of_a": pct(repo, a["repos"]),
        "repo_overlap_pct_of_b": pct(repo, b["repos"]),
        "fname_overlap": len(fname),
        "example_repos": sorted(repo)[:5],
    }


def full_containment_matrix(indices, splits):
    """matrix[a][b] = % of split a's unique bodies that also occur in split b (diagonal 100)."""
    matrix = {}
    for a in splits:
        row = {}
        a_hashes = indices[a]["exact_hashes"]
        for b in splits:
            if a == b:
                row[b] = 100.0
                continue
            b_hashes = indices[b]["exact_hashes"]
            inter = len(a_hashes & b_hashes)
            row[b] = round(100 * inter / len(a_hashes), 4) if a_hashes else 0.0
        matrix[a] = row
    return matrix


def corpus_dedup_summary(indices, splits, paper_reported_total=5_221_517):
    """Total rows vs. unique bodies across `splits`, next to the ExeBench paper's dedup total.

    The paper deduplicated the raw pools before splitting, so the delta is only
    a rough sanity check.
    """
    total_rows = sum(indices[s]["n"] for s in splits)
    union_hashes = set()
    for s in splits:
        union_hashes |= indices[s]["exact_hashes"]
    effective_unique = len(union_hashes)
    return {
        "splits_covered": splits,
        "total_rows": total_rows,
        "effective_unique_functions": effective_unique,
        "within_corpus_exact_dup_rows": total_rows - effective_unique,
        "paper_reported_dedup_total": paper_reported_total,
        "delta_vs_paper": effective_unique - paper_reported_total,
        "note": ("effective_unique_functions counts a body once even if it appears "
                 "in multiple splits (e.g. train_real_simple_io subset of "
                 "train_real_compilable) or multiple times within one split; it does "
                 "NOT reproduce the ExeBench paper's own pre-split, cross-Anghabench/"
                 "GitHub dedup pass exactly, so treat delta_vs_paper as directional, "
                 "not an exact reconciliation."),
    }


def union_index(indices, splits):
    """Merge the indices of `splits` into one pseudo-split (used for eval vs. all of train)."""
    exact_hashes, norm_hashes, repos, fnames = set(), set(), set(), set()
    for s in splits:
        idx = indices[s]
        exact_hashes |= idx["exact_hashes"]
        norm_hashes |= idx["norm_hashes"]
        repos |= idx["repos"]
        fnames |= idx["fnames"]
    return {"split": "+".join(splits), "n": sum(indices[s]["n"] for s in splits),
            "exact_hashes": exact_hashes, "norm_hashes": norm_hashes,
            "repos": repos, "fnames": fnames, "repo_examples": {}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default=",".join(ALL_SPLITS),
                    help=f"comma-separated split names (default: all 10 official splits). "
                         f"Choices: {', '.join(ALL_SPLITS)}")
    ap.add_argument("--limit-per-split", type=int, default=None,
                    help="only read the first N records of each split (smoke-testing)")
    ap.add_argument("--snapshot", default=SNAPSHOT, help="path to the ExeBench snapshot dir")
    ap.add_argument("--out", default=None, help="write the full JSON report here")
    ap.add_argument("--full-containment", action="store_true",
                    help="also write the directional NxN exact-dup containment matrix "
                         "and the corpus-wide unique-function count")
    ap.add_argument("--containment-out", default="output_files/split_containment_matrix.json",
                    help="where to write --full-containment's output")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for s in splits:
        if s not in SPLIT_TO_TARSTEM:
            ap.error(f"unknown split {s!r} - choices: {', '.join(ALL_SPLITS)}")

    print(f"Building indices for {len(splits)} split(s)...", file=sys.stderr)
    indices = {}
    for s in splits:
        print(f"== {s} ==", file=sys.stderr)
        indices[s] = build_index(s, args.snapshot, limit=args.limit_per_split)
        idx = indices[s]
        print(f"   n={idx['n']} unique_exact_bodies={len(idx['exact_hashes'])} "
              f"unique_repos={len(idx['repos'])} unique_fnames={len(idx['fnames'])}",
              file=sys.stderr)

    pairwise = [pair_overlap(indices[a], indices[b]) for a, b in combinations(splits, 2)]

    # Headline: each eval split against the union of the non-eval splits.
    headline = []
    for ev in splits:
        if ev not in EVAL_SPLITS:
            continue
        train_like = [s for s in splits if s != ev and s not in EVAL_SPLITS]
        if not train_like:
            continue
        train_union = union_index(indices, train_like)
        stat = pair_overlap(indices[ev], train_union)
        stat["train_splits"] = train_like
        headline.append(stat)

    report = {
        "splits": splits,
        "per_split": {s: {"n": indices[s]["n"],
                           "unique_exact_bodies": len(indices[s]["exact_hashes"]),
                           "unique_repos": len(indices[s]["repos"]),
                           "unique_fnames": len(indices[s]["fnames"])}
                      for s in splits},
        "headline_eval_vs_train_union": headline,
        "pairwise": pairwise,
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote full report to {args.out}", file=sys.stderr)

    if args.full_containment:
        containment_report = {
            "splits": splits,
            "containment_matrix_pct": full_containment_matrix(indices, splits),
            "corpus_dedup_summary": corpus_dedup_summary(indices, splits),
        }
        os.makedirs(os.path.dirname(args.containment_out) or ".", exist_ok=True)
        with open(args.containment_out, "w") as f:
            json.dump(containment_report, f, indent=2)
        print(f"Wrote containment matrix to {args.containment_out}", file=sys.stderr)
        dedup = containment_report["corpus_dedup_summary"]
        print(f"\n=== Corpus dedup summary ({len(splits)} splits) ===")
        print(f"  total_rows={dedup['total_rows']:,}  "
              f"effective_unique_functions={dedup['effective_unique_functions']:,}  "
              f"paper_reported={dedup['paper_reported_dedup_total']:,}  "
              f"delta={dedup['delta_vs_paper']:+,}")

    print("\n=== Per-split sizes ===")
    for s in splits:
        p = report["per_split"][s]
        print(f"  {s:28s} n={p['n']:>9}  unique_repos={p['unique_repos']:>7}  "
              f"unique_bodies={p['unique_exact_bodies']:>9}")

    if headline:
        print("\n=== Headline: eval split vs. UNION of all other requested splits ===")
        print(f"{'eval split':20s} {'exact-dup':>12s} {'norm-dup':>12s} "
              f"{'repo overlap':>14s} {'fname overlap':>14s}")
        for h in headline:
            print(f"{h['a']:20s} {h['exact_dup']:>6d} ({h['exact_dup_pct_of_a']:>5.2f}%) "
                  f"{h['normalized_dup']:>5d} ({h['normalized_dup_pct_of_a']:>5.2f}%) "
                  f"{h['repo_overlap']:>7d} ({h['repo_overlap_pct_of_a']:>5.2f}%) "
                  f"{h['fname_overlap']:>7d}")

    print("\n=== Full pairwise matrix ===")
    print(f"{'A':24s} {'B':24s} {'exact':>8s} {'norm':>8s} {'repo':>8s} {'repo%A':>8s} {'repo%B':>8s} {'fname':>8s}")
    for p in pairwise:
        print(f"{p['a']:24s} {p['b']:24s} {p['exact_dup']:>8d} {p['normalized_dup']:>8d} "
              f"{p['repo_overlap']:>8d} {p['repo_overlap_pct_of_a']:>7.2f}% "
              f"{p['repo_overlap_pct_of_b']:>7.2f}% {p['fname_overlap']:>8d}")


if __name__ == "__main__":
    main()
