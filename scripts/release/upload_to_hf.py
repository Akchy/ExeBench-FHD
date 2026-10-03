#!/usr/bin/env python3
"""Upload the Parquet release and dataset_card/README.md to a Hugging Face dataset repo, after checking
the card's splits against the shards. `{{repo_id}}` in the card becomes --repo-id. Resumable; needs
`hf auth login` or HF_TOKEN.

    python -m scripts.release.upload_to_hf --repo-id <user-or-org>/ExeBenchPlus \\
        --data-dir datasets_parquet [--card dataset_card/README.md] [--private]
    python -m scripts.release.upload_to_hf --repo-id <...> --card-only
"""
import argparse
import fnmatch
import glob
import json
import os
import sys

import yaml

DEFAULT_CARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset_card", "README.md")


def read_card_yaml(card_path):
    with open(card_path) as f:
        text = f.read()
    if not text.startswith("---\n"):
        raise ValueError(f"{card_path}: no YAML front matter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError(f"{card_path}: unterminated YAML front matter")
    return yaml.safe_load(text[4:end])


def render_card(card_path, repo_id):
    """The card's text with `{{repo_id}}` filled in."""
    with open(card_path) as f:
        return f.read().replace("{{repo_id}}", repo_id)


def card_splits(meta):
    """{split: data_files glob} from the card's single `configs` entry."""
    configs = meta.get("configs") or []
    if len(configs) != 1:
        raise ValueError(f"expected exactly one config in the card, found {len(configs)}")
    return {entry["split"]: entry["path"] for entry in configs[0]["data_files"]}


def check_card_against_data(card_path, data_dir):
    """List of mismatches between the card's splits and the shards (and splits.json) in data_dir."""
    splits = card_splits(read_card_yaml(card_path))
    shards = sorted(os.path.relpath(p, data_dir) for p in glob.glob(os.path.join(data_dir, "data", "*.parquet")))
    problems = []
    claimed = set()
    for split, pattern in splits.items():
        matched = [s for s in shards if fnmatch.fnmatch(s, pattern)]
        if not matched:
            problems.append(f"split {split!r}: no shard matches {pattern!r}")
        claimed.update(matched)
    for s in shards:
        if s not in claimed:
            problems.append(f"shard {s} is not covered by any split in the card")

    stats_path = os.path.join(data_dir, "splits.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        for split in sorted(set(splits) ^ set(stats)):
            where = "the card" if split in splits else "splits.json"
            problems.append(f"split {split!r} is only in {where}")
    if glob.glob(os.path.join(data_dir, ".tmp_*")):
        problems.append("unfinished conversion (.tmp_* directories present) - rerun the converter first")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True, help="<user-or-org>/<name> of the HF dataset repo")
    ap.add_argument("--data-dir", default="datasets_parquet", help="--output-dir of convert_jsonl_to_parquet")
    ap.add_argument("--card", default=DEFAULT_CARD, help="dataset card (README.md with YAML front matter)")
    ap.add_argument("--private", action="store_true", help="create the repo as private (only if it doesn't exist)")
    ap.add_argument("--card-only", action="store_true", help="upload only the card")
    ap.add_argument("--num-workers", type=int, default=None, help="upload_large_folder workers")
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi
    api = HfApi()

    if args.card_only:
        card_splits(read_card_yaml(args.card))  # fail early on a malformed card
        api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_file(path_or_fileobj=render_card(args.card, args.repo_id).encode(), path_in_repo="README.md",
                        repo_id=args.repo_id, repo_type="dataset", commit_message="Update dataset card")
        print(f"card uploaded to https://huggingface.co/datasets/{args.repo_id}")
        return

    problems = check_card_against_data(args.card, args.data_dir)
    if problems:
        sys.exit("card and data disagree:\n  " + "\n  ".join(problems))

    # upload_large_folder mirrors the folder, so the card goes at its root.
    with open(os.path.join(args.data_dir, "README.md"), "w") as f:
        f.write(render_card(args.card, args.repo_id))
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_large_folder(
        repo_id=args.repo_id, repo_type="dataset", folder_path=args.data_dir,
        allow_patterns=["README.md", "data/*.parquet"], num_workers=args.num_workers,
    )
    print(f"uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
