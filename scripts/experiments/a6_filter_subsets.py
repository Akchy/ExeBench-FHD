#!/usr/bin/env python3
"""Write the A.6 evaluation subsets of a split to output_files/a6/<split>_<arch>_<opt>.json.

`data_section`: non-empty `data_section[arch][opt]`. `mixed_params`: the signature mixes float and
int/pointer parameters (`has_mixed_arg_order`). Only evaluable records are kept; each gets a `task_id`
(`<split>:<line number>`) and an `a6_subsets` list.

    python3 -m scripts.experiments.a6_filter_subsets --split test_real valid_real --arch x86-64 --opt O2
"""
import argparse
import json
from pathlib import Path

from scripts.eval_utils import atomic_write_json
from scripts.survey.survey_mixed_arg_order import has_mixed_arg_order
from scripts.extract.extract_data_sections import data_section_by_arch
from scripts.extract.extract_func_call import func_call_by_arch

DATASET_DIR = Path('datasets_jsonl_v2')
SUBSETS = ('data_section', 'mixed_params')


def is_evaluable(row, arch, opt):
    """True if `row` has IO pairs, a wrapper, and the assembly and call site both prompts need."""
    if not row.get('real_io_pairs') or not row.get('real_exe_wrapper'):
        return False
    if not ((row.get('objdump_asm') or {}).get(arch) or {}).get(opt):
        return False
    return bool((func_call_by_arch(row.get('func_call'), opt) or {}).get(arch))


def classify(row, arch, opt):
    subsets = []
    if (data_section_by_arch(row.get('data_section'), opt) or {}).get(arch):
        subsets.append('data_section')
    if has_mixed_arg_order(row.get('signature')):
        subsets.append('mixed_params')
    return subsets


def filter_split(split, arch, opt):
    rows = []
    with open(DATASET_DIR / f'{split}.jsonl') as f:
        for line_no, line in enumerate(f):
            row = json.loads(line)
            if not is_evaluable(row, arch, opt):
                continue
            subsets = classify(row, arch, opt)
            if not subsets:
                continue
            row['task_id'] = f'{split}:{line_no}'
            row['a6_subsets'] = subsets
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', nargs='+', default=['test_real'])
    ap.add_argument('--arch', default='x86-64', choices=['x86', 'x86-64', 'ARM32', 'ARM64'])
    ap.add_argument('--opt', default='O2', choices=['O0', 'O1', 'O2', 'O3', 'Os'])
    ap.add_argument('--out-dir', default='output_files/a6')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.split:
        rows = filter_split(split, args.arch, args.opt)
        path = out_dir / f'{split}_{args.arch}_{args.opt}.json'
        atomic_write_json(path, rows, indent=None)
        counts = {name: sum(name in r['a6_subsets'] for r in rows) for name in SUBSETS}
        print(f'{split}: {len(rows)} records ({counts}) -> {path}')


if __name__ == '__main__':
    main()
