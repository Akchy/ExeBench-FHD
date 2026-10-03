#!/usr/bin/env python3
"""Re-compile / re-execute evaluation of decompiled C predictions against rows
in the raw ExeBench format (datasets_jsonl_v2/*.jsonl), where each io pair is
{"input": {...}, "output": {...}, "dummy_funcs": "...", ...} with native JSON
values instead of the HF {var, value} lists used by scripts.exebench_eval.

dummy_funcs are C stubs for external callees and can differ per io pair (on
the angha track), so a prediction is compiled once per distinct stub set.
--dataset is either a JSON list of rows with task_id or a .jsonl split, whose
rows get task_id "<file stem>:<line number>".

Usage:
    python3 -m scripts.exebench_eval_raw --predictions preds.json \\
        --dataset test_real_x86-64_O2.json --model-name a6/gpt --opt O2 --workers 16
"""
import argparse
import concurrent.futures
import json
from concurrent.futures import as_completed
from pathlib import Path

from scripts.eval_utils import atomic_write_json
from scripts.exebench import CompileError, Exebench_Wrapper, diff_io
from scripts.exebench_eval import _clean_signature, extract_function_headers_exeb, log_output, select_exe_wrapper

TRACKS = ('real', 'angha')


def pick_track(row, track='auto'):
    """Resolve 'auto' to the first track with io pairs (real first), else None."""
    if track != 'auto':
        return track
    for t in TRACKS:
        if row.get(f'{t}_io_pairs'):
            return t
    return None


def load_rows(exeb_dataset, task_ids=None):
    """Return {task_id: row}; for a .jsonl, only rows in `task_ids` are kept."""
    path = Path(exeb_dataset)
    if path.suffix != '.jsonl':
        return {row['task_id']: row for row in json.loads(path.read_text())}

    rows = {}
    with open(path) as f:
        for line_no, line in enumerate(f):
            task_id = f'{path.stem}:{line_no}'
            if task_ids is not None and task_id not in task_ids:
                continue
            row = json.loads(line)
            row['task_id'] = task_id
            rows[task_id] = row
            if task_ids is not None and len(rows) == len(task_ids):
                break
    return rows


def _process_prediction(item, idx, preds, row, track, debug, opt_level=None):
    logs = []
    tag = f'{item.get("task_id")} {item.get("type", "")} {idx}'
    preds['re-compile'] = False
    preds['re-execute'] = False
    try:
        pred_func = preds.get('pred_func', '').replace('func0', row.get('fname', ''))
        func_header, code = extract_function_headers_exeb(pred_func)
        if not func_header:
            logs.append(f'Could not extract header {tag}')
            return logs

        clean_signature = _clean_signature(func_header)

        io_pairs = row[f'{track}_io_pairs']
        deps = row.get(f'{track}_deps') or ''
        pred_deps = preds.get('pred_deps', '')
        cpp_wrapper = select_exe_wrapper(row, opt_level=opt_level, track=track)

        # One executable per distinct dummy_funcs stub set.
        wrappers = {}
        for pair in io_pairs:
            dummy_funcs = pair.get('dummy_funcs') or ''
            if dummy_funcs in wrappers:
                continue
            try:
                wrappers[dummy_funcs] = Exebench_Wrapper(
                    c_deps=deps + '\n' + dummy_funcs + '\n' + pred_deps + '\n' + code + '\n',
                    func_c_signature=clean_signature,
                    cpp_wrapper=cpp_wrapper)
            except CompileError as e:
                logs.append(f'Compile failed {tag}: {e}')
                return logs
    except Exception as e:
        logs.append(f'Failed before compile {tag}: {e}')
        return logs

    preds['re-compile'] = True

    try:
        for io_idx, pair in enumerate(io_pairs):
            observed_output = wrappers[pair.get('dummy_funcs') or ''](pair['input'])
            if debug:
                print(f"Debug\nI: {pair['input']}\nO: {observed_output}\nE: {pair['output']}\n")
            if not diff_io(observed_output=observed_output, expected_output=pair['output']):
                logs.append(f'Test case failed {tag} (io pair {io_idx})')
                return logs
        preds['re-execute'] = True
        logs.append(f'Test case passed {tag} ({len(io_pairs)} io pairs)')
    except Exception as e:
        logs.append(f'Execution failed {tag}: {e}')
    return logs


def exebench_validate_raw(model_name, file_path, exeb_dataset, parallel_workers=20, debug=False, resume=True,
                          flush_every=10, opt_level=None, track='auto'):
    """Raw-format counterpart of exebench_eval.exebench_validate_new.

    `track` is 'real', 'angha' or 'auto'. re-compile is False only when header
    extraction or the build fails; later errors only fail re-execute.
    """
    if track not in TRACKS + ('auto',):
        raise ValueError(f"track must be one of {TRACKS + ('auto',)}, got {track!r}")

    with open(file_path) as f:
        dataset = json.load(f)
    exebench_by_id = load_rows(exeb_dataset, task_ids={item.get('task_id') for item in dataset})

    parent_path = f'output_files/{model_name}'
    Path(parent_path).mkdir(parents=True, exist_ok=True)
    log_path = f'{parent_path}/exebench.log'

    work = []
    dataset_changed = False
    for item in dataset:
        if resume and item.get('no_oracle'):
            continue

        row = exebench_by_id.get(item.get('task_id'))
        if row is None:
            with open(log_path, 'a+') as df:
                log_output(f'No exebench row for {item.get("task_id")}', df)
            continue

        row_track = pick_track(row, track)
        if row_track is None or not row.get(f'{row_track}_io_pairs'):
            item['no_oracle'] = True
            dataset_changed = True
            for preds in item.get('predicted_c_funcs', []):
                preds['re-compile'] = None
                preds['re-execute'] = None
            with open(log_path, 'a+') as df:
                log_output(f'No {row_track or "real/angha"}_io_pairs for {item.get("task_id")} '
                           f'{item.get("type", "")} - skipping (no oracle)', df)
            continue

        item_opt_level = opt_level if opt_level is not None else item.get('opt')
        for idx, preds in enumerate(item.get('predicted_c_funcs', [])):
            if resume and 're-compile' in preds and 're-execute' in preds:
                continue
            work.append((item, idx, preds, row, row_track, item_opt_level))

    if not work:
        if dataset_changed:
            atomic_write_json(file_path, dataset)
        return

    completed = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [executor.submit(_process_prediction, item, idx, preds, row, row_track, debug, item_opt_level)
                       for item, idx, preds, row, row_track, item_opt_level in work]
            for fut in as_completed(futures):
                logs = fut.result()
                if logs:
                    with open(log_path, 'a+') as df:
                        for line in logs:
                            log_output(line, df)
                completed += 1
                if completed % flush_every == 0:
                    atomic_write_json(file_path, dataset)
    finally:
        atomic_write_json(file_path, dataset)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--predictions', required=True, help='JSON list of items with predicted_c_funcs (updated in place)')
    ap.add_argument('--dataset', required=True, help='JSON list of rows with task_id, or a raw .jsonl split')
    ap.add_argument('--model-name', required=True, help='log goes to output_files/<model-name>/exebench.log')
    ap.add_argument('--opt', default=None, help="O0/O1/O2/O3/Os; omitted = per-item 'opt' field")
    ap.add_argument('--track', default='auto', choices=TRACKS + ('auto',))
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--no-resume', action='store_true', help='re-evaluate predictions that already have results')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    exebench_validate_raw(args.model_name, args.predictions, args.dataset, parallel_workers=args.workers,
                          debug=args.debug, resume=not args.no_resume, opt_level=args.opt, track=args.track)

    items = json.loads(Path(args.predictions).read_text())
    preds = [p for item in items for p in item.get('predicted_c_funcs', []) if p.get('re-compile') is not None]
    n = len(preds)
    if n:
        rc = sum(1 for p in preds if p.get('re-compile'))
        rx = sum(1 for p in preds if p.get('re-execute'))
        print(f'\n{n} predictions: re-compile {rc} ({100 * rc / n:.1f}%), re-execute {rx} ({100 * rx / n:.1f}%)')


if __name__ == '__main__':
    main()
