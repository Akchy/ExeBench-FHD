#!/usr/bin/env python3
"""ExeBenchPlus evaluation: re-compile / re-execute decompiled C predictions against rows
in the released ExeBenchPlus format (datasets_jsonl_v2/*.jsonl), where each io pair is
{"input": {...}, "output": {...}, "dummy_funcs": "...", ...} with native JSON
values.

dummy_funcs are C stubs for external callees and can differ per io pair (on
the angha track), so a prediction is compiled once per distinct stub set.
--dataset is either a JSON list of rows with task_id or a .jsonl split, whose
rows get task_id "<file stem>:<line number>".

Usage:
    python3 -m scripts.exebenchplus_eval --predictions preds.json \\
        --dataset test_real_x86-64_O2.json --model-name a6/gpt --opt O2 --workers 16
"""
import argparse
import concurrent.futures
import json
import re
from concurrent.futures import as_completed
from pathlib import Path

from scripts.eval_utils import atomic_write_json
from scripts.exebench import CompileError, Exebench_Wrapper, diff_io

# At O0 every parameter is spilled to the stack even if unused, so the
# original fixed-arity *_exe_wrapper is always correct.
_NO_DROP_OPT_LEVELS = {'O0'}


def select_exe_wrapper(row, opt_level, track='real'):
    """Return the exe wrapper for `track` ('real'/'angha') at `opt_level`.

    Above O0 the compiler may drop unused parameters, so for rows with
    unused_param set the arity-adapted opt_{track}_exe_wrapper is used when
    present. opt_level=None is treated as O0.
    """
    wrapper = row.get(f'{track}_exe_wrapper', '')
    no_drop = opt_level is None or opt_level.upper() in _NO_DROP_OPT_LEVELS
    if row.get('unused_param') and not no_drop:
        opt_wrapper = row.get(f'opt_{track}_exe_wrapper')
        if opt_wrapper:
            return opt_wrapper
    return wrapper


def select_exe_wrappers(row, opt_level, track='real'):
    """ExeBenchPlus's wrapper: the candidate wrappers for `track` at `opt_level`, in order.

    Above O0 a row with unused_param set has two valid calling forms, the arity-reduced
    opt_{track}_exe_wrapper (what the optimized code shows) and the full-signature
    {track}_exe_wrapper (what a caller passes). A prediction is correct if it passes either.
    Every other row has just the one wrapper.
    """
    full = row.get(f'{track}_exe_wrapper', '')
    reduced = select_exe_wrapper(row, opt_level, track)
    return [reduced] if reduced == full else [reduced, full]


def log_output(msg, f):
    print(msg)
    f.write(f"{msg}\n")


def _clean_signature(func_header):
    sig = re.sub(r'\b(const|static|extern)\b', '', func_header)
    return re.sub(r'[ \t]{2,}', ' ', sig)


# Up to two levels of nested parens, e.g. "float (*theta)[5][5]".
_ARG_LIST_RE = r"(?:[^()]|\((?:[^()]|\([^()]*\))*\))*"
# GCC linemarkers between ')' and '{'. '[#]' keeps re.VERBOSE from eating it.
_LINEMARKER_RE = r"(?:\s*[#][^\n]*\n)*"
# K&R parameter declarations between ')' and '{', e.g. "f(a, b) int a; char *b; {".
# Kept in sync with _KR_DECL_LINE_RE.
_KR_DECL_BLOCK_RE = (
    r"(?:\s*(?:register\s+)?[A-Za-z_][\w\s]*?\s*"
    r"(?:\*\s*)?[A-Za-z_]\w*(?:\s*,\s*(?:\*\s*)?[A-Za-z_]\w*)*\s*;)*"
)

_KR_PARAM_LIST_RE = re.compile(r"\(([^()]*)\)\s*\Z", re.DOTALL)
# group(1) = base type; group(2) = names. The '*' binds to each name, since
# "int *x, y;" makes only x a pointer.
_KR_DECL_LINE_RE = re.compile(
    r"(?:register\s+)?([A-Za-z_][\w\s]*?)\s*"
    r"((?:\*\s*)?[A-Za-z_]\w*(?:\s*,\s*(?:\*\s*)?[A-Za-z_]\w*)*)\s*;"
)
_KR_DECL_NAME_RE = re.compile(r"\A(\*+)\s*([A-Za-z_]\w*)\Z")


def _rewrite_kr_header(header, decl_block):
    """Rewrite a K&R "name(a, b)" header plus its declaration block into a
    prototype. Returns None unless every parameter has a declared type.
    """
    if not decl_block.strip():
        return None

    m = _KR_PARAM_LIST_RE.search(header)
    if not m:
        return None
    param_names = [p.strip() for p in m.group(1).split(",") if p.strip()]
    if not param_names or not all(re.fullmatch(r"[A-Za-z_]\w*", p) for p in param_names):
        return None

    types_by_name = {}
    for line in _KR_DECL_LINE_RE.finditer(decl_block):
        decl_type = " ".join(line.group(1).split())
        for raw_name in line.group(2).split(","):
            raw_name = raw_name.strip()
            star_match = _KR_DECL_NAME_RE.match(raw_name)
            if star_match:
                stars, name = star_match.groups()
                types_by_name[name] = f"{decl_type} {stars}"
            else:
                types_by_name[raw_name] = decl_type

    if not all(p in types_by_name for p in param_names):
        return None

    prefix = header[: m.start()]
    typed_params = ", ".join(f"{types_by_name[p]} {p}" for p in param_names)
    return f"{prefix}({typed_params})"


def extract_function_headers_exeb(code):
    """Find the first function definition in `code`.

    Returns (header, code with const/static stripped from that header), or
    (False, False) if no header could be extracted.
    """
    pattern = re.compile(
        r"""
        (?P<header>
            [A-Za-z_][\w\s\*\[\]\(\)]*?   # qualifiers + return type
            \b[A-Za-z_]\w*                     # function name
            \s*\(
                """ + _ARG_LIST_RE + r"""       # argument list (handles nested-paren declarators)
            \)
            """ + _LINEMARKER_RE + r"""
        )
        (?P<kr_decls>""" + _KR_DECL_BLOCK_RE + r""")   # optional K&R parameter-type declarations
        \s*
        (?=\{)                                  # must be a function definition
        """,
        re.VERBOSE | re.MULTILINE | re.DOTALL,
    )

    match = pattern.search(code)
    if not match:
        return False, False

    full_header = match.group('header')
    kr_decls = match.group('kr_decls')
    header_end = match.end('kr_decls')

    if kr_decls.strip():
        rewritten = _rewrite_kr_header(full_header, kr_decls)
        if rewritten is None:
            return False, False
        full_header = rewritten

    new_func_header = re.sub(r'\b(const|static)\b', '', full_header)
    new_func_header = re.sub(r'[ \t]{2,}', ' ', new_func_header)

    new_code = code[:match.start('header')] + new_func_header + code[header_end:]

    return full_header, new_code


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
        cpp_wrappers = select_exe_wrappers(row, opt_level=opt_level, track=track)
    except Exception as e:
        logs.append(f'Failed before compile {tag}: {e}')
        return logs

    # ExeBenchPlus's wrapper: correct if it passes either calling form (reduced or full).
    for cpp_wrapper in cpp_wrappers:
        # One executable per distinct dummy_funcs stub set.
        wrappers = {}
        try:
            for pair in io_pairs:
                dummy_funcs = pair.get('dummy_funcs') or ''
                if dummy_funcs not in wrappers:
                    wrappers[dummy_funcs] = Exebench_Wrapper(
                        c_deps=deps + '\n' + dummy_funcs + '\n' + pred_deps + '\n' + code + '\n',
                        func_c_signature=clean_signature,
                        cpp_wrapper=cpp_wrapper)
        except CompileError as e:
            logs.append(f'Compile failed {tag}: {e}')
            continue

        preds['re-compile'] = True
        try:
            for io_idx, pair in enumerate(io_pairs):
                observed_output = wrappers[pair.get('dummy_funcs') or ''](pair['input'])
                if debug:
                    print(f"Debug\nI: {pair['input']}\nO: {observed_output}\nE: {pair['output']}\n")
                if not diff_io(observed_output=observed_output, expected_output=pair['output']):
                    logs.append(f'Test case failed {tag} (io pair {io_idx})')
                    break
            else:
                preds['re-execute'] = True
                logs.append(f'Test case passed {tag} ({len(io_pairs)} io pairs)')
                return logs
        except Exception as e:
            logs.append(f'Execution failed {tag}: {e}')
    return logs


def exebenchplus_validate(model_name, file_path, exeb_dataset, parallel_workers=20, debug=False, resume=True,
                          flush_every=10, opt_level=None, track='auto'):
    """Evaluate every prediction in `file_path`, writing re-compile/re-execute onto
    each predicted_c_funcs entry in place. Logs go to output_files/<model_name>/exebench.log.

    opt_level applies one level to all items; None reads each item's 'opt' field,
    and an item with neither is treated as O0.
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


def example_evaluate_dataset_at_o0(exeb_dataset, limit=None, debug=False, track='real'):
    """Run each row's ground-truth func_def against its own io pairs at O0
    and print PASS/FAIL per pair (the counterpart of upstream examples/basic.py).
    """
    rows = list(load_rows(exeb_dataset).values())
    for row in rows[:limit] if limit else rows:
        io_pairs = row.get(f'{track}_io_pairs')
        if not io_pairs:
            continue

        func_header, code = extract_function_headers_exeb(row.get('func_def', ''))
        if not func_header:
            print(f"Could not extract header {row.get('task_id')}")
            continue
        clean_signature = _clean_signature(func_header)
        cpp_wrapper = select_exe_wrappers(row, opt_level='O0', track=track)[0]

        wrappers = {}
        try:
            for pair in io_pairs:
                dummy_funcs = pair.get('dummy_funcs') or ''
                if dummy_funcs not in wrappers:
                    wrappers[dummy_funcs] = Exebench_Wrapper(
                        c_deps=(row.get(f'{track}_deps') or '') + '\n' + dummy_funcs + '\n' + code + '\n',
                        func_c_signature=clean_signature,
                        cpp_wrapper=cpp_wrapper)
        except CompileError as e:
            print(f"Compile failed {row.get('task_id')}: {e}")
            continue

        for io_idx, pair in enumerate(io_pairs):
            observed_output = wrappers[pair.get('dummy_funcs') or ''](pair['input'])
            if debug:
                print(f"Input: {pair['input']}\nObserved: {observed_output}\nExpected: {pair['output']}")
            passed = diff_io(observed_output=observed_output, expected_output=pair['output'])
            print(row.get('task_id'), io_idx, 'PASS' if passed else 'FAIL')


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

    exebenchplus_validate(args.model_name, args.predictions, args.dataset, parallel_workers=args.workers,
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
