"""Re-compile / re-execute evaluation of decompiled C predictions against
ExeBench rows in the Hugging Face format ({var, value} io pairs).

Entry points: exebench_validate_new() for a predictions file, and
example_evaluate_dataset_at_o0() as a minimal ground-truth example.
"""
import re
import json
import concurrent.futures
from pathlib import Path
from concurrent.futures import as_completed
from scripts.exebench import Exebench_Wrapper, diff_io, exebench_dict_to_dict, CompileError
from scripts.eval_utils import atomic_write_json

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
    """ExeBench-FHD's wrapper: the candidate wrappers for `track` at `opt_level`, in order.

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


def _process_prediction(item, idx, preds, row, debug, opt_level=None):
    logs = []
    try:
        pred_func: str = preds.get('pred_func', '')
        pred_func = pred_func.replace('func0', row.get('fname', ''))
        func_header, code = extract_function_headers_exeb(pred_func)
        if not func_header:
            preds['re-compile'] = False
            preds['re-execute'] = False
            logs.append(f'Could not extract header {item.get("task_id")} {idx}')
            return logs

        clean_signature = _clean_signature(func_header)
        pred_deps = preds.get('pred_deps', '')
        io_inputs = row['real_io_pairs']['input']
        io_outputs = row['real_io_pairs']['output']
        base = row.get('real_deps', '') + '\n' + pred_deps + '\n' + code + '\n'

        # ExeBench-FHD's wrapper: correct if it passes either calling form (reduced or full).
        preds['re-compile'] = False
        preds['re-execute'] = False
        for cpp_wrapper in select_exe_wrappers(row, opt_level=opt_level, track='real'):
            try:
                real_wrapper = Exebench_Wrapper(c_deps=base, func_c_signature=clean_signature,
                                                cpp_wrapper=cpp_wrapper)
            except CompileError as e:
                logs.append(f'Compile failed {item.get("task_id")} {item.get("type","")} {idx}: {e}')
                continue

            preds['re-compile'] = True
            try:
                all_pass = True
                for io_idx in range(len(io_inputs)):
                    observed_output = real_wrapper(exebench_dict_to_dict(io_inputs[io_idx]))

                    if debug:
                        print(f"Debug\nR: {io_inputs[io_idx]}\n{observed_output}\n\n")

                    if not diff_io(observed_output=observed_output,
                                   expected_output=exebench_dict_to_dict(io_outputs[io_idx])):
                        all_pass = False
                        break
            except Exception as e:
                logs.append(f'Execution failed {item.get("task_id")} {item.get("type","") } {idx}: {e}')
                continue

            if all_pass:
                preds['re-execute'] = True
                logs.append(f'Test case passed {item.get("task_id")} {item.get("type","")} {idx} ({len(io_inputs)} io pairs)')
                break
            logs.append(f'Test case failed {item.get("task_id")} {item.get("type","")} {idx} (io pair {io_idx})')

    except Exception as e:
        preds['re-compile'] = False
        preds['re-execute'] = False
        logs.append(f'Failed {item.get("task_id")} {item.get("type","") } {idx}: {e}')

    return logs


def exebench_validate_new(model_name, file_path, exeb_dataset, parallel_workers=20, debug=False, resume=True,
                           flush_every=10, opt_level=None):
    """Evaluate every prediction in `file_path`, writing re-compile/re-execute
    onto each predicted_c_funcs entry in place. Logs go to
    output_files/<model_name>/exebench.log.

    opt_level applies one level to all items; None reads each item's 'opt'
    field, and an item with neither is treated as O0.
    """

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    with open(exeb_dataset, 'r') as f1:
        exebench_dataset = json.load(f1)

    exebench_by_id = {row['task_id']: row for row in exebench_dataset}

    parent_path = f'output_files/{model_name}'
    Path(parent_path).mkdir(parents=True, exist_ok=True)
    log_path = f'{parent_path}/exebench.log'

    # One flat work list across all items; resume skips individual predictions.
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

        if row.get('real_io_pairs') is None:
            # Most train rows have no io pairs. Use None rather than False so
            # these are not counted as broken predictions downstream.
            item['no_oracle'] = True
            dataset_changed = True
            for preds in item.get('predicted_c_funcs', []):
                preds['re-compile'] = None
                preds['re-execute'] = None
            with open(log_path, 'a+') as df:
                log_output(
                    f'No real_io_pairs for {item.get("task_id")} {item.get("type","")} - skipping (no oracle)', df
                )
            continue

        item_opt_level = opt_level if opt_level is not None else item.get('opt')
        for idx, preds in enumerate(item.get('predicted_c_funcs', [])):
            if resume and 're-compile' in preds and 're-execute' in preds:
                continue
            work.append((item, idx, preds, row, item_opt_level))

    if not work:
        if dataset_changed:
            atomic_write_json(file_path, dataset)
        return

    completed = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [executor.submit(_process_prediction, item, idx, preds, row, debug, item_opt_level)
                       for item, idx, preds, row, item_opt_level in work]

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


def example_evaluate_dataset_at_o0(exeb_dataset_path, limit=None, debug=False, track='real'):
    """Run each row's ground-truth func_def against its own io pairs at O0
    and print PASS/FAIL per pair (the counterpart of upstream examples/basic.py).
    """
    with open(exeb_dataset_path, 'r') as f:
        dataset = json.load(f)

    io_pairs_field = f'{track}_io_pairs'
    deps_field = f'{track}_deps'

    for row in dataset[:limit] if limit else dataset:
        io_pairs = row.get(io_pairs_field)
        if not io_pairs:
            continue

        func_header, code = extract_function_headers_exeb(row.get('func_def', ''))
        if not func_header:
            print(f"Could not extract header {row.get('task_id')}")
            continue
        clean_signature = _clean_signature(func_header)
        cpp_wrapper = select_exe_wrapper(row, opt_level='O0', track=track)

        try:
            wrapper = Exebench_Wrapper(
                c_deps=row.get(deps_field, '') + '\n' + code + '\n',
                func_c_signature=clean_signature,
                cpp_wrapper=cpp_wrapper,
            )
        except CompileError as e:
            print(f"Compile failed {row.get('task_id')}: {e}")
            continue

        io_inputs = io_pairs['input']
        io_outputs = io_pairs['output']
        for io_idx in range(len(io_inputs)):
            observed_output = wrapper(exebench_dict_to_dict(io_inputs[io_idx]))
            expected_output = exebench_dict_to_dict(io_outputs[io_idx])

            if debug:
                print(f"Input: {io_inputs[io_idx]}\nObserved: {observed_output}\nExpected: {expected_output}")

            passed = diff_io(observed_output=observed_output, expected_output=expected_output)
            print(row.get('task_id'), io_idx, 'PASS' if passed else 'FAIL')
