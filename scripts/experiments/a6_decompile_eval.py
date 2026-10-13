#!/usr/bin/env python3
"""A.6 experiment: decompile each evaluable function of a split with an LLM, once from the assembly
alone (`asm_only`) and once with `data_section` and `func_call` added (`asm_context`), then
re-compile / re-execute the predictions with ExeBench-FHD's wrapper (reduced or full signature). Writes
<out-dir>/{asm_only,asm_context}.json, results.log and per_function.json; reruns resume.
Needs OPENAI_API_KEY.

    python3 -m scripts.experiments.a6_decompile_eval --limit 4      # smoke test
    python3 -m scripts.experiments.a6_decompile_eval --workers 16
"""
import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from scripts.eval_utils import atomic_write_json
from scripts.exebench_fhd_eval import exebench_fhd_validate
from scripts.experiments.a6_filter_subsets import DATASET_DIR, is_evaluable
from scripts.extract.extract_data_sections import data_section_by_arch
from scripts.extract.extract_func_call import func_call_by_arch

CONDITIONS = ('asm_only', 'asm_context')
ARCHS = ('x86', 'x86-64', 'ARM32', 'ARM64')

SYSTEM_PROMPT = """You are an expert reverse engineer. You decompile disassembled machine code back into \
the original C source code.

Rules:
- Output exactly one C function definition that, when compiled, behaves identically to the given assembly \
(same return type, same parameters in the same order with the same types, same side effects on memory \
reachable through pointer arguments).
- Name the function exactly as the label in the disassembly (e.g. `<foo>:` means the function is `foo`).
- Do not write helper functions, a `main`, or any test code. Do not add `#include` lines.
- If the function uses a struct, union or typedef that is not a standard C type, declare it above the \
function in the same code block.
- Reply with a single ```c fenced code block and nothing else."""

CONTEXT_EXPLANATION = """The following context was extracted from the same binary.

`data_section` is a hex dump of every non-code section (.rodata, .data, ...) the function reads, with \
its load address. Use it to recover literal constants (e.g. floating-point constants loaded via \
RIP-relative addressing, string literals, lookup tables).

`func_call` is the disassembly of a caller that sets up the arguments and calls this function (the \
callee appears as `func0` there). Use it to recover the number, types and ORDER of the parameters: \
the calling convention assigns integer/pointer and floating-point arguments to separate register \
sequences, so the callee alone does not reveal how they interleave."""


def build_prompt(row, arch, opt, condition):
    asm = row['objdump_asm'][arch][opt]
    parts = [f'Decompile this {arch} function (compiled with gcc -{opt}) into C.\n\n'
             f'### Assembly (objdump)\n```\n{asm}\n```']
    if condition == 'asm_context':
        data_section = ((data_section_by_arch(row.get('data_section'), opt) or {}).get(arch)
                        or '(the function references no data outside .text)')
        func_call = (func_call_by_arch(row.get('func_call'), opt) or {}).get(arch) or '(unavailable)'
        parts.append(CONTEXT_EXPLANATION)
        parts.append(f'### data_section\n```\n{data_section}\n```')
        parts.append(f'### func_call\n```\n{func_call}\n```')
    return '\n\n'.join(parts)


_CODE_BLOCK_RE = re.compile(r'```(?:c|C|cpp|c\+\+)?\s*\n(.*?)```', re.DOTALL)


def extract_c_code(text):
    """The first fenced code block (or the whole reply), without `#include` lines, since the
    prediction is compiled inside the wrapper's `extern "C" {}` block after `real_deps`."""
    m = _CODE_BLOCK_RE.search(text or '')
    code = m.group(1) if m else (text or '')
    code = '\n'.join(line for line in code.splitlines() if not line.lstrip().startswith('#include'))
    return code.strip() + '\n'


def call_model(client, model, prompt, reasoning_effort, retries=5):
    delay = 5
    for attempt in range(retries):
        try:
            kwargs = {'model': model, 'instructions': SYSTEM_PROMPT, 'input': prompt}
            if reasoning_effort:
                kwargs['reasoning'] = {'effort': reasoning_effort}
            return client.responses.create(**kwargs)
        except Exception as e:  # the SDK's own retries are exhausted by now
            if attempt == retries - 1:
                raise
            print(f'  API error ({e.__class__.__name__}: {e}); retrying in {delay}s', file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 2, 120)


def predict(client, model, rows, arch, opt, condition, pred_path, workers, reasoning_effort, show=False,
            flush_every=10):
    """Fill `pred_path` with one prediction per row, reusing entries made with the same model,
    reasoning effort, arch, opt and prompt text."""
    items = json.loads(pred_path.read_text()) if pred_path.exists() else []
    config = {'model': model, 'reasoning_effort': reasoning_effort, 'arch': arch, 'opt': opt}
    prompts = {row['task_id']: build_prompt(row, arch, opt, condition) for row in rows}

    def is_cached(it):
        return (it.get('predicted_c_funcs')
                and it.get('prompt') == prompts.get(it['task_id'])
                and it.get('system_prompt') == SYSTEM_PROMPT
                and all(it.get(k) == v for k, v in config.items()))

    # Keep predictions for functions outside this run (e.g. from a run without --limit).
    in_run = [it for it in items if it['task_id'] in prompts]
    kept_other = [it for it in items if it['task_id'] not in prompts and it.get('predicted_c_funcs')]
    stale = sum(1 for it in in_run if it.get('predicted_c_funcs') and not is_cached(it))
    items = kept_other + [it for it in in_run if is_cached(it)]  # also drops earlier failed attempts
    done = {it['task_id'] for it in items if it['task_id'] in prompts}
    todo = [row for row in rows if row['task_id'] not in done]
    print(f'[{pred_path.stem}] {len(done)} cached, {stale} stale (different config) discarded, '
          f'{len(todo)} to query')
    if todo and client is None:
        sys.exit(f'{len(todo)} predictions to query but OPENAI_API_KEY is not set')

    def one(row):
        prompt = prompts[row['task_id']]
        try:
            resp = call_model(client, model, prompt, reasoning_effort)
        except Exception as e:
            return {'task_id': row['task_id'], 'error': f'{e.__class__.__name__}: {e}'}
        text = resp.output_text
        usage = resp.usage.model_dump() if resp.usage else None
        if show:
            print(f"\n===== {row['task_id']} ({row['fname']}) [{condition}] =====\n"
                  f"--- ground truth ---\n{row['func_def']}\n--- model output ---\n{text}\n", flush=True)
        return {
            'task_id': row['task_id'],
            'fname': row['fname'],
            'type': condition,
            **config,
            'ground_truth': row['func_def'],
            'system_prompt': SYSTEM_PROMPT,
            'prompt': prompt,
            'raw_response': text,
            'usage': usage,
            'response': resp.model_dump(),
            'predicted_c_funcs': [{'pred_func': extract_c_code(text)}],
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for n, result in enumerate(ex.map(one, todo), 1):
            if 'error' in result:
                print(f'  FAILED {result["task_id"]}: {result["error"]}', file=sys.stderr)
            else:
                items.append(result)
            if n % flush_every == 0:
                atomic_write_json(pred_path, items)
    atomic_write_json(pred_path, items)
    return items


def summarize(items, rows):
    """Re-compile / re-execute counts over `rows`; a row without a prediction counts as a failure."""
    by_id = {it['task_id']: it for it in items if it.get('predicted_c_funcs')}
    preds = [by_id[r['task_id']]['predicted_c_funcs'][0] for r in rows if r['task_id'] in by_id]
    return {
        'records': len(rows),
        're_compile': sum(1 for p in preds if p.get('re-compile')),
        're_execute': sum(1 for p in preds if p.get('re-execute')),
    }


def per_function_outcomes(out_dir, rows, metric='re-execute', save=True):
    """Whether each function passed `metric` with and without context (saved to per_function.json)."""
    passed = {}
    for condition in CONDITIONS:
        items = json.loads((out_dir / f'{condition}.json').read_text())
        passed[condition] = {it['task_id']: bool(it['predicted_c_funcs'][0].get(metric))
                             for it in items if it.get('predicted_c_funcs')}
    outcomes = []
    for r in rows:
        a = passed['asm_only'].get(r['task_id'], False)
        c = passed['asm_context'].get(r['task_id'], False)
        outcome = 'both' if a and c else 'asm_only' if a else 'asm_context' if c else 'neither'
        outcomes.append({'task_id': r['task_id'], 'fname': r['fname'],
                         'asm_only': a, 'asm_context': c, 'passed_with': outcome})
    if save:
        atomic_write_json(out_dir / 'per_function.json', outcomes)
    return outcomes


def subsets(rows, arch, opt):
    """{name: rows} for the results.log breakdown."""
    return {
        'all': rows,
        'data_section': [r for r in rows
                         if (data_section_by_arch(r.get('data_section'), opt) or {}).get(arch)],
        'common_all_archs': [r for r in rows if r['common_all_archs']],
    }


def write_results_log(out_dir, rows, summaries, args, per_function=True, dropped_unused=0):
    """Print and save results.log: rates per subset and condition (`summaries` is
    {subset: {condition: summarize(...)}}), plus each function's outcome."""
    log = [f'model: {args.model}   dataset: {args.dataset}   arch: {args.arch}   opt: {args.opt}   '
           f'functions: {len(rows)}   ({time.strftime("%Y-%m-%d %H:%M:%S")})',
           f'({dropped_unused} unused_param functions dropped)',
           '',
           f'{"subset":<18} {"condition":<12} {"n":>6} {"re-compile":>11} {"re-execute":>11}']
    for subset, by_condition in summaries.items():
        for condition, c in by_condition.items():
            n = c['records'] or 1
            log.append(f'{subset:<18} {condition:<12} {c["records"]:>6} '
                       f'{100 * c["re_compile"] / n:>10.1f}% {100 * c["re_execute"] / n:>10.1f}%')

    if per_function:
        compiled = per_function_outcomes(out_dir, rows, metric='re-compile', save=False)
        executed = per_function_outcomes(out_dir, rows)
        log += ['', f'per function (details in {out_dir / "per_function.json"}):',
                f'  {"task_id":<18} {"function":<32} {"re-compile with":<16} re-execute with']
        for rc, rx in zip(compiled, executed):
            log.append(f'  {rx["task_id"]:<18} {rx["fname"]:<32} {rc["passed_with"]:<16} {rx["passed_with"]}')

    text = '\n'.join(log) + '\n'
    print('\n' + text)
    (out_dir / 'results.log').write_text(text)
    print(f'results saved to {out_dir / "results.log"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='test_real', help=f'release split, read from {DATASET_DIR}/<split>.jsonl')
    ap.add_argument('--dataset', default=None, help='a .jsonl split file; overrides --split')
    ap.add_argument('--model', default='gpt-6-sol')
    ap.add_argument('--reasoning-effort', default=None, help='e.g. low/medium/high; omitted = model default')
    ap.add_argument('--arch', default='x86-64')
    ap.add_argument('--opt', default='O2')
    ap.add_argument('--limit', type=int, default=None, help='only the first N evaluable functions (smoke test)')
    ap.add_argument('--workers', type=int, default=8, help='concurrent API requests')
    ap.add_argument('--eval-workers', type=int, default=24)
    ap.add_argument('--rescore', action='store_true',
                    help='re-run re-compile / re-execute on predictions that already have results')
    ap.add_argument('--out-dir', default=None, help='default: output_files/a6/<model>_<split>_<arch>_<opt>[_limitN]')
    ap.add_argument('--skip-eval', action='store_true')
    ap.add_argument('--show', action='store_true', help='print each ground truth and model reply as it arrives')
    args = ap.parse_args()

    args.dataset = args.dataset or str(DATASET_DIR / f'{args.split}.jsonl')
    split = Path(args.dataset).stem
    out_dir = Path(args.out_dir or f'output_files/a6/{args.model}_{split}_{args.arch}_{args.opt}'
                   + (f'_limit{args.limit}' if args.limit else ''))
    out_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    # Without a key, only cached predictions can be used (e.g. with --rescore).
    client = OpenAI(api_key=api_key, max_retries=3, timeout=600) if api_key else None

    # task_id = <split>:<line number>, matching exebench_fhd_validate's naming for a .jsonl dataset.
    # unused_param rows are scored with ExeBench-FHD's wrapper (passes the reduced or the full
    # signature), so a prediction that recovers the full signature from func_call is not rejected.
    drop_unused = False
    rows, skipped, dropped_unused = [], 0, 0
    with open(args.dataset) as f:
        for line_no, line in enumerate(f):
            row = json.loads(line)
            if not is_evaluable(row, args.arch, args.opt):
                skipped += 1
                continue
            if drop_unused and row.get('unused_param'):
                dropped_unused += 1
                continue
            row['task_id'] = f'{split}:{line_no}'
            row['common_all_archs'] = all(is_evaluable(row, a, args.opt) for a in ARCHS)
            rows.append(row)
            if args.limit and len(rows) >= args.limit:
                break
    print(f'{args.dataset}: {len(rows)} evaluable functions ({skipped} skipped: no io pairs / wrapper / '
          f'objdump_asm[{args.arch}][{args.opt}] / func_call[{args.arch}]; '
          f'{dropped_unused} unused_param dropped)')

    groups = subsets(rows, args.arch, args.opt)
    summaries = {name: {} for name in groups}
    for condition in CONDITIONS:
        pred_path = out_dir / f'{condition}.json'
        predict(client, args.model, rows, args.arch, args.opt, condition, pred_path,
                args.workers, args.reasoning_effort, show=args.show)
        if not args.skip_eval:
            # model_name makes the evaluator log to <out-dir>/exebench.log.
            exebench_fhd_validate(
                model_name=str(out_dir.relative_to('output_files')),
                file_path=str(pred_path),
                exeb_dataset=args.dataset,
                parallel_workers=args.eval_workers,
                opt_level=args.opt,
                resume=not args.rescore,
            )
        items = json.loads(pred_path.read_text())
        for name, group in groups.items():
            summaries[name][condition] = summarize(items, group)

    write_results_log(out_dir, rows, summaries, args, per_function=not args.skip_eval,
                      dropped_unused=dropped_unused)


if __name__ == '__main__':
    main()
