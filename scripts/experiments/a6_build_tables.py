#!/usr/bin/env python3
"""Build the A.6 LaTeX tables (gpt-6-sol on test_real at O2, asm_only vs asm_context) from the
evaluated predictions in output_files/a6/gpt-6-sol_test_real_<arch>_O2/.

    python -m scripts.experiments.a6_build_tables [--out experiments.tex] [--workers 24]
        [--unused-param-json output_files/a6/unused_param_wrapper_scoring.json]
"""
import argparse
import concurrent.futures
import json
import math

from scripts.exebench_eval_raw import _process_prediction
from scripts.experiments.a6_filter_subsets import is_evaluable
from scripts.extract.extract_data_sections import data_section_by_arch
from scripts.survey.survey_mixed_arg_order import has_mixed_arg_order

ARCHS = ('x86-64', 'x86', 'ARM64', 'ARM32')
CONDITIONS = ('asm_only', 'asm_context')
METRICS = ('re-compile', 're-execute')
OPT = 'O2'
PRED_DIR = 'output_files/a6/gpt-6-sol_test_real_{arch}_O2'


def load_rows():
    rows = {}
    with open('datasets_jsonl_v2/test_real.jsonl') as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            r['task_id'] = f'test_real:{i}'
            rows[r['task_id']] = r
    return rows


def load_preds(arch):
    """({condition: {task_id: item}}, {condition: {task_id: scored prediction}})."""
    d = PRED_DIR.format(arch=arch)
    preds = {}
    for c in CONDITIONS:
        with open(f'{d}/{c}.json') as f:
            preds[c] = {it['task_id']: it for it in json.load(f)}
    stored = {c: {t: it['predicted_c_funcs'][0] for t, it in preds[c].items()} for c in CONDITIONS}
    return preds, stored


def rates(results, ids):
    """{condition: (re-compile %, re-execute %)} over `ids`."""
    n = len(ids) or 1
    return {c: tuple(100 * sum(bool(results[c][t][k]) for t in ids) / n for k in METRICS)
            for c in CONDITIONS}


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value (binomial test on the b + c discordant pairs)."""
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def mcnemar(results, ids, metric='re-execute'):
    """(b, c, p): b = functions passing only with context, c = only without."""
    b = sum(bool(results['asm_context'][t][metric]) and not results['asm_only'][t][metric] for t in ids)
    c = sum(bool(results['asm_only'][t][metric]) and not results['asm_context'][t][metric] for t in ids)
    return b, c, mcnemar_p(b, c)


def rescore_plain_wrapper(preds, rows, ids, workers):
    """Re-score `ids`' predictions against the full-arity real_exe_wrapper, without touching
    the stored prediction files (unused_param=False makes select_exe_wrapper pick it)."""
    out, jobs = {c: {} for c in CONDITIONS}, []
    for c in CONDITIONS:
        for t in ids:
            p = {'pred_func': preds[c][t]['predicted_c_funcs'][0]['pred_func']}
            out[c][t] = p
            jobs.append((preds[c][t], p, dict(rows[t], unused_param=False)))
    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda j: _process_prediction(j[0], 0, j[1], j[2], 'real', False, OPT), jobs))
    return out


def score_unused_param(stored, plain, ids):
    """Rates against the reduced wrapper, the full wrapper and either one, with McNemar tests."""
    def pct(s):
        return round(100 * sum(s.values()) / (len(ids) or 1), 1)

    def mc(x, y):
        b = sum(y[t] and not x[t] for t in ids)
        c = sum(x[t] and not y[t] for t in ids)
        return b, c, mcnemar_p(b, c)

    out = {'n': len(ids)}
    for metric in METRICS:
        red = {c: {t: bool(stored[c][t][metric]) for t in ids} for c in CONDITIONS}
        full = {c: {t: bool(plain[c][t][metric]) for t in ids} for c in CONDITIONS}
        either = {c: {t: red[c][t] or full[c][t] for t in ids} for c in CONDITIONS}
        out[metric] = dict(
            reduced=(pct(red['asm_only']), pct(red['asm_context'])),
            full=(pct(full['asm_only']), pct(full['asm_context'])),
            either=(pct(either['asm_only']), pct(either['asm_context'])),
            mc_red_vs_full=mc(red['asm_only'], full['asm_context']),
            mc_either=mc(either['asm_only'], either['asm_context']),
            both_wrappers_asm=sum(red['asm_only'][t] and full['asm_only'][t] for t in ids))
    return out


def compute(workers):
    """(tables, unused_param scoring): tables[name][arch] = (n, rates)."""
    rows = load_rows()
    common = {t for t, r in rows.items()
              if not r.get('unused_param') and all(is_evaluable(r, a, OPT) for a in ARCHS)}
    tables = {name: {} for name in ('overall', 'common', 'data_section', 'func_call', 'unused_opt', 'unused_plain')}
    unused_scoring = {}
    for arch in ARCHS:
        preds, stored = load_preds(arch)
        evaluable = [t for t in preds['asm_only'] if is_evaluable(rows[t], arch, OPT)]
        kept = [t for t in evaluable if not rows[t].get('unused_param')]
        unused = [t for t in evaluable if rows[t].get('unused_param')]
        groups = {
            'overall': kept,
            'common': [t for t in kept if t in common],
            'data_section': [t for t in kept
                             if (data_section_by_arch(rows[t].get('data_section'), OPT) or {}).get(arch)],
            'func_call': [t for t in kept if has_mixed_arg_order(rows[t].get('signature'))],
            'unused_opt': unused,
        }
        for name, ids in groups.items():
            tables[name][arch] = (len(ids), rates(stored, ids))
            print(f'{arch} {name:<13} re-execute McNemar: +ctx only %d, asm only %d, p=%.2g'
                  % mcnemar(stored, ids))
        plain = rescore_plain_wrapper(preds, rows, unused, workers)
        tables['unused_plain'][arch] = (len(unused), rates(plain, unused))
        print(f'{arch} {"unused_plain":<13} re-execute McNemar: +ctx only %d, asm only %d, p=%.2g'
              % mcnemar(plain, unused))
        unused_scoring[arch] = score_unused_param(stored, plain, unused)
        print(f'{arch}: done', flush=True)
    return tables, unused_scoring


def fmt_row(label, n, r):
    # Round first so the printed delta is exactly the difference of the printed rates.
    (rc_a, rx_a), (rc_c, rx_c) = [tuple(round(v, 1) for v in r[c]) for c in CONDITIONS]
    return (f'    {label} & {n} & {rc_a:.1f} & {rc_c:.1f} & {rx_a:.1f} & {rx_c:.1f} & '
            f'{rx_c - rx_a:+.1f} \\\\')


HEADER = r"""  \begin{tabular}{%s}
    \hline
    %s & & \multicolumn{2}{c}{\textbf{Re-compile (\%%)}} & \multicolumn{3}{c}{\textbf{Re-execute (\%%)}} \\
    %s & $n$ & asm & +ctx & asm & +ctx & $\Delta$ \\
    \hline"""


def table(body_lines, caption, label, cols='lrrrrrr', head=('\\textbf{Arch}', '')):
    return '\n'.join([
        r'\begin{table}[h]',
        r'  \centering',
        r'  \small',
        r'  \resizebox{\columnwidth}{!}{%',
        HEADER % (cols, head[0], head[1]),
        *body_lines,
        r'    \hline',
        r'  \end{tabular}}',
        f'  \\caption{{{caption}}}',
        f'  \\label{{{label}}}',
        r'\end{table}',
        '',
    ])


def to_tex(t):
    common_n = t['common'][ARCHS[0]][0]
    overall = [fmt_row(a, *t['overall'][a]) for a in ARCHS]
    overall += [r'    \hline', r'    \multicolumn{7}{l}{\textit{Functions evaluable on all four architectures}} \\']
    overall += [fmt_row(a, *t['common'][a]) for a in ARCHS]

    unused = []
    for a in ARCHS:
        unused.append(fmt_row(f'{a} & reduced', *t['unused_opt'][a]))
        unused.append(fmt_row(' & full', *t['unused_plain'][a]))
        if a != ARCHS[-1]:
            unused.append(r'    \hline')

    caption_common = (r'\texttt{gpt-6-sol} decompilation of \texttt{test\_real} at \texttt{-O2}: '
                      r'assembly only (asm) vs.\ assembly plus \texttt{data\_section} and \texttt{func\_call} '
                      r'(+ctx). Functions with an unused parameter are excluded (see '
                      r'\cref{tab:a6-unused-param}). ')
    return '\n'.join([
        '% Generated by scripts/experiments/a6_build_tables.py - do not edit by hand.',
        '% Requires \\usepackage{graphicx} and \\usepackage{cleveref}.',
        '',
        table(overall, caption_common + 'Top: every function that can be prompted and evaluated on that '
              f'architecture; bottom: the {common_n} functions evaluable on all four, so rates are comparable '
              r'across architectures. $\Delta$ is the re-execute gain from the context.',
              'tab:a6-overall'),
        table([fmt_row(a, *t['data_section'][a]) for a in ARCHS],
              r'Functions that read bytes outside \texttt{.text} (non-empty \texttt{data\_section}), '
              r'e.g.\ floating-point constants or lookup tables in \texttt{.rodata}. Same setup as '
              r'\cref{tab:a6-overall}. ARM keeps most constants in literal pools inside \texttt{.text}, '
              r'hence the smaller $n$.',
              'tab:a6-data-section'),
        table([fmt_row(a, *t['func_call'][a]) for a in ARCHS],
              r'Functions whose signature mixes floating-point and integer/pointer parameters, where the '
              r'callee alone does not fix the parameter order and the caller in \texttt{func\_call} does. '
              r'32-bit x86 passes all arguments on the stack in order, so the callee already reveals it '
              r'and the gain is smallest there.',
              'tab:a6-func-call'),
        table(unused,
              r'Functions with a parameter the \texttt{-O2} code never uses, scored against the '
              r'arity-reduced wrapper (\texttt{opt\_real\_exe\_wrapper}, unused parameters removed from the '
              r'call) and the full-arity wrapper (\texttt{real\_exe\_wrapper}, original signature). With '
              r'context the model recovers the original signature from \texttt{func\_call}, which the reduced '
              r'wrapper rejects.',
              'tab:a6-unused-param', cols='llrrrrrr',
              head=(r'\textbf{Arch} & \textbf{Wrapper}', ' &')),
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='experiments.tex')
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--unused-param-json', default=None,
                    help='also write the per-arch unused_param wrapper scoring (reduced/full/either) here')
    args = ap.parse_args()
    tables, unused_scoring = compute(args.workers)
    with open(args.out, 'w') as f:
        f.write(to_tex(tables))
    print(f'wrote {args.out}')
    if args.unused_param_json:
        with open(args.unused_param_json, 'w') as f:
            json.dump(unused_scoring, f, indent=1)
        print(f'wrote {args.unused_param_json}')


if __name__ == '__main__':
    main()
