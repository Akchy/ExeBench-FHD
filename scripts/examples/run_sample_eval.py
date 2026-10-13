#!/usr/bin/env python3
"""Runnable example for scripts/exebench_fhd_eval.py using the two sample files here.

sample_exebench_dataset.json has one ordinary row and one with unused_param
set (example_add_ignore_y). Its prediction in sample_predictions.json drops
the unused `y`, as an O3 decompilation would, and is tagged "opt": "O3" so
opt_real_exe_wrapper is used for it.

Run: python3 -m scripts.examples.run_sample_eval
"""
import json
from pathlib import Path

from scripts.exebench_fhd_eval import example_evaluate_dataset_at_o0, exebench_fhd_validate, select_exe_wrapper

_HERE = Path(__file__).parent
_DATASET = _HERE / 'sample_exebench_dataset.json'
_PREDICTIONS = _HERE / 'sample_predictions.json'


def main():
    print('=== example_evaluate_dataset_at_o0 ===')
    print('Ground-truth func_def vs. its own real_io_pairs, always at O0 -')
    print("unused_param never matters here, real_exe_wrapper is always right.\n")
    example_evaluate_dataset_at_o0(str(_DATASET))

    print('\n=== exebench_fhd_validate (opt-aware) ===')
    print("example_add_ignore_y's prediction only declares `x` - matching a")
    print("real -O3 decompilation, where the unused `y` left no trace.")
    print("Its item['opt'] == 'O3' makes select_exe_wrapper() pick")
    print("opt_real_exe_wrapper instead of the fixed-arity real_exe_wrapper,")
    print('so the reduced-arity prediction compiles and runs at all.\n')

    # exebench_fhd_validate writes results back into the file, so use a copy.
    work_path = _HERE / '_sample_predictions_scratch.json'
    work_path.write_text(_PREDICTIONS.read_text())

    exebench_fhd_validate(
        model_name='sample_eval_example',
        file_path=str(work_path),
        exeb_dataset=str(_DATASET),
        parallel_workers=1,
        resume=False,
    )

    results = json.loads(work_path.read_text())
    for item in results:
        pred = item['predicted_c_funcs'][0]
        print(f"{item['task_id']} (opt={item.get('opt')}): "
              f"re-compile={pred['re-compile']} re-execute={pred['re-execute']}")
    work_path.unlink()

    print('\n=== select_exe_wrapper() directly, for comparison ===')
    dataset = json.loads(_DATASET.read_text())
    row = next(r for r in dataset if r['task_id'] == 'example_add_ignore_y')
    print("opt_level='O0'  ->", 'opt_real_exe_wrapper' if select_exe_wrapper(row, 'O0') == row.get('opt_real_exe_wrapper')
          else 'real_exe_wrapper (unchanged)')
    print("opt_level='O3'  ->", 'opt_real_exe_wrapper' if select_exe_wrapper(row, 'O3') == row.get('opt_real_exe_wrapper')
          else 'real_exe_wrapper (unchanged)')


if __name__ == '__main__':
    main()
