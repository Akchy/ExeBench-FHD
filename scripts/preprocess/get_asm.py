"""Compile a single ExeBench function with gcc and extract its disassembly via objdump."""
from datetime import datetime
import os
import re
import subprocess
from pathlib import Path
from typing import Literal, Optional
from scripts.preprocess.helper_func import get_all_optz
from scripts.preprocess.preprocess_asm import processing_asm

# Wall-clock bound per gcc/objdump call, so one hung invocation can't stall a corpus run.
GCC_TIMEOUT = 60


def _log_tool_failure(parent_folder, command, reason):
    try:
        with open(f'{parent_folder}/gcc_errors.log', 'a+') as df:
            stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df.write(f'{stamp}: {command}\n    {str(reason).strip()[:2000]}\n')
    except OSError:
        pass  # logging must never break the run


def mod_path(source_file_path, extension, parent_path_flag=False):
    """Swap the extension of `source_file_path`; optionally also return its grandparent dir."""
    c_folder_path = Path(source_file_path).parent
    new_file_path = f'{c_folder_path}/{Path(source_file_path).stem}.{extension}'
    if parent_path_flag:
        return new_file_path, c_folder_path.parent
    return new_file_path


def compile_c_program(source_file, opt_flag, indv_opt_flag='', timeout=GCC_TIMEOUT) -> bool:
    """Build `source_file` at -`opt_flag`.

    `indv_opt_flag` keeps one individual pass enabled and disables the rest of
    that level's passes (see helper_func.get_all_optz); 'none' disables all of them.
    """
    output_file, parent_folder = mod_path(source_file, 'out', parent_path_flag=True)
    try:
        if not os.path.exists(source_file):
            return False

        if indv_opt_flag:
            gcc_command = ["gcc", f'-{opt_flag}']
            if indv_opt_flag != 'none':
                gcc_command.append(indv_opt_flag)
            gcc_command += get_all_optz(opt_flag, indv_opt_flag) or []
            gcc_command += [source_file, "-o", output_file, '-lm']
        else:
            gcc_command = ["gcc", f'-{opt_flag}', source_file, "-o", output_file, '-lm']

        with open(f'{parent_folder}/gcc_commands.log', 'a+') as df:
            df.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: {gcc_command}\n')

        result = subprocess.run(gcc_command, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return True
        _log_tool_failure(parent_folder, gcc_command, result.stderr)
        return False

    except subprocess.TimeoutExpired:
        _log_tool_failure(parent_folder, gcc_command, f'timeout after {timeout}s')
        return False
    except subprocess.SubprocessError as e:
        _log_tool_failure(parent_folder, gcc_command, e)
        return False
    except Exception as e:
        _log_tool_failure(parent_folder, source_file, f'unexpected: {e!r}')
        return False


def disassemble_out_file(source_file_path, save_asm: bool = False, timeout=GCC_TIMEOUT) -> str:
    """Return `objdump -d` of the built binary, or False on failure.

    `save_asm` also writes a .asm next to it (off by default: at corpus scale
    that is millions of extra files).
    """
    out_file, parent_folder = mod_path(source_file_path, 'out', parent_path_flag=True)
    asm_file = mod_path(source_file_path, 'asm')

    try:
        if not os.path.exists(out_file):
            return False

        command = ["objdump", "-d", out_file]
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            if save_asm:
                with open(asm_file, "w+") as asmfile:
                    asmfile.write(result.stdout)
            return result.stdout
        _log_tool_failure(parent_folder, command, result.stderr)
        return False

    except subprocess.TimeoutExpired:
        _log_tool_failure(parent_folder, command, f'timeout after {timeout}s')
        return False
    except subprocess.SubprocessError as e:
        _log_tool_failure(parent_folder, command, e)
        return False
    except Exception as e:
        _log_tool_failure(parent_folder, out_file, f'unexpected: {e!r}')
        return False


def rename_to_func0(c_func, fname):
    """Rename the target's definition and recursive self-calls to `func0`.

    extract_function looks for a `<func0>` label, so every target is compiled
    under that name, including `main`-named ones (which would otherwise
    collide with the appended driver). Plain text substitution of `fname(`,
    so a call-shaped occurrence inside a string or comment is renamed too.
    """
    if not fname:
        return c_func
    pattern = re.compile(r'\b' + re.escape(fname) + r'\b(?=\s*\()')
    return pattern.sub('func0', c_func)


def create_c_file(dict_db, parent_dir):
    """Write the renamed target plus an empty `main` driver to <parent_dir>/C_files/<task_id>_<type>.c."""
    opt_flag = dict_db['type']
    task_id = dict_db['task_id']
    c_func = dict_db['c_func']
    fname = dict_db.get('fname')

    c_folder_path = f'{parent_dir}/C_files'
    Path(c_folder_path).mkdir(parents=True, exist_ok=True)

    c_file_path = f'{c_folder_path}/{task_id}_{opt_flag}.c'

    with open(c_file_path, "w+") as outfile:
        src = rename_to_func0(c_func, fname).strip() + '\n'
        src += """
int main() {
    return 0;
}
"""
        outfile.write(src)

    return c_file_path


def _cleanup_artifacts(c_file_path):
    for ext in ('c', 'out', 'asm'):
        try:
            os.remove(mod_path(c_file_path, ext))
        except OSError:
            pass


def get_processed_asm(dict_db, parent_dir, asm_only: bool = False, mode: Optional[Literal["default", "full", 'no_process', 'no_process_w_addr']] = 'default', indv_opt_flag='', debug_print: bool = False, cleanup: bool = True, save_asm: bool = False):
    """Compile `dict_db['c_func']` at `dict_db['type']` and return its processed asm, or False."""
    opt_flag = dict_db['type']
    fname = dict_db.get('fname')

    c_file_path = create_c_file(dict_db, parent_dir)

    try:
        if not compile_c_program(c_file_path, opt_flag, indv_opt_flag=indv_opt_flag):
            return False

        asm = disassemble_out_file(c_file_path, save_asm=save_asm)
        if not asm:
            return False

        if mode == 'full':
            return asm

        norm_asm = processing_asm(asm, 'func0', mode, asm_only, debug_print)
        if not norm_asm:
            if debug_print:
                print("[-] Preprocessing failed.")
            return False

        if fname == 'main':
            # Give main-named records their real label back.
            norm_asm = re.sub(r'\bfunc0\b', 'main', norm_asm)

        return norm_asm
    finally:
        if cleanup:
            _cleanup_artifacts(c_file_path)
