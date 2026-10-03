#!/usr/bin/env python3
"""Rewrite an ExeBench exe wrapper so it calls the target without its unused
parameters.

The original wrapper is generated from the full parameter list, so a correct
decompilation that drops an unused parameter would not even compile against
it. We remove the parameter from the call site, from the code that reads it
from input JSON, and from the `write_output` echo helper. Unknown shapes raise
WrapperAdaptError rather than guessing.
"""
import re

from scripts.extract.extract_unused_param import split_top_level

# Input-reading shapes. The TYPE prefix can contain spaces (`unsigned char*`),
# hence `.*?` rather than a word class.
#     TYPE name = input_json["name"];
_SCALAR_READ_RE = r'[ \t]*.*?\b{name}\b\s*=\s*input_json\[\s*"{name}"\s*\];\s*\n'
_VECTOR_READ_RE = (
    r'[ \t]*std::vector<[^>]+>\s+(?P<vec>input_temp_\d+_vec);\s*\n'
    r'[ \t]*for\s*\(\s*auto&\s*elem\s*:\s*input_json\[\s*"{name}"\s*\]\s*\)\s*\{{\s*\n'
    # The loop body may contain braces (struct initializers); each repetition
    # consumes one whole line, so the match stops at the first bare `}`.
    r'(?:.*\n)*?'
    r'[ \t]*\}}\s*\n'
    r'[ \t]*.*?\b{name}\b\s*=\s*&\(?(?P=vec)\)?\[0\];\s*\n'
)
# char* name = strdup(input_json["name"].get<std::string>().c_str());
_STRING_READ_RE = (
    r'[ \t]*.*?\b{name}\b\s*=\s*strdup\(\s*input_json\[\s*"{name}"\s*\]'
    r'\.get<\s*std::string\s*>\(\)\.c_str\(\)\s*\);\s*\n'
)
# TYPE name = input_json["name"].get<TYPE>();
_GET_CAST_READ_RE = (
    r'[ \t]*.*?\b{name}\b\s*=\s*input_json\[\s*"{name}"\s*\]\.get<[^>]+>\(\);\s*\n'
)
# Single struct pointer:
#     int dev_pointer__dummy = input_json["dev"]["dummy"];
#     struct block_device dev_pointer = {dev_pointer__dummy};
#     struct block_device* dev = &dev_pointer;
_SINGLE_STRUCT_READ_RE = (
    r'(?:[ \t]*[\w:\*<>]+\s+{name}_pointer__\w+\s*=\s*input_json\[\s*"{name}"\s*\]\[\s*"\w+"\s*\];\s*\n)+'
    r'[ \t]*.*?\b{name}_pointer\b\s*=\s*\{{[^{{}}]*\}};\s*\n'
    r'[ \t]*.*?\b{name}\b\s*=\s*&{name}_pointer;\s*\n'
)
# Struct by value:
#     int x__dummy = input_json["x"]["dummy"];
#     struct S x = {x__dummy};
_STRUCT_BY_VALUE_READ_RE = (
    r'(?:[ \t]*[\w:\*<>]+\s+{name}__\w+\s*=\s*input_json\[\s*"{name}"\s*\]\[\s*"\w+"\s*\];\s*\n)+'
    r'[ \t]*.*?\b{name}\b\s*=\s*\{{[^{{}}]*\}};\s*\n'
)

# Echo helper in generated wrappers. It receives every parameter, so the
# removed ones must go from its signature, calls and body too. Optional.
_ECHO_HELPER_NAME = "write_output"

# Output-echo shapes inside write_output's body.
_VECTOR_OUTPUT_ECHO_RE = (
    r'[ \t]*std::vector<json>\s+(?P<tmp>output_temp_\d+);\s*\n'
    r'[ \t]*for\s*\([^)]*\)\s*\{{\s*\n'
    r'(?:.*\n)*?'
    r'[ \t]*\}}\s*\n'
    r'[ \t]*output_json\[\s*"{name}"\s*\]\s*=\s*(?P=tmp);\s*\n'
)
_SCALAR_OUTPUT_ECHO_RE = r'[ \t]*output_json\[\s*"{name}"\s*\]\s*=\s*{name};\s*\n'
# Single struct pointer:
#     struct block_device output_temp_4 = *dev;
#     json output_temp_5;
#     output_temp_5["dummy"] = output_temp_4.dummy;
#     output_json["dev"] = output_temp_5;
_SINGLE_STRUCT_OUTPUT_ECHO_RE = (
    r'[ \t]*[\w:<>]+\s+(?P<tmp1>output_temp_\d+)\s*=\s*\*{name};\s*\n'
    r'[ \t]*json\s+(?P<tmp2>output_temp_\d+);\s*\n'
    r'(?:[ \t]*\n)?'
    r'(?:[ \t]*(?P=tmp2)\[\s*"\w+"\s*\]\s*=\s*(?P=tmp1)\.\w+;\s*\n)+'
    r'[ \t]*output_json\[\s*"{name}"\s*\]\s*=\s*(?P=tmp2);\s*\n'
)


class WrapperAdaptError(Exception):
    """Unhandled wrapper shape. `.name` is the parameter, `.stage` one of
    "json_read", "call_site", "output_echo"."""
    def __init__(self, message, name=None, stage=None):
        super().__init__(message)
        self.name = name
        self.stage = stage


def _drop_call_args(wrapper_src, target, unused_names):
    """Drop `unused_names` from every call `target(...);`. Arguments are bare
    identifiers, so exact matching is enough."""
    call_re = re.compile(r'\b' + re.escape(target) + r'\s*\(([^;{}]*)\)\s*;')

    def _sub(m):
        args = [a.strip() for a in split_top_level(m.group(1))]
        kept = [a for a in args if a not in unused_names]
        return f"{target}({', '.join(kept)});"

    return call_re.sub(_sub, wrapper_src)


def _drop_def_param(wrapper_src, target, unused_names):
    """Drop `unused_names` from the parameter list of `target`'s definition, if any."""
    def_re = re.compile(r'\b([\w:\*<>]+)\s+' + re.escape(target) + r'\s*\(([^)]*)\)\s*\{')
    m = def_re.search(wrapper_src)
    if not m:
        return wrapper_src
    params = [p.strip() for p in split_top_level(m.group(2))]

    def _param_name(p):
        # (?<!\w) skips literal suffixes such as the L in `A[4L*16L]`.
        idents = re.findall(r'(?<!\w)[A-Za-z_]\w*', p)
        return idents[-1] if idents else None

    kept = [p for p in params if _param_name(p) not in unused_names]
    new_def = f"{m.group(1)} {target}({', '.join(kept)}) {{"
    return wrapper_src[:m.start()] + new_def + wrapper_src[m.end():]


# Tried in order; the names are what shape_log records.
_JSON_READ_SHAPES = (
    (_VECTOR_READ_RE, "vector"),
    (_SINGLE_STRUCT_READ_RE, "single_struct_pointer"),
    (_STRUCT_BY_VALUE_READ_RE, "struct_by_value"),
    (_STRING_READ_RE, "string_strdup"),
    (_GET_CAST_READ_RE, "get_cast_scalar"),
    (_SCALAR_READ_RE, "plain_scalar"),
)


def _drop_json_read(wrapper_src, name, shape_log=None):
    """Remove the code that reads `name` from input JSON, or raise."""
    for pattern, shape_name in _JSON_READ_SHAPES:
        rx = re.compile(pattern.format(name=re.escape(name)))
        new_src, n = rx.subn('', wrapper_src)
        if n:
            if shape_log is not None:
                shape_log.append(shape_name)
            return new_src
    raise WrapperAdaptError(
        f"no known JSON-read shape (vector/string/scalar) found for '{name}'", name=name, stage="json_read",
    )


def _drop_output_echo(wrapper_src, name):
    """Remove `name`'s output echo if present. Input-only parameters have none;
    a missed echo is caught by _check_no_dangling_refs."""
    for pattern in (_VECTOR_OUTPUT_ECHO_RE, _SINGLE_STRUCT_OUTPUT_ECHO_RE, _SCALAR_OUTPUT_ECHO_RE):
        rx = re.compile(pattern.format(name=re.escape(name)))
        new_src, n = rx.subn('', wrapper_src)
        if n:
            return new_src
    return wrapper_src


def _function_body_span(wrapper_src, target):
    """Brace-matched body of `target`'s definition, or None."""
    def_re = re.compile(r'\b[\w:\*<>]+\s+' + re.escape(target) + r'\s*\([^)]*\)\s*\{')
    m = def_re.search(wrapper_src)
    if not m:
        return None
    depth = 1
    i = m.end()
    start = i
    while i < len(wrapper_src) and depth > 0:
        if wrapper_src[i] == '{':
            depth += 1
        elif wrapper_src[i] == '}':
            depth -= 1
        i += 1
    return wrapper_src[start:i - 1]


def _check_no_dangling_refs(wrapper_src, target, unused_names):
    """Raise if any removed name is still referenced in `target`'s body, which
    would fail to compile. Catches echo shapes we don't recognize."""
    body = _function_body_span(wrapper_src, target)
    if body is None:
        return
    for name in unused_names:
        for m in re.finditer(r'\b' + re.escape(name) + r'\b', body):
            before = body[:m.start()]
            # Member accesses and JSON keys with the same name are not
            # references to the parameter (e.g. a parameter named `dummy`).
            if before.endswith('.') or before.endswith('->') or before.endswith('"'):
                continue
            raise WrapperAdaptError(
                f"'{name}' still referenced inside {target}()'s body after removal - unhandled echo shape",
                name=name, stage="output_echo",
            )


def adapt_exe_wrapper(wrapper_src, fname, unused_names, shape_log=None):
    """Return `wrapper_src` with `unused_names` removed from the call to
    `fname`, the input-reading code, and the write_output helper.

    If `shape_log` is a list, one shape name from _JSON_READ_SHAPES is
    appended per removed parameter, plus "libc_collision_rename" when the
    call site was found as `<fname>__bench`.
    """
    if not unused_names:
        return wrapper_src
    out = wrapper_src
    for name in unused_names:
        out = _drop_json_read(out, name, shape_log=shape_log)

    # ExeBench renames functions that clash with libc/libm (pow, puts, ...)
    # to <fname>__bench in the wrapper.
    call_target = fname
    call_re = re.compile(r'\b' + re.escape(call_target) + r'\s*\(([^;{}]*)\)\s*;')
    if not call_re.search(out):
        call_target = f"{fname}__bench"
        call_re = re.compile(r'\b' + re.escape(call_target) + r'\s*\(([^;{}]*)\)\s*;')
        if call_re.search(out) and shape_log is not None:
            shape_log.append("libc_collision_rename")
    if not call_re.search(out):
        raise WrapperAdaptError(f"no call site found for {fname}(...) or {fname}__bench(...)", stage="call_site")

    out = _drop_call_args(out, call_target, unused_names)
    out = _drop_def_param(out, _ECHO_HELPER_NAME, unused_names)
    out = _drop_call_args(out, _ECHO_HELPER_NAME, unused_names)
    for name in unused_names:
        out = _drop_output_echo(out, name)
    _check_no_dangling_refs(out, _ECHO_HELPER_NAME, unused_names)
    return out
