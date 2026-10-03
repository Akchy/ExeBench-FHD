#!/usr/bin/env python3
"""Check that extract_data_sections captures every data-like section, on every arch.

Compiles one synthetic function that touches .data, .bss, .rodata, .tdata and
.tbss, then checks each section is reported and spot-checks the bytes
(global_init == 42, tls_init == 7, and msg's pointer equals its .rodata
address, which only holds because addresses come from the linked binary).
Exits non-zero on any failure.

    python3 -m scripts.verify.verify_data_sections_multi
"""
import re
import struct
import sys

from scripts.extract.extract_data_sections import ARCH_TOOLCHAIN, process_record

FUNC_DEF = """
int global_init = 42;                    /* -> .data */
int global_uninit;                       /* -> .bss */
const char *msg = "multi-section test";  /* bytes -> .rodata, pointer -> .data */
__thread int tls_init = 7;               /* -> .tdata */
__thread int tls_uninit;                 /* -> .tbss */

int calc(int x) {
    global_uninit = x + global_init;
    return global_uninit + (int)msg[0] + tls_init + tls_uninit;
}
""".strip()

REC = {
    "fname": "calc",
    "func_def": FUNC_DEF,
    "real_deps": None,
    "angha_deps": None,
}

WANT_SECTIONS = (".rodata", ".data", ".bss", ".tdata", ".tbss")

# Block format written by extract_data_sections.format_contribution:
#   ".data @0x4010 (4 bytes):"                          (PROGBITS header)
#   ".bss @0x4014 (4 bytes, zero-initialized)"           (NOBITS, no rows follow)
#   "0000  2a 00 00 00                        |*...|"    (hexdump row)
_HEADER_RE = re.compile(r"^(\.\S+) @0x([0-9a-fA-F]+) \((\d+) bytes(, zero-initialized)?\)")
_ROW_RE = re.compile(r"^[0-9a-fA-F]{4}  ([0-9a-fA-F ]+?)  \|")


def parse_blocks(text):
    """Parse one arch's data_section text into [(section, addr, size, bytes or None for NOBITS)]."""
    blocks = []
    for chunk in text.split("\n\n"):
        lines = chunk.splitlines()
        if not lines:
            continue
        m = _HEADER_RE.match(lines[0])
        if not m:
            continue
        section, addr_s, size_s, is_zero = m.groups()
        if is_zero:
            blocks.append((section, int(addr_s, 16), int(size_s), None))
            continue
        raw = bytearray()
        for row in lines[1:]:
            rm = _ROW_RE.match(row)
            if rm:
                raw += bytes.fromhex(rm.group(1).replace(" ", ""))
        blocks.append((section, int(addr_s, 16), int(size_s), bytes(raw)))
    return blocks


def main():
    archs = list(ARCH_TOOLCHAIN)
    data_section, errors = process_record(REC, archs=archs)

    all_ok = True
    for arch in archs:
        print(f"\n=== {arch} ===")
        err = errors[arch]
        if err:
            print(f"  FAIL: build error: {err[:300]}")
            all_ok = False
            continue

        text = data_section[arch]
        blocks = parse_blocks(text)
        present = {b[0] for b in blocks}
        missing = [s for s in WANT_SECTIONS if s not in present]
        if missing:
            print(f"  FAIL: missing section(s): {missing}")
            all_ok = False
        else:
            print(f"  OK: all {len(WANT_SECTIONS)} section kinds present: {sorted(present)}")

        # Presence alone wouldn't catch wrong address slicing.
        data_blocks = [b for b in blocks if b[0] == ".data"]
        rodata_blocks = [b for b in blocks if b[0] == ".rodata"]
        tdata_blocks = [b for b in blocks if b[0] == ".tdata"]
        rodata_addr = rodata_blocks[0][1] if rodata_blocks else None

        checks = {
            "global_init == 42": any(
                data[3] == struct.pack("<i", 42) for data in data_blocks
            ),
            "tls_init == 7": any(
                data[3] == struct.pack("<i", 7) for data in tdata_blocks
            ),
            "msg pointer resolves to .rodata address": rodata_addr is not None and any(
                data[3] is not None and len(data[3]) in (4, 8)
                and int.from_bytes(data[3], "little") == rodata_addr
                for data in data_blocks
            ),
        }

        for label, ok in checks.items():
            mark = "OK" if ok else "FAIL"
            print(f"  {mark}: {label}")
            all_ok = all_ok and ok

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
