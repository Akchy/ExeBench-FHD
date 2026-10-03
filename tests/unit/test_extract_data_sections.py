from scripts.extract import extract_data_sections as eds


def test_data_section_name_re_matches_data_like_sections():
    for name in [".data", ".rodata", ".bss", ".tdata", ".tbss",
                 ".data.rel.local", ".data.rel.ro", ".rodata.str1.1", ".bss.foo"]:
        assert eds._DATA_SECTION_NAME_RE.match(name), name


def test_data_section_name_re_rejects_non_data_sections():
    for name in [".text", ".text.startup", ".got", ".got.plt",
                 ".init_array", ".fini_array", ".comment", ".debug_info", ".eh_frame"]:
        assert not eds._DATA_SECTION_NAME_RE.match(name), name


_MAP_TEXT = """\
Linker script and memory map

.text           0x0000000000001000      0x100
 .text          0x0000000000001000       0x50 /tmp/xyz/f.o
 .text          0x0000000000001050       0xb0 /usr/lib/crt1.o

.rodata         0x0000000000002000        0x8
 .rodata        0x0000000000002000        0x8 /tmp/xyz/f.o

.data           0x0000000000004000       0x20
 .data          0x0000000000004000        0x4 /tmp/xyz/f.o
 .data.rel.local
                0x0000000000004004       0x1c /tmp/xyz/f.o
 .data          0x0000000000004020        0x8 /usr/lib/crt1.o

.bss            0x0000000000005000       0x10
 .bss           0x0000000000005000       0x10 /tmp/xyz/f.o

.zerosize       0x0000000000006000        0x0
 .zerosize      0x0000000000006000        0x0 /tmp/xyz/f.o
"""


def test_parse_map_contributions_data_like_only_and_own_object():
    contributions = list(eds.parse_map_contributions(_MAP_TEXT, "/tmp/xyz/f.o"))
    assert contributions == [
        (".rodata", 0x2000, 0x8),
        (".data", 0x4000, 0x4),
        (".data", 0x4004, 0x1c),
        (".bss", 0x5000, 0x10),
    ]


def test_parse_map_contributions_skips_other_objects_and_text():
    contributions = list(eds.parse_map_contributions(_MAP_TEXT, "/usr/lib/crt1.o"))
    assert all(sec != ".text" for sec, _, _ in contributions)
    assert (".data", 0x4020, 0x8) in contributions


def test_parse_map_contributions_no_matches_for_unrelated_object():
    assert list(eds.parse_map_contributions(_MAP_TEXT, "/no/such/f.o")) == []


_OBJDUMP_TEXT = """\
f.out:     file format elf64-x86-64

Contents of section .rodata:
 2000 68656c6c 6f000000 00000000 00000000  hello...........
 2010 0a000000                             ....

Contents of section .data:
 4000 2a000000                             *...

Contents of section .gap:
 3000 01020304                             ....
 3020 05060708                             ....
"""


def test_parse_objdump_sections_reads_base_and_bytes():
    sections = eds.parse_objdump_sections(_OBJDUMP_TEXT)
    expected_rodata = (
        bytes.fromhex("68656c6c") + bytes.fromhex("6f000000")
        + bytes.fromhex("00000000") + bytes.fromhex("00000000")
        + bytes.fromhex("0a000000")
    )
    assert sections[".rodata"] == (0x2000, expected_rodata)
    assert sections[".data"] == (0x4000, bytes.fromhex("2a000000"))


def test_parse_objdump_sections_zero_fills_address_gaps():
    _, data = eds.parse_objdump_sections(_OBJDUMP_TEXT)[".gap"]
    assert data == bytes.fromhex("01020304") + b"\x00" * 0x1c + bytes.fromhex("05060708")


def test_parse_objdump_sections_nobits_section_absent():
    sections = eds.parse_objdump_sections(_OBJDUMP_TEXT)
    assert ".bss" not in sections


def test_hexdump_format():
    out = eds.hexdump(b"Hello")
    assert out.startswith("0000  48 65 6c 6c 6f")
    assert out.endswith("|Hello|")


def test_format_contribution_zero_initialized_when_data_none():
    text = eds.format_contribution(".bss", 0x5000, 16, None)
    assert text == ".bss @0x5000 (16 bytes, zero-initialized)"


def test_format_contribution_includes_hexdump_when_data_present():
    text = eds.format_contribution(".data", 0x4000, 4, b"\x2a\x00\x00\x00")
    assert text.startswith(".data @0x4000 (4 bytes):\n")
    assert "2a 00 00 00" in text


def test_extract_data_from_binary_no_contributions_is_empty_not_error():
    rendered, err = eds.extract_data_from_binary("", "/tmp/xyz/f.o", "/nonexistent/f.out")
    assert rendered == ""
    assert err is None


def test_build_source_strips_static_and_prepends_deps(record_factory):
    rec = record_factory(
        real_deps="typedef int myint;\n# 1",
        func_def="static int add_one(int x) {\n    return x + 1;\n}\n",
    )
    src = eds.build_source(rec)
    assert "typedef int myint;" in src
    assert "static int" not in src
    assert "int add_one(int x)" in src


def test_build_driver_c_file_renames_and_appends_driver(tmp_path, record_factory):
    rec = record_factory(fname="add_one", func_def="int add_one(int x) {\n    return x + 1;\n}\n")
    out = tmp_path / "f.c"
    eds.build_driver_c_file(rec, str(out))
    text = out.read_text()
    assert "func0" in text
    assert "add_one" not in text.replace("int main(void) { return 0; }", "")
    assert text.rstrip().endswith("int main(void) { return 0; }")


def test_build_driver_c_file_leaves_main_named_target_untouched(tmp_path, record_factory):
    rec = record_factory(fname="main", func_def="int main(void) {\n    return 0;\n}\n")
    out = tmp_path / "f.c"
    eds.build_driver_c_file(rec, str(out))
    text = out.read_text()
    assert text.count("int main") == 1


_MERGE_MAP_TEXT = """\
.rodata         0x0000000000002000        0xc
 .rodata        0x0000000000002000        0x4 /usr/lib/crt1.o
 .rodata.cst4   0x0000000000002004        0x8 /usr/lib/crt1.o
 .rodata.cst4   0x000000000000200c        0x4 /tmp/xyz/f.o
 .rodata.str1.1
                0x0000000000002010        0x6 /tmp/xyz/f.o
"""


def test_parse_map_members_keeps_input_section_names_including_wrapped():
    members = eds.parse_map_members(_MERGE_MAP_TEXT)
    assert [(m["in_section"], m["addr"], m["obj"]) for m in members] == [
        (".rodata", 0x2000, "/usr/lib/crt1.o"),
        (".rodata.cst4", 0x2004, "/usr/lib/crt1.o"),
        (".rodata.cst4", 0x200c, "/tmp/xyz/f.o"),
        (".rodata.str1.1", 0x2010, "/tmp/xyz/f.o"),
    ]


def test_merge_section_re():
    for name in (".rodata.cst4", ".rodata.cst8", ".rodata.cst16", ".rodata.str1.1", ".rodata.str4.4", ".rodata.str1.8"):
        assert eds._MERGE_SECTION_RE.match(name), name
    for name in (".rodata", ".rodata.foo", ".data", ".rodata.cst"):
        assert not eds._MERGE_SECTION_RE.match(name), name


def test_merge_pieces_cst_and_str():
    assert eds._merge_pieces(".rodata.cst4", b"\x00\x00\x00\x3f\x00\x00\x80\x3f") == (
        [b"\x00\x00\x00\x3f", b"\x00\x00\x80\x3f"], 4)
    assert eds._merge_pieces(".rodata.str1.1", b"hi\x00bye\x00") == ([b"hi\x00", b"bye\x00"], 1)
    assert eds._merge_pieces(".rodata.str2.2", b"a\x00\x00\x00") == ([b"a\x00\x00\x00"], 2)


def test_locate_merged_respects_alignment_and_search_start():
    # base 0x2000: aligned copy at 0x2000 (before search_from), unaligned
    # copy at 0x2005, aligned copy at 0x200c
    out = b"\x00\x00\x00\x3f" + b"\x11\x00\x00\x00" + b"\x3f\x22\x22\x22" + b"\x00\x00\x00\x3f"
    assert eds._locate_merged([b"\x00\x00\x00\x3f"], 4, 0x2000, out, 0x2004) == [(0x200c, b"\x00\x00\x00\x3f")]
    # nothing at/after search_from -> falls back to the whole section
    assert eds._locate_merged([b"\x00\x00\x00\x3f"], 4, 0x2000, out[:4], 0x2004) == [(0x2000, b"\x00\x00\x00\x3f")]
    assert eds._locate_merged([b"zz"], 1, 0x2000, out, 0x2000) == [(None, b"zz")]


def test_render_merged_coalesces_contiguous_and_reports_missing():
    blocks = eds._render_merged(".rodata", ".rodata.cst4",
                                [(0x2008, b"\x01\x00\x00\x00"), (0x200c, b"\x02\x00\x00\x00"), (None, b"\x03\x00\x00\x00")])
    assert blocks[0].startswith(".rodata @0x2008 (8 bytes):")
    assert blocks[1].startswith(".rodata (.rodata.cst4, 4 bytes, not located in linked output):")


def test_data_section_by_arch_reads_both_shapes():
    per_level = {"x86": {"O0": "a", "O2": "b"}, "ARM64": {"O0": "", "O2": "c"}}
    assert eds.data_section_by_arch(per_level, "O2") == {"x86": "b", "ARM64": "c"}
    assert eds.data_section_by_arch(per_level) == {"x86": "a", "ARM64": ""}
    legacy = {"x86": "old-O0-dump", "ARM64": None}  # single -O0 build per arch
    assert eds.data_section_by_arch(legacy, "O2") == legacy
    assert eds.data_section_by_arch(None) is None
