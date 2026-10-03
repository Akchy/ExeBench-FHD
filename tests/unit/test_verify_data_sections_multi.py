from scripts.verify import verify_data_sections_multi as vdsm


def test_parse_blocks_parses_progbits_block_with_hexdump():
    text = ".data @0x4000 (4 bytes):\n0000  2a 00 00 00                        |*...|"
    blocks = vdsm.parse_blocks(text)
    assert blocks == [(".data", 0x4000, 4, bytes.fromhex("2a000000"))]


def test_parse_blocks_parses_nobits_zero_initialized_block():
    text = ".bss @0x5000 (4 bytes, zero-initialized)"
    blocks = vdsm.parse_blocks(text)
    assert blocks == [(".bss", 0x5000, 4, None)]


def test_parse_blocks_multiple_blocks_separated_by_blank_line():
    text = (
        ".rodata @0x2000 (4 bytes):\n0000  01 02 03 04                        |....|"
        "\n\n"
        ".bss @0x5000 (4 bytes, zero-initialized)"
    )
    blocks = vdsm.parse_blocks(text)
    assert [b[0] for b in blocks] == [".rodata", ".bss"]


def test_parse_blocks_empty_text_returns_empty_list():
    assert vdsm.parse_blocks("") == []
