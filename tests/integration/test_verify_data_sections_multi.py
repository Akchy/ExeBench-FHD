import pytest

from scripts.verify import verify_data_sections_multi as vdsm
from tests.conftest import requires_gcc, requires_arm32_gcc, requires_arm64_gcc, requires_x86_32

pytestmark = pytest.mark.slow


@requires_gcc
@requires_arm32_gcc
@requires_arm64_gcc
@requires_x86_32
def test_verification_passes_across_all_four_architectures(capsys):
    with pytest.raises(SystemExit) as exc_info:
        vdsm.main()
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "ALL CHECKS PASSED" in out
    assert "FAIL" not in out


@requires_gcc
def test_process_record_x86_64_reports_all_five_section_kinds():
    data_section, errors = vdsm.process_record(vdsm.REC, archs=["x86-64"])
    assert errors["x86-64"] is None
    blocks = vdsm.parse_blocks(data_section["x86-64"])
    present = {b[0] for b in blocks}
    assert set(vdsm.WANT_SECTIONS) <= present
