from scripts.extract import extract_dummy_struct_types as edst


def test_dummy_struct_detects_placeholder_in_angha_deps(record_factory):
    rec = record_factory(angha_deps="struct sockaddr { int dummy ; };", real_deps="")
    assert edst.dummy_struct(rec) is True


def test_dummy_struct_detects_placeholder_in_real_deps(record_factory):
    rec = record_factory(real_deps="struct sockaddr {int dummy;};", angha_deps="")
    assert edst.dummy_struct(rec) is True


def test_dummy_struct_tolerates_arbitrary_whitespace():
    assert edst._DUMMY_STRUCT_RE.search("{  int   dummy   ;  }")


def test_dummy_struct_false_for_real_struct_definition(record_factory):
    rec = record_factory(angha_deps="struct sockaddr { short sa_family; char sa_data[14]; };", real_deps="")
    assert edst.dummy_struct(rec) is False


def test_dummy_struct_false_when_dummy_is_one_of_several_fields(record_factory):
    rec = record_factory(angha_deps="struct foo { int dummy; int real_field; };", real_deps="")
    assert edst.dummy_struct(rec) is False


def test_dummy_struct_false_when_no_deps_at_all(record_factory):
    rec = record_factory(real_deps="", angha_deps="")
    assert edst.dummy_struct(rec) is False


def test_dummy_struct_false_when_deps_fields_are_none(record_factory):
    rec = record_factory(real_deps=None, angha_deps=None)
    assert edst.dummy_struct(rec) is False


def test_process_record_adds_dummy_udt_field_true(record_factory):
    rec = record_factory(angha_deps="struct sockaddr { int dummy; };", real_deps="")
    out = edst.process_record(rec)
    assert out["dummy_udt"] is True
    assert out is not rec


def test_process_record_adds_dummy_udt_field_false_by_default(record_factory):
    rec = record_factory(angha_deps="struct point { int x; int y; };", real_deps="")
    out = edst.process_record(rec)
    assert out["dummy_udt"] is False
