from scripts.survey import survey_func_call_param_types as sfc


def test_categorize_void_placeholder_and_varargs():
    assert sfc.categorize("void") == "void_placeholder"
    assert sfc.categorize("") == "void_placeholder"
    assert sfc.categorize("...") == "varargs"


def test_categorize_int_family_and_typedef_aliases():
    assert sfc.categorize("int") == "int_family"
    assert sfc.categorize("unsigned long") == "int_family"
    assert sfc.categorize("const int") == "int_family"
    assert sfc.categorize("size_t") == "int_family"
    assert sfc.categorize("uint32_t") == "int_family"


def test_categorize_float_family():
    assert sfc.categorize("float") == "float_family"
    assert sfc.categorize("double") == "float_family"
    assert sfc.categorize("long double") == "float_family"


def test_categorize_char_or_void_pointer():
    assert sfc.categorize("char *") == "char_or_void_pointer"
    assert sfc.categorize("void*") == "char_or_void_pointer"


def test_categorize_other_typed_pointer():
    assert sfc.categorize("int *") == "other_typed_pointer"


def test_categorize_struct_union_by_value_and_pointer():
    assert sfc.categorize("struct foo") == "struct_or_union_by_value"
    assert sfc.categorize("struct foo *") == "struct_or_union_pointer"
    assert sfc.categorize("union bar") == "struct_or_union_by_value"


def test_categorize_function_pointer():
    assert sfc.categorize("int (*)(int, int)") == "function_pointer"


def test_categorize_simd_or_complex():
    assert sfc.categorize("__m128i") == "simd_or_complex"
    assert sfc.categorize("_Complex double") == "simd_or_complex"


def test_categorize_uncategorized_fallback():
    assert sfc.categorize("FooBarUnknownType_t") == "uncategorized"


def test_categorize_extended_int_typedef_aliases():
    for alias in ("u32", "u8", "uint32", "BOOL", "GLenum", "GLint",
                  "DWORD", "boolean", "Oid", "ULONG"):
        assert sfc.categorize(alias) == "int_family"


def test_categorize_extended_float_typedef_aliases():
    for alias in ("GLfloat", "GLdouble", "f32", "f64", "real", "float_t"):
        assert sfc.categorize(alias) == "float_family"


def test_categorize_leaves_pointer_like_typedefs_uncategorized():
    # Frequent, but conventionally handles/pointers, so not int_family.
    for alias in ("HANDLE", "yyscan_t", "png_structp", "rtx", "tree",
                  "device_t", "clObject", "JAVA_OBJECT", "vec3_t", "int32x4_t"):
        assert sfc.categorize(alias) == "uncategorized"


def test_survey_counts_categories_and_return_types(monkeypatch):
    records = [
        {"signature": ["int", "int", "char *"]},
        {"signature": ["void", "float"]},
        {"signature": None},
    ]

    def fake_iter_split(split, limit=None):
        data = records if limit is None else records[:limit]
        yield from data

    monkeypatch.setattr(sfc, "iter_split", fake_iter_split)

    report = sfc.survey("test_real")
    assert report["records_scanned"] == 3
    assert report["records_with_signature"] == 2
    assert report["param_type_categories"]["int_family"] == 1
    assert report["param_type_categories"]["char_or_void_pointer"] == 1
    assert report["param_type_categories"]["float_family"] == 1
    assert report["top_return_types"] == {"int": 1, "void": 1}
