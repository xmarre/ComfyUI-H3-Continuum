import pytest

from ComfyUI_H3_Continuum_Join.conditioning import (
    conditioning_mode_from_presence,
)


@pytest.mark.parametrize(
    ("has_first", "has_last", "has_reference", "expected"),
    [
        (False, False, False, "t2va"),
        (True, False, False, "i2va"),
        (False, True, False, "last_only"),
        (True, True, False, "fl2va"),
        (False, False, True, "reference"),
        (True, False, True, "reference"),
        (False, True, True, "reference"),
        (True, True, True, "reference"),
    ],
)
def test_conditioning_mode_matrix(
    has_first, has_last, has_reference, expected
):
    assert conditioning_mode_from_presence(
        has_first=has_first,
        has_last=has_last,
        has_reference=has_reference,
    ) == expected
