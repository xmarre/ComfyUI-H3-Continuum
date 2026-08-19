from ComfyUI_H3_Continuum_Join.reference_precedence import (
    H3ContinuumSamplerProductionReferencePrecedence,
    H3ContinuumSamplerTimelineVideoReferencePrecedence,
    install_reference_frame_precedence,
    normalize_reference_frame_precedence,
)
from ComfyUI_H3_Continuum_Join.v3.driving_nodes import H3ContinuumSamplerV34
from ComfyUI_H3_Continuum_Join.v3.nodes import H3ContinuumSamplerProduction


def test_two_references_clear_first_and_last_frames():
    first = object()
    last = object()
    ref1 = object()
    ref2 = object()

    normalized = normalize_reference_frame_precedence(
        {
            "first_frame": first,
            "last_frame": last,
            "reference_image_1": ref1,
            "reference_image_2": ref2,
        }
    )

    assert normalized["first_frame"] is None
    assert normalized["last_frame"] is None
    assert normalized["reference_image_1"] is ref1
    assert normalized["reference_image_2"] is ref2


def test_no_reference_preserves_first_and_last_frames():
    first = object()
    last = object()

    normalized = normalize_reference_frame_precedence(
        {"first_frame": first, "last_frame": last}
    )

    assert normalized["first_frame"] is first
    assert normalized["last_frame"] is last


def test_saved_production_node_routes_two_refs_without_frame_leakage(monkeypatch):
    captured = {}

    def fake_run(self, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(H3ContinuumSamplerProduction, "run", fake_run)

    ref1 = object()
    ref2 = object()
    result = H3ContinuumSamplerProductionReferencePrecedence().run(
        first_frame=object(),
        last_frame=object(),
        reference_image_1=ref1,
        reference_image_2=ref2,
    )

    assert result == "ok"
    assert captured["first_frame"] is None
    assert captured["last_frame"] is None
    assert captured["reference_image_1"] is ref1
    assert captured["reference_image_2"] is ref2


def test_legacy_node_ids_are_replaced_without_renaming():
    mappings = {
        "H3ContinuumSamplerProduction": object(),
        "H3ContinuumSamplerTimelineVideo": object(),
        "unrelated": object(),
    }
    unrelated = mappings["unrelated"]

    install_reference_frame_precedence(mappings)

    assert (
        mappings["H3ContinuumSamplerProduction"]
        is H3ContinuumSamplerProductionReferencePrecedence
    )
    assert (
        mappings["H3ContinuumSamplerTimelineVideo"]
        is H3ContinuumSamplerTimelineVideoReferencePrecedence
    )
    assert mappings["unrelated"] is unrelated


def test_v34_still_declares_both_reference_slots():
    optional = H3ContinuumSamplerV34.INPUT_TYPES()["optional"]
    assert optional["reference_image_1"][0] == "IMAGE"
    assert optional["reference_image_2"][0] == "IMAGE"
