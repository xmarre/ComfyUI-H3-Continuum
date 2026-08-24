from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from ComfyUI_H3_Continuum_Join.constants import CONTINUUM_INTEROP_KEY
from ComfyUI_H3_Continuum_Join.model_patch import clone_model_for_chunk
from ComfyUI_H3_Continuum_Join.refdelta_interop import (
    REFERENCE_DIAGNOSTIC_CONTRACT,
    REFERENCE_DIAGNOSTIC_MODEL_OPTION,
    RefDeltaDiagnosticInteropError,
    reference_diagnostic_from_model,
)
from ComfyUI_H3_Continuum_Join.v2.sampling import _make_basic_guider


class FakeReferenceModel:
    def __init__(self, model_options=None):
        self.model_options = dict(model_options or {})

    def clone(self):
        return FakeReferenceModel(dict(self.model_options))


class FakeDiagnosticMixin:
    def initialize_reference(self, reference_model, positive, negative=None):
        self.reference_initialization = (reference_model, positive, negative)


class FakeSpec:
    contract = REFERENCE_DIAGNOSTIC_CONTRACT
    guider_mixin = FakeDiagnosticMixin

    def __init__(self, reference_model):
        self.reference_model = reference_model

    def with_reference_model(self, reference_model):
        return FakeSpec(reference_model)


class FakeGuiderModel:
    def __init__(self, spec=None):
        self.model_options = {}
        if spec is not None:
            self.model_options[REFERENCE_DIAGNOSTIC_MODEL_OPTION] = spec


def _install_fake_samplers(monkeypatch):
    class CFGGuider:
        def __init__(self, model):
            self.model_patcher = model
            self.original_conds = {}
            self.cfg = 1.0

        def inner_set_conds(self, conds):
            self.original_conds.update(conds)

    samplers = ModuleType("comfy.samplers")
    samplers.CFGGuider = CFGGuider
    comfy = sys.modules.get("comfy") or ModuleType("comfy")
    comfy.samplers = samplers
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers)
    return CFGGuider


def test_basic_guider_preserves_normal_path_without_diagnostic(monkeypatch):
    cfg_guider = _install_fake_samplers(monkeypatch)
    conditioning = [{"prompt": "exact"}]
    model = FakeGuiderModel()

    guider = _make_basic_guider(model, conditioning)

    assert isinstance(guider, cfg_guider)
    assert guider.original_conds == {"positive": conditioning}
    assert guider.cfg == 1.0
    assert not hasattr(guider, "reference_initialization")


def test_basic_guider_uses_exact_chunk_conditioning_for_reference(monkeypatch):
    _install_fake_samplers(monkeypatch)
    reference = FakeReferenceModel()
    conditioning = [{"prompt": "exact chunk conditioning"}]
    model = FakeGuiderModel(FakeSpec(reference))

    guider = _make_basic_guider(model, conditioning)

    assert guider.original_conds["positive"] is conditioning
    assert guider.reference_initialization == (reference, conditioning, None)
    assert guider.cfg == 1.0


def test_malformed_reference_contract_fails_closed():
    spec = FakeSpec(FakeReferenceModel())
    spec.contract = ("wrong", 99, "contract")
    with pytest.raises(RefDeltaDiagnosticInteropError, match="unsupported"):
        reference_diagnostic_from_model(FakeGuiderModel(spec))


class FakeModelPatcher:
    def __init__(self, model_options=None):
        inner_type = type(
            "MiniMaxH3Model",
            (),
            {
                "__module__": "comfy.ldm.minimax.model",
                "blocks": (),
                "final_layer": object(),
                "patch_size": (1, 2, 2),
                "latents_dim": 24,
                "audio_latents_dim": 32,
                "rope_freqs": lambda self, position_ids, device: position_ids,
            },
        )
        self.model = SimpleNamespace(diffusion_model=inner_type())
        self.model_options = dict(model_options or {})
        self.added = []
        self.removed = []

    def clone(self):
        clone = FakeModelPatcher(dict(self.model_options))
        clone.model = self.model
        return clone

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.added.append((wrapper_type, key, wrapper))

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.removed.append((wrapper_type, key))


def _install_patcher_extension(monkeypatch):
    extension = ModuleType("comfy.patcher_extension")
    extension.WrappersMP = SimpleNamespace(APPLY_MODEL="apply_model")
    comfy = sys.modules.get("comfy") or ModuleType("comfy")
    comfy.patcher_extension = extension
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", extension)


def test_chunk_clone_configures_fused_and_reference_models_identically(monkeypatch):
    _install_patcher_extension(monkeypatch)
    reference_source = FakeModelPatcher({"reference_only": True})
    source_spec = FakeSpec(reference_source)
    source = FakeModelPatcher(
        {
            "transformer_options": {"existing": "kept"},
            REFERENCE_DIAGNOSTIC_MODEL_OPTION: source_spec,
        }
    )

    chunk = clone_model_for_chunk(
        source,
        strict=False,
        debug=False,
        chunk_index=2,
        context_frames=39,
    )

    chunk_spec = reference_diagnostic_from_model(chunk)
    reference_chunk = chunk_spec.reference_model
    assert chunk is not source
    assert reference_chunk is not reference_source
    assert source.model_options[REFERENCE_DIAGNOSTIC_MODEL_OPTION] is source_spec
    assert REFERENCE_DIAGNOSTIC_MODEL_OPTION not in reference_source.model_options
    assert REFERENCE_DIAGNOSTIC_MODEL_OPTION not in reference_chunk.model_options

    fused_request = chunk.model_options["transformer_options"][CONTINUUM_INTEROP_KEY]
    reference_request = reference_chunk.model_options["transformer_options"][CONTINUUM_INTEROP_KEY]
    assert fused_request == reference_request == {
        "api": 1,
        "active": True,
        "min_actual_prefix_steps": 2,
        "chunk_index": 2,
        "context_frames": 39,
    }
    assert chunk.added
    assert reference_chunk.added


def test_initial_chunk_clones_both_models_without_context_hint(monkeypatch):
    _install_patcher_extension(monkeypatch)
    reference_source = FakeModelPatcher()
    source = FakeModelPatcher(
        {REFERENCE_DIAGNOSTIC_MODEL_OPTION: FakeSpec(reference_source)}
    )

    chunk = clone_model_for_chunk(
        source,
        strict=False,
        debug=False,
        chunk_index=1,
        context_frames=None,
    )
    reference_chunk = reference_diagnostic_from_model(chunk).reference_model

    assert CONTINUUM_INTEROP_KEY not in chunk.model_options["transformer_options"]
    assert CONTINUUM_INTEROP_KEY not in reference_chunk.model_options["transformer_options"]
    assert chunk.added
    assert reference_chunk.added
