from __future__ import annotations

import sys
import types

import pytest
import torch

from ComfyUI_H3_Continuum_Join.v2.sampling import (
    _record_chunk_conditioning,
    _record_chunk_refine_state,
    capture_chunk_conditioning,
    capture_chunk_refine_state,
    sample_chunk,
)
from ComfyUI_H3_Continuum_Join.v3 import driving_nodes
from ComfyUI_H3_Continuum_Join.v3.driving_nodes import (
    H3ContinuumSamplerV34,
    _refine_state_output,
)
from ComfyUI_H3_Continuum_Join.v3.nodes import H3ContinuumSamplerProduction


class _Nested:
    is_nested = True

    def __init__(self, members):
        self._members = list(members)

    def unbind(self):
        return tuple(self._members)

    def to(self, _device):
        return self


class _FakeCFGGuider:
    def __init__(self, model):
        self.model = model
        self.conds = None

    def inner_set_conds(self, conds):
        self.conds = conds

    def sample(
        self,
        _noise,
        latent_image,
        _sampler,
        _sigmas,
        *,
        denoise_mask,
        callback,
        disable_pbar,
        seed,
    ):
        del denoise_mask, callback, disable_pbar, seed
        return latent_image


def _install_fake_comfy_runtime(monkeypatch):
    comfy = types.ModuleType("comfy")
    samplers = types.ModuleType("comfy.samplers")
    samplers.CFGGuider = _FakeCFGGuider
    sample = types.ModuleType("comfy.sample")
    sample.fix_empty_latent_channels = lambda _model, latent, *_ratios: latent
    sample.prepare_noise = lambda samples, _seed, _batch_inds: samples
    model_management = types.ModuleType("comfy.model_management")
    model_management.intermediate_device = lambda: torch.device("cpu")
    utils = types.ModuleType("comfy.utils")
    utils.PROGRESS_BAR_ENABLED = False
    latent_preview = types.ModuleType("latent_preview")
    latent_preview.prepare_callback = lambda *_args, **_kwargs: None

    comfy.samplers = samplers
    comfy.sample = sample
    comfy.model_management = model_management
    comfy.utils = utils
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers)
    monkeypatch.setitem(sys.modules, "comfy.sample", sample)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)
    monkeypatch.setitem(sys.modules, "latent_preview", latent_preview)


def test_capture_scopes_are_context_local_and_preserve_exact_objects():
    first = [[torch.tensor([1.0]), {"chunk": 1}]]
    second = [[torch.tensor([2.0]), {"chunk": 2}]]
    model = object()

    _record_chunk_conditioning(first)
    with capture_chunk_conditioning() as outer:
        _record_chunk_conditioning(first)
        _record_chunk_conditioning(second)
    assert outer == [first, second]
    assert outer[0] is first
    assert outer[1] is second

    with capture_chunk_refine_state() as states:
        _record_chunk_refine_state(model=model, conditioning=first, noise_mask="mask")
    assert len(states) == 1
    assert states[0]["model"] is model
    assert states[0]["positive"] is first
    assert states[0]["noise_mask"] == "mask"


def test_sample_chunk_captures_exact_model_conditioning_and_mask(monkeypatch):
    _install_fake_comfy_runtime(monkeypatch)
    samples = _Nested(
        [
            torch.zeros((1, 24, 2, 4, 4)),
            torch.zeros((1, 32, 2, 4)),
        ]
    )
    mask = _Nested(
        [
            torch.ones((1, 1, 2, 4, 4)),
            torch.zeros((1, 1, 2, 4)),
        ]
    )
    conditioning = [[torch.tensor([4.0]), {"minimax_refs": [{"id": "ref"}]}]]
    model = object()

    with capture_chunk_refine_state() as captured:
        output = sample_chunk(
            model=model,
            conditioning=conditioning,
            latent={"samples": samples, "noise_mask": mask},
            sampler=object(),
            sigmas=torch.tensor([0.4, 0.0]),
            seed=7,
            enable_preview=False,
        )

    assert output["samples"] is samples
    assert len(captured) == 1
    assert captured[0]["model"] is model
    assert captured[0]["positive"] is conditioning
    assert captured[0]["noise_mask"] is mask


def test_v34_run_exposes_exact_refine_state_and_split_masks(monkeypatch):
    _install_fake_comfy_runtime(monkeypatch)
    first = [[torch.tensor([1.0]), {"chunk": 1}]]
    second = [[torch.tensor([2.0]), {"chunk": 2}]]
    models = [object(), object()]

    def fake_parent_run(self, **_kwargs):
        video_latents = []
        audio_latents = []
        for index, conditioning in enumerate((first, second)):
            samples = _Nested(
                [
                    torch.zeros((1, 24, 2, 4, 4)),
                    torch.zeros((1, 32, 2, 4)),
                ]
            )
            mask = _Nested(
                [
                    torch.full((1, 1, 2, 4, 4), float(index + 1)),
                    torch.full((1, 1, 2, 4), float(index + 3)),
                ]
            )
            sampled = sample_chunk(
                model=models[index],
                conditioning=conditioning,
                latent={"samples": samples, "noise_mask": mask},
                sampler=object(),
                sigmas=torch.tensor([0.4, 0.0]),
                seed=11 + index,
                enable_preview=False,
            )
            video, audio = sampled["samples"].unbind()
            video_latents.append({"samples": video})
            audio_latents.append({"samples": audio})
        return video_latents, audio_latents, {"target_frames": 240}, "status"

    monkeypatch.setattr(H3ContinuumSamplerProduction, "run", fake_parent_run)
    monkeypatch.setattr(
        driving_nodes,
        "_fresh_refine_model",
        lambda sampled_model, *, debug: ("fresh", sampled_model, bool(debug)),
    )
    outputs = H3ContinuumSamplerV34().run(
        chunks=2,
        chunk_seconds=5.0,
        width=64,
        height=64,
        audio_continuity=True,
        emit_refine_conditioning=True,
    )

    assert len(outputs[0]) == 2
    assert len(outputs[1]) == 2
    assert outputs[:5][2:] == ({"target_frames": 240}, "status", None)
    assert torch.equal(outputs[0][0]["noise_mask"], torch.ones((1, 1, 2, 4, 4)))
    assert torch.equal(outputs[1][0]["noise_mask"], torch.full((1, 1, 2, 4), 3.0))
    assert len(outputs[5]) == 2
    assert outputs[5][0]["api"] == 1
    assert outputs[5][0]["positive"] is first
    assert outputs[5][1]["positive"] is second
    assert outputs[5][0]["model"] == ("fresh", models[0], False)
    assert outputs[5][1]["model"] == ("fresh", models[1], False)


def test_v34_appends_refine_state_without_shifting_existing_outputs():
    assert H3ContinuumSamplerV34.RETURN_TYPES[:5] == (
        "LATENT",
        "LATENT",
        "H3_CONTINUUM_ASSEMBLY_PLAN",
        "STRING",
        "AUDIO",
    )
    assert H3ContinuumSamplerV34.RETURN_NAMES[:5] == (
        "video_latents",
        "audio_latents",
        "assembly_plan",
        "status",
        "driving_audio",
    )
    assert H3ContinuumSamplerV34.OUTPUT_IS_LIST[:5] == (
        True,
        True,
        False,
        False,
        False,
    )
    assert H3ContinuumSamplerV34.RETURN_TYPES[-1] == "H3_CONTINUUM_REFINE_STATE"
    assert H3ContinuumSamplerV34.RETURN_NAMES[-1] == "refine_state"
    assert H3ContinuumSamplerV34.OUTPUT_IS_LIST[-1] is True

    schema = H3ContinuumSamplerV34.INPUT_TYPES()
    required = schema["required"]
    assert list(required)[-1] == "emit_refine_conditioning"
    assert required["emit_refine_conditioning"][0] == "BOOLEAN"
    assert required["emit_refine_conditioning"][1]["default"] is False
    assert required["emit_refine_conditioning"][1]["advanced"] is True


def test_refine_state_is_opt_in_and_keeps_chunk_order(monkeypatch):
    first = [[torch.tensor([1.0]), {"chunk": 1}]]
    second = [[torch.tensor([2.0]), {"chunk": 2}]]
    videos = [{"samples": torch.zeros(1)}, {"samples": torch.zeros(1)}]
    captured = [
        {"model": "m1", "positive": first, "noise_mask": None},
        {"model": "m2", "positive": second, "noise_mask": None},
    ]
    monkeypatch.setattr(
        driving_nodes,
        "_fresh_refine_model",
        lambda sampled_model, *, debug: sampled_model,
    )

    assert _refine_state_output(
        enabled=False,
        captured=captured,
        video_latents=videos,
        debug=False,
    ) == []

    output = _refine_state_output(
        enabled=True,
        captured=captured,
        video_latents=videos,
        debug=False,
    )
    assert [item["model"] for item in output] == ["m1", "m2"]
    assert output[0]["positive"] is first
    assert output[1]["positive"] is second


def test_refine_state_refuses_run_storage_prefix_misalignment():
    videos = [
        {"samples": torch.zeros(1)},
        {"samples": torch.zeros(1)},
        {"samples": torch.zeros(1)},
    ]
    generated_suffix = [
        {"model": "m2", "positive": [[torch.tensor([2.0]), {}]], "noise_mask": None},
        {"model": "m3", "positive": [[torch.tensor([3.0]), {}]], "noise_mask": None},
    ]

    with pytest.raises(ValueError, match="Regenerate From = Chunk 1"):
        _refine_state_output(
            enabled=True,
            captured=generated_suffix,
            video_latents=videos,
            debug=False,
        )
