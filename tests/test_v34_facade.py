import torch

from ComfyUI_H3_Continuum_Join.nodes import NODE_CLASS_MAPPINGS
from ComfyUI_H3_Continuum_Join.v3 import driving_nodes
from ComfyUI_H3_Continuum_Join.v3.driving_nodes import (
    H3ContinuumAssembleSeamV34,
    H3ContinuumSamplerV34,
)


def test_v34_public_nodes_are_registered():
    assert NODE_CLASS_MAPPINGS["H3ContinuumSamplerV34"] is H3ContinuumSamplerV34
    assert NODE_CLASS_MAPPINGS["H3ContinuumAssembleSeamV34"] is H3ContinuumAssembleSeamV34


def test_v34_sampler_keeps_official_contract_and_eight_reference_capacity():
    schema = H3ContinuumSamplerV34.INPUT_TYPES()
    optional = schema["optional"]
    required = schema["required"]

    for index in range(1, 9):
        assert optional[f"reference_image_{index}"][0] == "IMAGE"
    assert optional["reference_video_1"][0] == "IMAGE"
    assert optional["driving_audio"][0] == "AUDIO"
    assert optional["audio_vae"][0] == "VAE"
    assert "reference_audio_1" not in optional
    assert "reference_audio_vae" not in optional
    assert "video_reference_size" in required
    assert H3ContinuumSamplerV34.RETURN_NAMES == (
        "video_latents",
        "audio_latents",
        "assembly_plan",
        "status",
        "driving_audio",
    )
    assert H3ContinuumSamplerV34.OUTPUT_IS_LIST == (True, True, False, False, False)


def test_v34_reference_mode_precedence_and_runtime_handoff(monkeypatch):
    captured = {}
    selected = {
        "waveform": torch.arange(24, dtype=torch.float32).reshape(1, 2, 12),
        "sample_rate": 32000,
    }

    def fake_run(self, **kwargs):
        captured.update(kwargs)
        return (["v"], ["a"], {"target_frames": 120}, "status", selected)

    monkeypatch.setattr(driving_nodes._V34RuntimeSampler, "run", fake_run)
    outputs = H3ContinuumSamplerV34().run(
        driving_audio=selected,
        audio_vae="audio-vae",
        reference_video_1="video-ref",
        video_reference_size="Efficient - 0.4 MP",
        first_frame="first",
        last_frame="last",
        reference_image_1="ref-1",
        reference_image_8="ref-8",
        marker="kept",
    )

    assert captured["first_frame"] is None
    assert captured["last_frame"] is None
    assert captured["reference_image_1"] == "ref-1"
    assert captured["reference_image_8"] == "ref-8"
    assert captured["reference_video_1"] == "video-ref"
    assert captured["driving_audio"] is selected
    assert captured["audio_vae"] == "audio-vae"
    assert captured["marker"] == "kept"
    assert torch.equal(
        outputs[2][driving_nodes._DRIVING_AUDIO_PLAN_KEY]["waveform"],
        selected["waveform"],
    )
    assert outputs[4]["sample_rate"] == 32000


def test_v34_assembler_prefers_preserved_plan_audio(monkeypatch):
    selected = {
        "waveform": torch.arange(20, dtype=torch.float32).reshape(1, 2, 10),
        "sample_rate": 1000,
    }

    def fake_parent(self, *args, **kwargs):
        return (
            torch.zeros((5, 4, 4, 3)),
            {"waveform": torch.ones((1, 2, 10)), "sample_rate": 1000},
            "base report",
        )

    monkeypatch.setattr(
        driving_nodes.H3ContinuumAssembleSeamExperimental,
        "assemble",
        fake_parent,
    )
    plan = {
        "target_frames": 5,
        "preserve_final_frame": False,
        driving_nodes._DRIVING_AUDIO_PLAN_KEY: selected,
    }
    images, audio, report = H3ContinuumAssembleSeamV34().assemble(
        images=[],
        audio=[],
        assembly_plan=plan,
        exact_total_duration=False,
        audio_seam="Auto",
        video_seam="Off",
        diagnostics="Basic",
        driving_audio={
            "waveform": torch.zeros((1, 2, 10)),
            "sample_rate": 1000,
        },
    )

    assert images.shape[0] == 5
    assert torch.equal(audio["waveform"], selected["waveform"])
    assert "assembly plan" in report
    assert "generated audio and Audio Seam bypassed" in report
