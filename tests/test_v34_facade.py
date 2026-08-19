from types import SimpleNamespace

import torch

from ComfyUI_H3_Continuum_Join.nodes import NODE_CLASS_MAPPINGS
from ComfyUI_H3_Continuum_Join.v3.assembly import (
    AUDIO_SEAM_OFF,
    H3ContinuumAssembleSeamExperimental,
)
from ComfyUI_H3_Continuum_Join.v3.driving_nodes import (
    H3ContinuumAssembleSeamV34,
    H3ContinuumSamplerV34,
    _prompt_graph_with_audio_vae_alias,
    _reference_video_storage_contract,
)


def test_v34_public_nodes_are_registered():
    assert NODE_CLASS_MAPPINGS["H3ContinuumSamplerV34"] is H3ContinuumSamplerV34
    assert NODE_CLASS_MAPPINGS["H3ContinuumAssembleSeamV34"] is H3ContinuumAssembleSeamV34


def test_v34_sampler_matches_release_contract_and_keeps_eight_reference_capacity():
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


def test_persistent_reference_video_storage_contract_has_one_record_per_chunk():
    source = SimpleNamespace(
        contract={
            "reference_video_contract_version": 1,
            "source_sha256": "source",
            "combined_hash": "combined",
        },
        combined_hash="combined",
    )
    contract = _reference_video_storage_contract(source, chunks=3)
    assert contract["reference_video_contract_version"] == 1
    assert [item["chunk_number"] for item in contract["chunk_slices"]] == [1, 2, 3]
    assert all(item["persistent"] is True for item in contract["chunk_slices"])
    assert all(item["reference_video_hash"] == "combined" for item in contract["chunk_slices"])


def test_run_storage_audio_vae_alias_does_not_mutate_queued_prompt():
    prompt = {
        "17": {
            "class_type": "H3ContinuumSamplerV34",
            "inputs": {"audio_vae": ["9", 0]},
        }
    }
    adapted = _prompt_graph_with_audio_vae_alias(prompt, "17", enabled=True)
    assert "reference_audio_vae" not in prompt["17"]["inputs"]
    assert adapted["17"]["inputs"]["reference_audio_vae"] == ["9", 0]


def test_v34_assembler_selects_preserved_driving_audio_and_disables_audio_seam(monkeypatch):
    seen = {}

    def fake_assemble(
        self,
        images,
        audio,
        assembly_plan,
        exact_total_duration,
        audio_seam,
        video_seam,
        diagnostics,
    ):
        seen["audio_seam"] = audio_seam
        return (
            torch.zeros(5, 4, 4, 3),
            {"waveform": torch.ones(1, 2, 50), "sample_rate": 1000},
            "base report",
        )

    monkeypatch.setattr(H3ContinuumAssembleSeamExperimental, "assemble", fake_assemble)
    driving = {
        "waveform": torch.arange(40, dtype=torch.float32).reshape(1, 2, 20),
        "sample_rate": 1000,
    }
    images, audio, report = H3ContinuumAssembleSeamV34().assemble(
        images=[],
        audio=[],
        assembly_plan={},
        exact_total_duration=True,
        audio_seam="Auto",
        video_seam="Off",
        diagnostics="Basic",
        driving_audio=driving,
    )
    assert images.shape[0] == 5
    assert seen["audio_seam"] == AUDIO_SEAM_OFF
    assert audio["sample_rate"] == 1000
    assert torch.equal(audio["waveform"], driving["waveform"])
    assert "preserved effective source" in report
