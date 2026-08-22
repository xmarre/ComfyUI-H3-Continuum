from __future__ import annotations

import pytest
import torch

from ComfyUI_H3_Continuum_Join.constants import CONTINUITY_OPTIONS
from ComfyUI_H3_Continuum_Join.v2 import sequence
from ComfyUI_H3_Continuum_Join.masked_continuation import (
    CONTINUATION_GUIDE,
    CONTINUATION_NATIVE_MASKED,
)
from ComfyUI_H3_Continuum_Join.state import make_plan
from ComfyUI_H3_Continuum_Join.temporal import (
    audio_latent_t,
    largest_context_capacity,
    video_latent_t,
)
from ComfyUI_H3_Continuum_Join.v3.plan import prepare_physical_decode_entries
from ComfyUI_H3_Continuum_Join.v3.driving_nodes import H3ContinuumSamplerV34


def test_terminal_merge_eligibility_preserves_fork_continuation_contracts():
    common = {
        "latent_only": True,
        "initial_state_present": False,
        "multi_chunk_flf": True,
        "chunk_seconds": 5.0,
        "timeline_video_source": None,
    }

    assert sequence._terminal_flf_merge_enabled(
        **common,
        chunks=2,
        continuation_method=CONTINUATION_NATIVE_MASKED,
        continuity=CONTINUITY_OPTIONS[2],
    )
    assert sequence._terminal_flf_merge_enabled(
        **common,
        chunks=3,
        continuation_method=CONTINUATION_GUIDE,
        continuity=CONTINUITY_OPTIONS[0],
    )
    assert not sequence._terminal_flf_merge_enabled(
        **common,
        chunks=3,
        continuation_method=CONTINUATION_NATIVE_MASKED,
        continuity=CONTINUITY_OPTIONS[2],
    )
    assert not sequence._terminal_flf_merge_enabled(
        **common,
        chunks=3,
        continuation_method=CONTINUATION_GUIDE,
        continuity=CONTINUITY_OPTIONS[2],
    )
    assert not sequence._terminal_flf_merge_enabled(
        **{**common, "latent_only": False},
        chunks=2,
        continuation_method=CONTINUATION_GUIDE,
        continuity=CONTINUITY_OPTIONS[0],
    )
    assert not sequence._terminal_flf_merge_enabled(
        **{**common, "timeline_video_source": object()},
        chunks=2,
        continuation_method=CONTINUATION_GUIDE,
        continuity=CONTINUITY_OPTIONS[0],
    )
    assert not sequence._terminal_flf_merge_enabled(
        **{**common, "initial_state_present": True},
        chunks=2,
        continuation_method=CONTINUATION_GUIDE,
        continuity=CONTINUITY_OPTIONS[0],
    )


def test_two_chunk_flf_native_validation_has_no_inter_sample_audio_boundary(monkeypatch):
    observed = {}

    def capture_validation(**kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(
        "ComfyUI_H3_Continuum_Join.v3.driving_nodes.validate_native_masked_request",
        capture_validation,
    )
    monkeypatch.setattr(
        H3ContinuumSamplerV34.__mro__[1],
        "run",
        lambda self, **kwargs: ([], [], {}, "status"),
    )

    H3ContinuumSamplerV34().run(
        continuation_method=CONTINUATION_NATIVE_MASKED,
        continuity=CONTINUITY_OPTIONS[0],
        chunks=2,
        chunk_seconds=5.0,
        width=32,
        height=32,
        first_frame=object(),
        last_frame=object(),
        audio_continuity=True,
    )

    assert observed["chunks"] == 1


@pytest.mark.parametrize(
    ("initial_pair", "physical_frames", "physical_context", "logical_frames", "logical_trims"),
    [
        (True, 243, 0, (124, 141), (0, 22)),
        (False, 260, 22, (141, 141), (22, 22)),
    ],
)
def test_terminal_pair_contract_matches_h3_grids(
    initial_pair, physical_frames, physical_context, logical_frames, logical_trims
):
    contract = sequence._terminal_pair_contract(
        initial_pair=initial_pair,
        chunk_seconds=5.0,
    )

    assert contract["physical_frames"] == physical_frames
    assert contract["physical_context_frames"] == physical_context
    assert contract["logical_frames"] == logical_frames
    assert contract["logical_trims"] == logical_trims


def test_terminal_pair_prompt_preserves_distinct_timeline_sections():
    prompt, policy = sequence._terminal_pair_prompt(
        ["opening", "middle", "finish"],
        pair_start=1,
        chunk_seconds=5.0,
    )

    assert prompt == "[0-5s]\nmiddle\n\n[5-10s]\nfinish"
    assert policy == sequence.TERMINAL_PROMPT_POLICY_TIMELINE
    assert sequence._terminal_pair_prompt(
        ["same", "same"], pair_start=0, chunk_seconds=5.0
    ) == ("same", sequence.TERMINAL_PROMPT_POLICY_SHARED)


def test_terminal_storage_contract_invalidates_only_the_pair():
    plan = {"hashes": ["a", "b", "c", "d"], "prompts": ["A", "B", "C", "D"]}
    salted = sequence._terminal_storage_plan(
        plan,
        enabled=True,
        prompt_policy=sequence.TERMINAL_PROMPT_POLICY_SHARED,
    )

    assert salted is not plan
    assert salted["hashes"][:2] == plan["hashes"][:2]
    assert salted["hashes"][2:] != plan["hashes"][2:]
    assert plan["hashes"] == ["a", "b", "c", "d"]


def _logical_terminal_entries(*, initial_pair: bool):
    contract = sequence._terminal_pair_contract(
        initial_pair=initial_pair,
        chunk_seconds=5.0,
    )
    video = torch.arange(
        24 * video_latent_t(contract["physical_frames"]), dtype=torch.float32
    ).reshape(1, 24, video_latent_t(contract["physical_frames"]), 1, 1)
    audio = torch.arange(
        32 * 2 * audio_latent_t(contract["physical_frames"]), dtype=torch.float32
    ).reshape(1, 32, 2, audio_latent_t(contract["physical_frames"]))
    parts = sequence._split_terminal_merged_latents(video, audio, contract)
    entries = []
    clip_start = 1 if initial_pair else 2
    for role, (video_part, audio_part) in enumerate(parts):
        total_frames = contract["logical_frames"][role]
        trim_frames = contract["logical_trims"][role]
        plan = make_plan(
            continuation=trim_frames > 0,
            clip_index=clip_start + role,
            total_frames=total_frames,
            trim_frames=trim_frames,
            width=16,
            height=16,
            context_frames=trim_frames if trim_frames else 5,
            state_capacity_frames=largest_context_capacity(total_frames - trim_frames),
            requested_extend_seconds=5.0,
            debug=False,
        )
        entries.append(
            {
                "video": video_part,
                "audio": audio_part,
                "plan": sequence._mark_terminal_plan(
                    plan, contract=contract, role=role
                ),
            }
        )
    return entries, video, audio


def _initial_entry():
    frames = 124
    return {
        "video": torch.zeros((1, 24, video_latent_t(frames), 1, 1)),
        "audio": torch.zeros((1, 32, 2, audio_latent_t(frames))),
        "plan": make_plan(
            continuation=False,
            clip_index=1,
            total_frames=frames,
            trim_frames=0,
            width=16,
            height=16,
            context_frames=5,
            state_capacity_frames=39,
            requested_extend_seconds=5.0,
            debug=False,
        ),
    }


def test_two_logical_chunks_recombine_into_one_bit_exact_decode_group():
    entries, physical_video, physical_audio = _logical_terminal_entries(
        initial_pair=True
    )

    decode_entries, plan = prepare_physical_decode_entries(
        entries,
        chunk_seconds=5.0,
        preserve_final_frame=True,
        terminal_merged=True,
    )

    assert len(decode_entries) == 1
    assert torch.equal(decode_entries[0]["video"], physical_video)
    assert torch.equal(decode_entries[0]["audio"], physical_audio)
    assert plan["logical_chunk_count"] == 2
    assert plan["physical_decode_group_count"] == 1
    assert plan["decode_groups"][0]["logical_chunk_indices"] == [1, 2]
    assert plan["decode_groups"][0]["net_frames"] == 243


def test_long_sequence_keeps_prior_chunk_and_one_terminal_decode_group():
    terminal, physical_video, physical_audio = _logical_terminal_entries(
        initial_pair=False
    )
    entries = [_initial_entry(), *terminal]

    decode_entries, plan = prepare_physical_decode_entries(
        entries,
        chunk_seconds=5.0,
        preserve_final_frame=True,
        terminal_merged=True,
    )

    assert len(decode_entries) == 2
    assert torch.equal(decode_entries[-1]["video"], physical_video)
    assert torch.equal(decode_entries[-1]["audio"], physical_audio)
    assert plan["physical_decode_group_count"] == 2
    assert plan["decode_groups"][-1]["logical_chunk_indices"] == [2, 3]
    assert plan["natural_frames"] == 362


def test_terminal_recombine_rejects_changed_overlap():
    entries, _physical_video, _physical_audio = _logical_terminal_entries(
        initial_pair=True
    )
    entries[-1]["video"] = entries[-1]["video"].clone()
    entries[-1]["video"][:, :, 0] += 1

    with pytest.raises(ValueError, match="overlap differs"):
        prepare_physical_decode_entries(
            entries,
            chunk_seconds=5.0,
            preserve_final_frame=True,
            terminal_merged=True,
        )


def test_terminal_prefix_is_atomic_for_partial_reroll_and_stale_contract():
    terminal, _video, _audio = _logical_terminal_entries(initial_pair=False)
    entries = [_initial_entry(), *terminal]

    preserved, reset = sequence._atomic_terminal_prefix(
        entries[:2], chunks=3, reroll_from_chunk=0, chunk_seconds=5.0
    )
    assert reset and len(preserved) == 1

    preserved, reset = sequence._remove_inactive_terminal_prefix(entries)
    assert reset and len(preserved) == 1

    preserved, reset = sequence._atomic_terminal_prefix(
        entries, chunks=3, reroll_from_chunk=3, chunk_seconds=5.0
    )
    assert reset and len(preserved) == 1

    stale = [dict(entry) for entry in entries]
    stale[-1] = {**stale[-1], "plan": dict(stale[-1]["plan"])}
    stale[-1]["plan"].pop("terminal_merge")
    preserved, reset = sequence._atomic_terminal_prefix(
        stale, chunks=3, reroll_from_chunk=0, chunk_seconds=5.0
    )
    assert reset and len(preserved) == 1


def test_changed_first_resets_session_and_changed_last_invalidates_tail(monkeypatch):
    monkeypatch.setattr(sequence, "validate_session", lambda session: session)
    monkeypatch.setattr(sequence, "validate_chunk_entry", lambda entry: entry)
    session = {
        "width": 96,
        "height": 64,
        "chunk_seconds": 5.0,
        "identity_hash": "identity",
        "settings": {"first_frame_hash": "first", "last_frame_hash": "last"},
        "chunks": [
            {"prompt_hash": "a", "plan": {"net_frames": 124}},
            {"prompt_hash": "b", "plan": {"net_frames": 119}},
            {"prompt_hash": "c", "plan": {"net_frames": 119}},
        ],
    }
    common = {
        "session": session,
        "prompt_hashes": ["a", "b", "c"],
        "chunks": 3,
        "reroll_from_chunk": 0,
        "width": 96,
        "height": 64,
        "chunk_seconds": 5.0,
        "identity_hash": "identity",
        "continuation_method": CONTINUATION_GUIDE,
    }

    preserved, _notes = sequence._preserved_prefix(
        **common, first_frame_hash="changed", last_frame_hash="last"
    )
    assert preserved == []

    preserved, notes = sequence._preserved_prefix(
        **common, first_frame_hash="first", last_frame_hash="changed"
    )
    assert len(preserved) == 2
    assert any("Last Frame differs" in note for note in notes)
