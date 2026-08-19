"""Stable V3.4 Driving Audio and Video Reference facade.

V3.4 keeps the proven latent-first Continuum engine and adds two production
conditioning sources: a persistent native H3 video reference and absolute-time
driving audio whose effective source stream is preserved for final output.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import torch

from ..compatibility import accelerator_summary, check_comfy_h3_runtime
from ..constants import (
    CONTINUUM_ACTUAL_PREFIX_STEPS,
    DIAGNOSTICS_FULL,
    DIAGNOSTICS_OFF,
    FPS,
    normalize_diagnostics_mode,
)
from ..continuation import POLICY_REPLACE, prepare_conditioning
from ..driving_audio import (
    attach_driving_audio,
    combine_driving_audio_identity,
    encode_driving_audio,
    prepare_driving_audio_source,
    slice_driving_audio_latent,
)
from ..media import validate_audio
from ..model_patch import clone_model_for_chunk
from ..reference import (
    encode_reference_latents,
    prepare_reference_assets,
    validate_reference_prompts,
)
from ..reference_video import (
    REFERENCE_VIDEO_SIZE_OPTIONS,
    combine_reference_video_identity,
    encode_reference_video,
    prepare_reference_video_source,
    validate_reference_video_prompts,
)
from ..run_storage import (
    automatic_project_key,
    get_active_run_storage,
    resolve_run_storage_name,
    run_storage_scope,
)
from ..state import (
    assert_context_unchanged,
    context_fingerprint,
    make_plan,
    select_context,
)
from ..temporal import (
    align_frame_count_up,
    largest_context_capacity,
    make_extension_shape,
    make_extension_shape_at_least,
)
from ..version import PACKAGE_VERSION
from ..v2.h3_builder import (
    attach_keyframes,
    empty_h3_latent,
    encode_identity_latents,
    prepare_identity_assets,
)
from ..v2.motion import choose_context_frames
from ..v2.prompts import build_sampler_prompt_plan, prompt_plan_report, validate_prompt_plan
from ..v2.sampling import sample_chunk
from ..v2.seeds import derive_chunk_seed
from ..v2.sequence import _conditioning_cache, _preserved_prefix
from ..v2.session import (
    entry_to_state,
    make_chunk_entry,
    make_session,
    model_fingerprint,
    session_summary,
)
from .assembly import (
    AUDIO_SEAM_OFF,
    H3ContinuumAssembleSeamExperimental,
    _singleton,
)
from .nodes import (
    CATEGORY,
    DIAGNOSTICS_OPTIONS,
    H3ContinuumSamplerProduction,
    REGENERATE_AUTO,
    _regenerate_from_value,
    _validate_regenerate_storage,
)
from .plan import make_assembly_plan


RUN_STORAGE_OFF = "Off"
RUN_STORAGE_AUTO = "Save + Auto Resume"


def _reference_video_storage_contract(source, *, chunks: int) -> dict[str, Any] | None:
    """Adapt a persistent V3.4 video reference to the existing safe storage slot.

    Run Storage already hashes one video-conditioning contract globally plus one
    deterministic slice payload per chunk. A persistent reference has the same
    dependency shape, except every chunk uses the same source, so repeat a tiny
    per-chunk identity record instead of teaching persistence about a second
    equivalent video slot.
    """
    if source is None:
        return None
    contract = dict(source.contract)
    contract["chunk_slices"] = [
        {
            "chunk_number": index + 1,
            "reference_video_hash": source.combined_hash,
            "persistent": True,
        }
        for index in range(int(chunks))
    ]
    return contract


def _prompt_graph_with_audio_vae_alias(prompt: Any, unique_id: Any, *, enabled: bool):
    """Expose V3.4 audio_vae to the existing Run Storage VAE-route fingerprint.

    The persistence layer historically named this route reference_audio_vae.
    V3.4 removed Reference Audio from the stable facade and renamed the public
    VAE input to audio_vae. The alias exists only in the copied graph inspected
    for hashing; the queued workflow and runtime inputs are never mutated.
    """
    if not enabled or not isinstance(prompt, dict) or unique_id is None:
        return prompt
    copied = copy.deepcopy(prompt)
    node = copied.get(str(unique_id))
    if not isinstance(node, dict):
        node = copied.get(unique_id)
    inputs = node.get("inputs") if isinstance(node, dict) else None
    if isinstance(inputs, dict) and "audio_vae" in inputs:
        inputs["reference_audio_vae"] = copy.deepcopy(inputs["audio_vae"])
    return copied


def _run_v34_sequence(
    *,
    model: Any,
    clip: Any,
    video_vae: Any,
    audio_vae: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    prompt_plan: dict[str, Any],
    first_frame: torch.Tensor | None,
    last_frame: torch.Tensor | None,
    reference_assets,
    reference_video_source,
    driving_audio_source,
    width: int,
    height: int,
    continuity: str,
    base_seed: int,
    audio_continuity: bool,
    diagnostics: str,
    reroll_from_chunk: int,
    reroll_nonce: int,
    show_preview: bool,
):
    """Sample raw V3.4 AV chunks without changing the legacy V3.3 engine path."""
    from ..conditioning import conditioning_mode_label, detect_conditioning_mode

    plan = validate_prompt_plan(prompt_plan)
    chunks = int(plan["chunks"])
    chunk_seconds = float(plan["chunk_seconds"])
    prompts = list(plan["prompts"])
    prompt_hashes = list(plan["hashes"])
    width, height = int(width), int(height)
    diagnostics_mode = normalize_diagnostics_mode(str(diagnostics))
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("width and height must be positive multiples of 32")
    if not 0 <= int(reroll_from_chunk) <= chunks:
        raise ValueError("Regenerate From must be Auto or a configured chunk")

    conditioning_mode = detect_conditioning_mode(
        first_frame=first_frame,
        last_frame=last_frame,
        reference_assets=reference_assets,
    )
    if reference_assets is not None and (first_frame is not None or last_frame is not None):
        raise ValueError("Reference Images cannot be combined with First Frame or Last Frame")

    notes: list[str] = []
    if reference_assets is not None:
        warning = validate_reference_prompts(prompts, reference_assets.count)
        notes.append(
            f"Reference conditioning: {reference_assets.count} image(s), "
            f"size={reference_assets.size_mode}; persistent across all chunks."
        )
        if warning:
            notes.append(warning)
    if reference_video_source is not None:
        warning = validate_reference_video_prompts(prompts, reference_video_source)
        notes.append(
            "Video Reference: persistent native H3 video reference, "
            f"size={reference_video_source.size_mode}, "
            f"resolved={reference_video_source.target_width}x"
            f"{reference_video_source.target_height}."
        )
        if warning:
            notes.append(warning)
    if driving_audio_source is not None:
        notes.append(
            "Driving Audio: absolute-time guide active; generated audio continuity "
            "is disabled and the effective source stream is preserved for final output."
        )

    runtime_issues = check_comfy_h3_runtime()
    if runtime_issues:
        notes.append("Core compatibility notes: " + "; ".join(runtime_issues))

    assets = prepare_identity_assets(
        video_vae,
        width=width,
        height=height,
        first_frame=first_frame,
        last_frame=last_frame,
        encode_latents=False,
    )
    visual_identity_hash = (
        reference_assets.combined_hash if reference_assets is not None else assets.identity_hash
    )
    sequence_identity_hash = combine_reference_video_identity(
        visual_identity_hash, reference_video_source
    )
    sequence_identity_hash = combine_driving_audio_identity(
        sequence_identity_hash, driving_audio_source
    )
    current_model_fingerprint = model_fingerprint(
        model, extra_wrapper_keys=("h3_continuum_join.apply_model.v1",)
    )

    storage_controller = get_active_run_storage()
    session = None
    if storage_controller is not None:
        stored_session = storage_controller.prepare(
            model=model,
            model_fingerprint_value=current_model_fingerprint,
            clip=clip,
            video_vae=video_vae,
            sampler=sampler,
            sigmas=sigmas,
            prompt_plan=plan,
            width=width,
            height=height,
            chunk_seconds=chunk_seconds,
            continuity=continuity,
            audio_continuity=(bool(audio_continuity) and driving_audio_source is None),
            base_seed=int(base_seed),
            reroll_from_chunk=int(reroll_from_chunk),
            reroll_nonce=int(reroll_nonce),
            first_frame_hash=assets.first_frame_hash,
            last_frame_hash=assets.last_frame_hash,
            identity_hash=sequence_identity_hash,
            strict_compatibility=False,
            existing_session=None,
            reference_contract=(
                reference_assets.contract if reference_assets is not None else None
            ),
            conditioning_mode=conditioning_mode,
            reference_audio_contract=(
                driving_audio_source.contract if driving_audio_source is not None else None
            ),
            reference_audio_vae=(audio_vae if driving_audio_source is not None else None),
            timeline_video_contract=_reference_video_storage_contract(
                reference_video_source, chunks=chunks
            ),
        )
        reroll_nonce = storage_controller.effective_reroll_nonce
        if stored_session is not None:
            session = stored_session

    effective_reroll = (
        0
        if storage_controller is not None
        and session is not None
        and bool((session.get("settings") or {}).get("run_storage_validated_prefix"))
        else int(reroll_from_chunk)
    )
    preserved, reuse_notes = _preserved_prefix(
        session=session,
        prompt_hashes=prompt_hashes,
        chunks=chunks,
        reroll_from_chunk=effective_reroll,
        width=width,
        height=height,
        chunk_seconds=chunk_seconds,
        identity_hash=sequence_identity_hash,
    )
    notes.extend(reuse_notes)

    reference_video_assets = None
    driving_audio_assets = None
    cache = {}
    if len(preserved) < chunks:
        assets = encode_identity_latents(video_vae, assets)
        if reference_assets is not None:
            reference_assets = encode_reference_latents(video_vae, reference_assets)
        if reference_video_source is not None:
            reference_video_assets = encode_reference_video(video_vae, reference_video_source)
        if driving_audio_source is not None:
            driving_audio_assets = encode_driving_audio(driving_audio_source, audio_vae)
        cache = _conditioning_cache(
            clip=clip,
            prompts=prompts,
            assets=assets,
            final_has_last_frame=last_frame is not None,
            reference_assets=reference_assets,
            reference_audio_assets=None,
            timeline_video_assets=reference_video_assets,
        )

    entries = list(preserved)
    previous_state = entry_to_state(entries[-1]) if entries else None
    initial_frame_count = align_frame_count_up(int(round(chunk_seconds * FPS)))
    retained_frames = sum(int(entry["plan"]["net_frames"]) for entry in entries)
    sampling_reports: list[str] = []

    for sequence_index in range(len(entries), chunks):
        prompt = prompts[sequence_index]
        prompt_hash_value = prompt_hashes[sequence_index]
        is_final = sequence_index == chunks - 1
        effective_nonce = (
            int(reroll_nonce)
            if int(reroll_from_chunk) > 0
            and sequence_index + 1 >= int(reroll_from_chunk)
            else 0
        )
        seed = derive_chunk_seed(base_seed, sequence_index, effective_nonce)
        video_context = None
        audio_context = None
        context_before = None
        motion_score = 0.0
        include_last = bool(last_frame is not None and is_final)

        if previous_state is None:
            total_frames = initial_frame_count
            trim_frames = 0
            context_frames = 0
            clip_index = 1
            reason = "initial clip"
            latent = empty_h3_latent(width, height, total_frames)
            conditioning = attach_keyframes(
                cache[(prompt, include_last)],
                frame_count=total_frames,
                first_latent=assets.first_latent,
                last_latent=assets.last_latent if is_final else None,
            )
            chunk_plan = make_plan(
                continuation=False,
                clip_index=clip_index,
                total_frames=total_frames,
                trim_frames=0,
                width=width,
                height=height,
                context_frames=5,
                state_capacity_frames=largest_context_capacity(total_frames),
                requested_extend_seconds=chunk_seconds,
                debug=False,
            )
        else:
            context_frames, motion_score, reason = choose_context_frames(
                continuity, previous_state
            )
            desired_cumulative = int(round((sequence_index + 1) * chunk_seconds * FPS))
            requested_new_frames = max(1, desired_cumulative - retained_frames)
            shape = (
                make_extension_shape_at_least(context_frames, requested_new_frames)
                if is_final
                else make_extension_shape(context_frames, requested_new_frames / FPS)
            )
            total_frames = int(shape.total_frames)
            trim_frames = int(context_frames)
            latent = empty_h3_latent(width, height, total_frames)
            base_conditioning = attach_keyframes(
                cache[(prompt, include_last)],
                frame_count=total_frames,
                first_latent=assets.first_latent,
                last_latent=assets.last_latent if is_final else None,
            )
            video_context, audio_context, grid_offset = select_context(
                previous_state,
                context_frames,
                include_audio=(bool(audio_continuity) and driving_audio_assets is None),
            )
            context_before = context_fingerprint(video_context, audio_context)
            conditioning = prepare_conditioning(
                base_conditioning,
                video_context=video_context,
                audio_context=audio_context,
                audio_grid_offset=grid_offset,
                context_frames=context_frames,
                new_frame_count=total_frames,
                first_frame_policy=POLICY_REPLACE,
                preserve_last_frame=True,
            )
            clip_index = int(previous_state["clip_index"]) + 1
            chunk_plan = make_plan(
                continuation=True,
                clip_index=clip_index,
                total_frames=total_frames,
                trim_frames=trim_frames,
                width=width,
                height=height,
                context_frames=context_frames,
                state_capacity_frames=largest_context_capacity(shape.net_new_frames),
                requested_extend_seconds=chunk_seconds,
                debug=False,
            )

        driving_slice = slice_driving_audio_latent(
            driving_audio_assets,
            cumulative_retained_before=retained_frames,
            total_frames=total_frames,
            trim_frames=trim_frames,
            fps=int(FPS),
        )
        conditioning = attach_driving_audio(conditioning, driving_slice)

        chunk_model = clone_model_for_chunk(
            model,
            strict=False,
            debug=False,
            chunk_index=clip_index,
            context_frames=context_frames if previous_state is not None else None,
        )
        sampled = sample_chunk(
            model=chunk_model,
            conditioning=conditioning,
            latent=latent,
            sampler=sampler,
            sigmas=sigmas,
            seed=seed,
            enable_preview=bool(show_preview),
        )
        if context_before is not None and video_context is not None:
            assert_context_unchanged(video_context, audio_context, context_before)
        entry = make_chunk_entry(
            latent=sampled,
            plan=chunk_plan,
            prompt=prompt,
            prompt_hash=prompt_hash_value,
            seed=seed,
            context_frames=context_frames,
            motion_score=motion_score,
            reused=False,
        )
        previous_state = entry_to_state(entry)
        entries.append(entry)
        if storage_controller is not None:
            storage_controller.commit_chunk(entry, position=sequence_index)
        retained_frames += int(chunk_plan["net_frames"])
        if diagnostics_mode != DIAGNOSTICS_OFF:
            sampling_reports.append(
                f"chunk {sequence_index + 1}/{chunks}: seed={seed}, "
                f"frames={chunk_plan['total_frames']}, trim={chunk_plan['trim_frames']}, "
                f"retained_total={retained_frames}, context={context_frames} ({reason}), "
                f"motion={motion_score:.6f}, "
                + (
                    f"interop=emitted actual_prefix={CONTINUUM_ACTUAL_PREFIX_STEPS}"
                    if context_before is not None
                    else "interop=not_emitted"
                )
            )
        del sampled, latent, conditioning, chunk_model

    if len(entries) != chunks:
        raise RuntimeError(
            f"internal V3.4 sequence length mismatch: expected {chunks}, got {len(entries)}"
        )
    target_frames = int(round(chunks * chunk_seconds * FPS))
    retained_total = sum(int(entry["plan"]["net_frames"]) for entry in entries)
    if retained_total < target_frames:
        raise RuntimeError(
            f"V3.4 sequence retained {retained_total} frames for {target_frames}-frame "
            "target; refusing to manufacture a frozen tail"
        )

    last_state = entry_to_state(entries[-1])
    settings = {
        "continuity": continuity,
        "audio_continuity": bool(audio_continuity) and driving_audio_source is None,
        "exact_total_duration": False,
        "prompt_mode": plan["mode"],
        "conditioning_mode": conditioning_mode,
        "base_seed": int(base_seed),
        "reroll_nonce": int(reroll_nonce),
        "diagnostics_mode": diagnostics_mode,
        "latent_first": True,
        "first_frame_hash": assets.first_frame_hash,
        "last_frame_hash": assets.last_frame_hash,
        "reference_contract": reference_assets.contract if reference_assets is not None else None,
        "reference_video_contract": (
            reference_video_source.contract if reference_video_source is not None else None
        ),
        "driving_audio_contract": (
            driving_audio_source.contract if driving_audio_source is not None else None
        ),
    }
    new_session = make_session(
        chunks=entries,
        width=width,
        height=height,
        chunk_seconds=chunk_seconds,
        identity_hash=sequence_identity_hash,
        model_fingerprint_value=current_model_fingerprint,
        parent_session_id=session.get("session_id") if session is not None else None,
        reroll_from_chunk=int(reroll_from_chunk),
        settings=settings,
    )
    report_lines = [
        f"H3 Continuum V3.4 {PACKAGE_VERSION}",
        f"Conditioning mode: {conditioning_mode_label(conditioning_mode)}.",
        prompt_plan_report(plan),
        "Decode: external ComfyUI Core VAE nodes; full raw AV chunks retained.",
        accelerator_summary(model),
        *notes,
    ]
    if diagnostics_mode != DIAGNOSTICS_OFF:
        report_lines.extend(sampling_reports)
    report_lines.extend(
        [
            session_summary(new_session),
            f"Output: {len(entries)} raw AV latent chunk(s); retained plan={retained_total} "
            f"frames for exact target={target_frames}.",
        ]
    )
    return entries, last_state, new_session, "\n".join(report_lines)


class H3ContinuumSamplerV34(H3ContinuumSamplerProduction):
    """Stable V3.4 sampler with Driving Audio and persistent Video Reference."""

    DEPRECATED = False
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "H3 Continuum V3.4 production sampler with restartable chunks, up to eight "
        "ordered image references, persistent Video Reference, and Driving Audio."
    )
    SEARCH_ALIASES = [
        "H3 Continuum Sampler V3.4",
        "H3 Driving Audio",
        "H3 Video Reference",
    ]
    RETURN_TYPES = (
        "LATENT",
        "LATENT",
        "H3_CONTINUUM_ASSEMBLY_PLAN",
        "STRING",
        "AUDIO",
    )
    RETURN_NAMES = (
        "video_latents",
        "audio_latents",
        "assembly_plan",
        "status",
        "driving_audio",
    )
    OUTPUT_IS_LIST = (True, True, False, False, False)

    @classmethod
    def INPUT_TYPES(cls):
        schema = H3ContinuumSamplerProduction.INPUT_TYPES()
        required = {}
        for name, definition in schema["required"].items():
            required[name] = definition
            if name == "project_id":
                required["video_reference_size"] = (
                    REFERENCE_VIDEO_SIZE_OPTIONS,
                    {
                        "default": REFERENCE_VIDEO_SIZE_OPTIONS[0],
                        "display_name": "Video Reference Size",
                        "advanced": True,
                        "tooltip": (
                            "Efficient is the normal production default; Balanced keeps more "
                            "reference detail; Match Output can be substantially heavier."
                        ),
                    },
                )
        schema["required"] = required
        optional = dict(schema.get("optional", {}))
        optional.pop("reference_audio_1", None)
        optional.pop("reference_audio_vae", None)
        optional["reference_video_1"] = (
            "IMAGE",
            {
                "display_name": "Video Reference",
                "tooltip": (
                    "Optional persistent H3 video reference as an IMAGE frame batch. "
                    "It guides appearance, motion, framing, and timing across every chunk."
                ),
            },
        )
        optional["driving_audio"] = (
            "AUDIO",
            {
                "tooltip": (
                    "Optional source audio that guides absolute sequence time and is preserved "
                    "as the final audio stream."
                )
            },
        )
        optional["audio_vae"] = (
            "VAE",
            {"tooltip": "Required when Driving Audio is connected."},
        )
        schema["optional"] = optional
        return schema

    def run(
        self,
        model,
        clip,
        video_vae,
        sampler,
        sigmas,
        sequence_prompt,
        prompt_mode,
        chunks,
        chunk_seconds,
        width,
        height,
        continuity,
        base_seed,
        audio_continuity,
        diagnostics,
        reroll_from_chunk,
        reroll_nonce,
        strict_compatibility=False,
        debug=False,
        show_preview=True,
        run_storage=RUN_STORAGE_OFF,
        run_name="",
        reference_size="Match Output",
        project_id="",
        video_reference_size=REFERENCE_VIDEO_SIZE_OPTIONS[0],
        first_frame=None,
        last_frame=None,
        reference_image_1=None,
        reference_image_2=None,
        reference_image_3=None,
        reference_image_4=None,
        reference_image_5=None,
        reference_image_6=None,
        reference_image_7=None,
        reference_image_8=None,
        reference_video_1=None,
        driving_audio=None,
        audio_vae=None,
        prompt=None,
        unique_id=None,
        **_unused,
    ):
        del strict_compatibility, debug  # V3.4 is intentionally Core-first/permissive.
        runtime_started_at = time.perf_counter()
        chunks = int(chunks)
        chunk_seconds = float(chunk_seconds)
        regenerate_from = _regenerate_from_value(reroll_from_chunk, chunks=chunks)
        _validate_regenerate_storage(run_storage, regenerate_from)

        plan = build_sampler_prompt_plan(
            prompt_mode=prompt_mode,
            prompt_script=sequence_prompt,
            sequence_prompt=sequence_prompt,
            prompt_plan=None,
            chunks=chunks,
            chunk_seconds=chunk_seconds,
        )
        reference_assets = prepare_reference_assets(
            reference_image_1=reference_image_1,
            reference_image_2=reference_image_2,
            reference_image_3=reference_image_3,
            reference_image_4=reference_image_4,
            reference_image_5=reference_image_5,
            reference_image_6=reference_image_6,
            reference_image_7=reference_image_7,
            reference_image_8=reference_image_8,
            output_width=int(width),
            output_height=int(height),
            size_mode=str(reference_size),
        )
        target_frames = int(round(chunks * chunk_seconds * FPS))
        reference_video_source = prepare_reference_video_source(
            reference_video_1,
            target_frames=target_frames,
            output_width=int(width),
            output_height=int(height),
            size_mode=str(video_reference_size),
        )
        driving_source = prepare_driving_audio_source(
            driving_audio,
            audio_vae,
            target_frames=target_frames,
            fps=int(FPS),
        )

        def execute():
            entries, last_state, session, report = _run_v34_sequence(
                model=model,
                clip=clip,
                video_vae=video_vae,
                audio_vae=audio_vae,
                sampler=sampler,
                sigmas=sigmas,
                prompt_plan=plan,
                first_frame=first_frame,
                last_frame=last_frame,
                reference_assets=reference_assets,
                reference_video_source=reference_video_source,
                driving_audio_source=driving_source,
                width=int(width),
                height=int(height),
                continuity=continuity,
                base_seed=int(base_seed),
                audio_continuity=bool(audio_continuity),
                diagnostics=diagnostics,
                reroll_from_chunk=regenerate_from,
                reroll_nonce=int(reroll_nonce),
                show_preview=bool(show_preview),
            )
            assembly_plan = make_assembly_plan(
                entries,
                chunk_seconds=chunk_seconds,
                preserve_final_frame=last_frame is not None,
            )
            assembly_plan = dict(assembly_plan)
            assembly_plan["_runtime_started_at"] = runtime_started_at
            video_latents = [{"samples": entry["video"]} for entry in entries]
            audio_latents = [{"samples": entry["audio"]} for entry in entries]
            return video_latents, audio_latents, assembly_plan, last_state, session, report

        if str(run_storage) == RUN_STORAGE_OFF:
            video_latents, audio_latents, assembly_plan, _, _, report = execute()
        elif str(run_storage) == RUN_STORAGE_AUTO:
            storage_name = resolve_run_storage_name(
                project_id=project_id,
                legacy_run_name=run_name,
                automatic_key=automatic_project_key(prompt, unique_id),
            )
            storage_prompt = _prompt_graph_with_audio_vae_alias(
                prompt, unique_id, enabled=driving_source is not None
            )
            with run_storage_scope(
                storage_name, prompt=storage_prompt, unique_id=unique_id
            ) as storage:
                video_latents, audio_latents, assembly_plan, _, session, report = execute()
                report = report + "\n" + storage.summary(
                    detailed=str(diagnostics) == DIAGNOSTICS_FULL
                )
                storage.finalize(session=session, report=report)
        else:
            raise ValueError(f"unknown Run Storage mode: {run_storage!r}")

        effective_audio = driving_source.source_audio if driving_source is not None else None
        return video_latents, audio_latents, assembly_plan, report, effective_audio


class H3ContinuumAssembleSeamV34(H3ContinuumAssembleSeamExperimental):
    """V3.4 assembler that selects preserved Driving Audio when present."""

    DEPRECATED = False
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Assemble externally decoded V3.4 chunks with guarded video seams. When "
        "Driving Audio is connected, generated-audio seam correction is bypassed and "
        "the preserved source stream is returned unchanged in content."
    )

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        optional = dict(schema.get("optional", {}))
        optional["driving_audio"] = ("AUDIO",)
        schema["optional"] = optional
        return schema

    def assemble(
        self,
        images,
        audio,
        assembly_plan,
        exact_total_duration,
        audio_seam,
        video_seam,
        diagnostics,
        driving_audio=None,
    ):
        driving = None
        if driving_audio is not None:
            driving = _singleton(driving_audio, "driving_audio")
        if driving is None:
            return super().assemble(
                images,
                audio,
                assembly_plan,
                exact_total_duration,
                audio_seam,
                video_seam,
                diagnostics,
            )

        result_images, _generated_audio, report = super().assemble(
            images,
            audio,
            assembly_plan,
            exact_total_duration,
            AUDIO_SEAM_OFF,
            video_seam,
            diagnostics,
        )
        waveform, sample_rate = validate_audio(driving)
        preserved_audio = {
            "waveform": waveform.detach().to("cpu").contiguous(),
            "sample_rate": int(sample_rate),
        }
        report = (
            report.rstrip()
            + "\nDriving Audio: preserved effective source selected for final output; "
            "generated Audio Seam processing bypassed."
        )
        return result_images, preserved_audio, report


NODE_CLASS_MAPPINGS = {
    "H3ContinuumSamplerV34": H3ContinuumSamplerV34,
    "H3ContinuumAssembleSeamV34": H3ContinuumAssembleSeamV34,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuumSamplerV34": "H3 Continuum Sampler V3.4",
    "H3ContinuumAssembleSeamV34": "H3 Continuum Assemble + Seam V3.4",
}
