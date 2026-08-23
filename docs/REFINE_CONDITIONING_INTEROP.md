# Exact per-chunk state for latent refinement

H3 Continuum V3.4 does not sample a chunk from video/audio tensors alone. The sampler also receives chunk-specific MiniMax H3 conditioning, a chunk-local MODEL clone with Continuum's APPLY_MODEL wrapper/context hint, and, for Native Masked continuation, an exact AV denoise mask.

A downstream learned-latent refinement pass must therefore keep all of those channels aligned with the chunk it refines. Reconstructing only a generic positive conditioning after Continuum is insufficient.

## V3.4 output contract

`H3 Continuum Sampler V3.4` appends one sixth list output without changing the existing first five indices:

1. `video_latents` — split 24-channel video LATENT chunks
2. `audio_latents` — split 32-channel audio LATENT chunks
3. `assembly_plan`
4. `status`
5. `driving_audio`
6. `refine_state` — one `H3_CONTINUUM_REFINE_STATE` object per sampled chunk

When `refine_state` is connected, Continuum captures the exact state at the internal `sample_chunk()` boundary. Each state contains:

- a fresh MODEL clone preserving the same chunk-specific Continuum model options while installing a fresh APPLY_MODEL wrapper closure for sampler 2;
- the exact positive CONDITIONING object that sampler 1 received.

If sampler 1 used a denoise mask, V3.4 also restores its exact video/audio members onto the corresponding split `video_latents` / `audio_latents` entries. This is required for Native Masked continuation so sampler 2 cannot accidentally re-denoise the protected prefix.

The frontend enables the hidden capture switch only when `refine_state` is actually linked. Normal V3.4 workflows therefore retain the previous output/memory behavior.

## Correct wiring

With the companion learned-upscaler PR, the completed path is:

```text
H3 Continuum Sampler V3.4
  video_latents -----> MiniMax H3 Latent Upscaler + Refine (3D).latent
  audio_latents -----> MiniMax H3 Latent Upscaler + Refine (3D).audio_latent
  refine_state ------> MiniMax H3 Latent Upscaler + Refine (3D).refine_state

RandomNoise ---------> MiniMax H3 Latent Upscaler + Refine (3D).noise
KSamplerSelect ------> MiniMax H3 Latent Upscaler + Refine (3D).sampler
low-sigma SIGMAS ----> MiniMax H3 Latent Upscaler + Refine (3D).sigmas

MiniMax H3 Latent Upscaler + Refine (3D).latent
  -------------------> VAE Decode / downstream assembly
```

There is no external BasicGuider, DisableNoise, or SamplerCustomAdvanced. The upscaler/refine node performs the actual second H3 sampling pass internally.

The three Continuum list outputs are parallel and map by chunk index, so chunk N's video, audio, model wrapper, conditioning, and mask remain aligned.

## Natural-timeline image refinement

Image-space refinement that tracks across the complete result must retain every physical-group
frame until the refined images have been stitched back. On **H3 Continuum Assemble + Seam V3.4**,
select `Timeline Output = Natural retained timeline (Refinement)`. This is an explicit opt-in;
the default remains `Exact requested duration (Recommended)`.

After downstream refinement/stitching, connect the natural IMAGE batch, the assembler AUDIO, and
the same `assembly_plan` to **H3 Continuum Finalize Duration V3.4**. The finalizer requires the
plan's exact natural frame count and applies the existing `enforce_total_frames` policy once:

- trim to `target_frames`;
- preserve the final anchored frame when `preserve_final_frame` is set;
- trim/pad audio at the corresponding sample boundary, including final-anchor audio handling.

An already compacted IMAGE input is rejected because its removed physical-group frames cannot be
reconstructed after tracking or refinement.

## Why MODEL state is included

Guide / Motion Context and hybrid First/Last + Reference workflows rely on Continuum's per-MODEL APPLY_MODEL wrapper, not only CONDITIONING. The wrapper performs Continuum layout/RoPE adaptation and mixed keyframe/reference normalization. Reusing an unrelated raw H3 MODEL for sampler 2 is therefore not a complete reproduction of sampler 1 semantics.

The refine state does not reuse the already-executed wrapper closure directly. It clones the exact chunk MODEL after sampling and reinstalls a fresh Continuum wrapper closure while preserving the chunk's model options/context hint.

## Why masks are restored onto split LATENTs

Native Masked continuation protects the previous generated prefix with Core-native AV denoise masks. V3's normal public split LATENT facade historically exposed only the sampled tensors. For refinement interop, capture restores the video and audio mask members to their matching split LATENT dictionaries before downstream mapping.

The learned-upscaler node then nearest-resizes only the target video mask to the enlarged grid and preserves the audio mask. With `lock_audio=True`, audio is additionally masked and restored exactly after refinement.

## Run Storage and reused chunks

Raw CONDITIONING/model runtime state is intentionally not persisted in Run Storage sessions. Therefore exact `refine_state` can only be emitted when every output chunk was sampled in the current execution.

If Run Storage reused a prefix while refinement capture is requested, V3.4 raises rather than pairing a newly captured suffix with reused latent chunks. Use either:

- `Run Storage = Off`; or
- `Run Storage = Save + Auto Resume` with `Regenerate From = Chunk 1` for that refinement run.

A fresh Save + Auto Resume run with no reusable prefix is valid.

## Memory behavior

Capture is output-link driven. With `refine_state` unused, no extra MODEL/conditioning/mask state is retained. When it is used, MODEL clones share model weights through ComfyUI's normal ModelPatcher clone semantics; masks are copied to CPU before being attached to the split LATENT outputs.
