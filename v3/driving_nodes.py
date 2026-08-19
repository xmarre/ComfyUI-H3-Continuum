"""Stable V3.4 public facade with V3.4 conditioning precedence."""

from __future__ import annotations

from .driving_nodes_impl import (
    H3ContinuumAssembleSeamV34,
    H3ContinuumSamplerV34 as _H3ContinuumSamplerV34,
    _prompt_graph_with_audio_vae_alias,
    _reference_video_storage_contract,
)


class H3ContinuumSamplerV34(_H3ContinuumSamplerV34):
    """Apply V3.4's Reference-mode precedence before entering the core engine."""

    def run(self, *args, **kwargs):
        has_reference = any(
            kwargs.get(f"reference_image_{index}") is not None
            for index in range(1, 9)
        )
        if has_reference:
            # V3.4's conditioning contract resolves any image-reference presence
            # to Reference mode. First/Last sockets may remain connected in a
            # saved workflow, but they are not active keyframes in that mode.
            kwargs["first_frame"] = None
            kwargs["last_frame"] = None
        return super().run(*args, **kwargs)


NODE_CLASS_MAPPINGS = {
    "H3ContinuumSamplerV34": H3ContinuumSamplerV34,
    "H3ContinuumAssembleSeamV34": H3ContinuumAssembleSeamV34,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuumSamplerV34": "H3 Continuum Sampler V3.4",
    "H3ContinuumAssembleSeamV34": "H3 Continuum Assemble + Seam V3.4",
}
