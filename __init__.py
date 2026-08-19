"""H3 Continuum — native MiniMax H3 chunk continuation."""

from __future__ import annotations

import logging

WEB_DIRECTORY = "./web"

# ComfyUI loads custom-node folders as packages. Standalone test collection may
# import this file as a top-level module; keep that path inert because ComfyUI is
# intentionally not a unit-test dependency.
if __package__:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from .reference_precedence import install_reference_frame_precedence
    from .temporal import run_temporal_self_test
    from .version import PACKAGE_VERSION

    # Saved V3.2/V3.3 production node IDs remain loadable. Keep them aligned
    # with V3.4's Reference-mode precedence so stale First/Last connections
    # cannot leak into reference-conditioned generations.
    install_reference_frame_precedence(NODE_CLASS_MAPPINGS)

    try:
        run_temporal_self_test()
    except Exception as exc:
        raise RuntimeError(f"H3 Continuum Join self-test failed: {exc}") from exc
    logging.getLogger("h3_continuum_join").info(
        "H3 Continuum %s loaded (V2 integrated sampler + hidden legacy workflow compatibility)",
        PACKAGE_VERSION,
    )
else:  # pragma: no cover
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
