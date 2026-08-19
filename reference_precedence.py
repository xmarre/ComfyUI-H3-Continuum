"""Compatibility wrappers enforcing Reference-mode frame precedence.

V3.4 defines Reference conditioning as authoritative whenever any Reference
Image input is present. Older saved production/timeline sampler node IDs remain
loadable, so they must obey the same rule instead of forwarding stale
first_frame/last_frame tensors alongside reference conditioning.
"""

from __future__ import annotations

from typing import Any

from .v3.nodes import H3ContinuumSamplerProduction, H3ContinuumSamplerTimelineVideo


def normalize_reference_frame_precedence(inputs: dict[str, Any]) -> dict[str, Any]:
    """Return a call copy with First/Last disabled whenever references are active."""

    normalized = dict(inputs)
    has_reference = any(
        normalized.get(f"reference_image_{index}") is not None
        for index in range(1, 9)
    )
    if has_reference:
        normalized["first_frame"] = None
        normalized["last_frame"] = None
    return normalized


class H3ContinuumSamplerProductionReferencePrecedence(H3ContinuumSamplerProduction):
    """Saved-production-node compatibility facade with Reference precedence."""

    def run(self, **kwargs):
        return super().run(**normalize_reference_frame_precedence(kwargs))


class H3ContinuumSamplerTimelineVideoReferencePrecedence(H3ContinuumSamplerTimelineVideo):
    """Saved-timeline-node compatibility facade with Reference precedence."""

    def run(self, **kwargs):
        return super().run(**normalize_reference_frame_precedence(kwargs))


def install_reference_frame_precedence(node_class_mappings: dict[str, Any]) -> None:
    """Replace legacy public sampler mappings without changing their node IDs."""

    if "H3ContinuumSamplerProduction" in node_class_mappings:
        node_class_mappings["H3ContinuumSamplerProduction"] = (
            H3ContinuumSamplerProductionReferencePrecedence
        )
    if "H3ContinuumSamplerTimelineVideo" in node_class_mappings:
        node_class_mappings["H3ContinuumSamplerTimelineVideo"] = (
            H3ContinuumSamplerTimelineVideoReferencePrecedence
        )
