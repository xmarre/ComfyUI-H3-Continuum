"""Optional structural interop with MiniMax H3 RefDelta reference diagnostics.

Continuum deliberately has no import-time dependency on the RefDelta Solver.
The solver places a small opaque specification in MODEL.model_options; this
module validates that public shape and lets the runtime consume it fail-closed.
"""

from __future__ import annotations

from typing import Any


REFERENCE_DIAGNOSTIC_MODEL_OPTION = "refdelta_reference_diagnostic"
REFERENCE_DIAGNOSTIC_CONTRACT = (
    "comfyui-refdelta-reference-diagnostic",
    1,
    "model-option-guider-mixin",
)


class RefDeltaDiagnosticInteropError(RuntimeError):
    pass


def reference_diagnostic_from_model(model: Any):
    """Return a validated diagnostic spec attached to ``model``, or ``None``."""
    model_options = getattr(model, "model_options", None) or {}
    spec = model_options.get(REFERENCE_DIAGNOSTIC_MODEL_OPTION)
    if spec is None:
        return None
    if getattr(spec, "contract", None) != REFERENCE_DIAGNOSTIC_CONTRACT:
        raise RefDeltaDiagnosticInteropError(
            "unsupported RefDelta reference diagnostic contract"
        )
    reference_model = getattr(spec, "reference_model", None)
    guider_mixin = getattr(spec, "guider_mixin", None)
    replace_reference = getattr(spec, "with_reference_model", None)
    if reference_model is None or not hasattr(reference_model, "clone"):
        raise RefDeltaDiagnosticInteropError(
            "RefDelta reference diagnostic does not expose a cloneable reference MODEL"
        )
    if not isinstance(guider_mixin, type):
        raise RefDeltaDiagnosticInteropError(
            "RefDelta reference diagnostic does not expose a guider mixin type"
        )
    if not callable(replace_reference):
        raise RefDeltaDiagnosticInteropError(
            "RefDelta reference diagnostic cannot replace its call-local reference MODEL"
        )
    return spec


def replace_reference_diagnostic_model(spec: Any, reference_model: Any):
    """Replace only the reference MODEL while preserving the validated contract."""
    original_mixin = getattr(spec, "guider_mixin", None)
    replacement = spec.with_reference_model(reference_model)
    if getattr(replacement, "contract", None) != REFERENCE_DIAGNOSTIC_CONTRACT:
        raise RefDeltaDiagnosticInteropError(
            "RefDelta reference diagnostic replacement changed its contract"
        )
    if getattr(replacement, "reference_model", None) is not reference_model:
        raise RefDeltaDiagnosticInteropError(
            "RefDelta reference diagnostic replacement rejected the call-local reference MODEL"
        )
    if getattr(replacement, "guider_mixin", None) is not original_mixin:
        raise RefDeltaDiagnosticInteropError(
            "RefDelta reference diagnostic replacement changed its guider mixin"
        )
    return replacement


__all__ = [
    "REFERENCE_DIAGNOSTIC_CONTRACT",
    "REFERENCE_DIAGNOSTIC_MODEL_OPTION",
    "RefDeltaDiagnosticInteropError",
    "reference_diagnostic_from_model",
    "replace_reference_diagnostic_model",
]
