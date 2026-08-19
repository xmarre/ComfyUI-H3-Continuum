"""Deterministic fingerprints for arbitrary ComfyUI upstream loader graphs."""

from __future__ import annotations

import hashlib
import json
from typing import Any


GRAPH_CONTRACT_VERSION = 3

# Legacy samplers keep the V3.3 fail-closed policy. V3.4 is deliberately
# Core-first: arbitrary wrapper classes are accepted when their complete route
# can be deterministically serialized, so new accelerator wrappers do not need
# a Continuum release just to resume stored chunks safely.
_KNOWN_MODEL_NODES = {
    "UNETLoader",
    "LoraLoader",
    "LoraLoaderModelOnly",
    "Power Lora Loader (rgthree)",
    "PathchSageAttentionKJ",
    "PatchSageAttentionKJ",
    "SpectrumApplyMiniMaxH3",
    "MiniMaxH3MemoryEfficientSageAttention",
    "MiniMaxH3MemoryEfficientSageAttentionPatch",
    "MiniMaxH3SigmaShift",
    "ModelSamplingMiniMaxH3",
}
_KNOWN_CLIP_NODES = {"CLIPLoader", "DualCLIPLoader", "TripleCLIPLoader"}
_KNOWN_VAE_NODES = {"VAELoader"}
_KNOWN_SHARED_NODES = {"Reroute", "Any Switch (rgthree)"}
_V34_SAMPLER_CLASSES = {"H3ContinuumSamplerV34"}

CHECKPOINT_REF2VA = "ref2va"
CHECKPOINT_FL2VA = "fl2va"
CHECKPOINT_UNKNOWN = "unknown"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _literal(value: Any) -> tuple[Any, bool]:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value, True
    if isinstance(value, (list, tuple)):
        result = []
        safe = True
        for item in value:
            rendered, item_safe = _literal(item)
            result.append(rendered)
            safe = safe and item_safe
        return result, safe
    if isinstance(value, dict):
        result = {}
        safe = True
        for key in sorted(value, key=str):
            rendered, item_safe = _literal(value[key])
            result[str(key)] = rendered
            safe = safe and item_safe
        return result, safe
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}, False


def _node(
    prompt: dict[str, Any],
    node_id: str,
    *,
    allowed: set[str] | None,
    stack: set[str],
) -> tuple[dict[str, Any], bool, list[str]]:
    if node_id in stack:
        return (
            {"node_id": node_id, "error": "cycle"},
            False,
            [f"upstream graph contains a cycle at node {node_id}"],
        )
    raw = prompt.get(str(node_id))
    if not isinstance(raw, dict):
        return (
            {"node_id": node_id, "error": "missing"},
            False,
            [f"upstream node {node_id} is missing"],
        )
    class_type = str(raw.get("class_type", ""))
    if allowed is None:
        safe = bool(class_type)
        reasons = [] if safe else [f"upstream node {node_id} has no class_type"]
    else:
        safe = class_type in allowed or class_type in _KNOWN_SHARED_NODES
        reasons = (
            []
            if safe
            else [f"unknown upstream node class: {class_type or '<missing>'}"]
        )
    inputs = raw.get("inputs") or {}
    if not isinstance(inputs, dict):
        return (
            {"node_id": node_id, "class_type": class_type, "error": "invalid inputs"},
            False,
            reasons + [f"upstream node {node_id} inputs are invalid"],
        )
    rendered_inputs: dict[str, Any] = {}
    next_stack = set(stack)
    next_stack.add(node_id)
    for name in sorted(inputs):
        value = inputs[name]
        if (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and isinstance(value[0], (str, int))
            and isinstance(value[1], int)
            and str(value[0]) in prompt
        ):
            child, child_safe, child_reasons = _node(
                prompt,
                str(value[0]),
                allowed=allowed,
                stack=next_stack,
            )
            rendered_inputs[str(name)] = {"output": int(value[1]), "node": child}
            safe = safe and child_safe
            reasons.extend(child_reasons)
        else:
            rendered, literal_safe = _literal(value)
            rendered_inputs[str(name)] = rendered
            safe = safe and literal_safe
            if not literal_safe:
                reasons.append(f"upstream node {node_id}.{name} is not serializable")
    return {
        "node_id": str(node_id),
        "class_type": class_type,
        "inputs": rendered_inputs,
    }, safe, reasons


def build_upstream_graph_contract(
    prompt: Any,
    unique_id: Any,
    *,
    require_video_vae: bool,
    require_reference_audio_vae: bool = False,
    require_audio_vae: bool = False,
) -> tuple[dict[str, Any], bool, list[str]]:
    """Fingerprint MODEL/CLIP/VAE routes feeding the Continuum sampler."""
    if not isinstance(prompt, dict):
        return (
            {"version": GRAPH_CONTRACT_VERSION},
            False,
            ["ComfyUI PROMPT graph is unavailable"],
        )
    node_id = str(unique_id)
    sampler_node = prompt.get(node_id)
    if not isinstance(sampler_node, dict):
        return (
            {"version": GRAPH_CONTRACT_VERSION},
            False,
            ["Continuum sampler node is absent from PROMPT graph"],
        )
    inputs = sampler_node.get("inputs") or {}
    routes: dict[str, Any] = {}
    safe = True
    reasons: list[str] = []
    sampler_class_type = str(sampler_node.get("class_type", ""))
    permissive = sampler_class_type in _V34_SAMPLER_CLASSES
    specs: list[tuple[str, set[str] | None, bool]] = [
        ("model", None if permissive else _KNOWN_MODEL_NODES, True),
        ("clip", None if permissive else _KNOWN_CLIP_NODES, True),
        ("video_vae", None if permissive else _KNOWN_VAE_NODES, bool(require_video_vae)),
    ]
    if require_reference_audio_vae:
        specs.append(
            ("reference_audio_vae", None if permissive else _KNOWN_VAE_NODES, True)
        )
    if require_audio_vae:
        specs.append(("audio_vae", None if permissive else _KNOWN_VAE_NODES, True))
    for name, allowed, required in specs:
        if not required:
            continue
        link = inputs.get(name)
        if not (
            isinstance(link, (list, tuple))
            and len(link) == 2
            and isinstance(link[0], (str, int))
            and isinstance(link[1], int)
        ):
            routes[name] = {"error": "input is not connected"}
            safe = False
            reasons.append(f"{name} upstream connection is unavailable")
            continue
        route, route_safe, route_reasons = _node(
            prompt,
            str(link[0]),
            allowed=allowed,
            stack={node_id},
        )
        routes[name] = {"output": int(link[1]), "node": route}
        safe = safe and route_safe
        reasons.extend(route_reasons)
    descriptor = {
        "version": GRAPH_CONTRACT_VERSION,
        "sampler_class_type": sampler_class_type,
        "routes": routes,
    }
    descriptor["sha256"] = _sha256(descriptor)
    return descriptor, safe, list(dict.fromkeys(reasons))


def classify_h3_checkpoint(prompt: Any, unique_id: Any) -> str:
    """Classify the base H3 UNET connected to the sampler, without guessing."""
    if not isinstance(prompt, dict):
        return CHECKPOINT_UNKNOWN
    sampler_node = prompt.get(str(unique_id))
    if not isinstance(sampler_node, dict):
        return CHECKPOINT_UNKNOWN
    model_link = (sampler_node.get("inputs") or {}).get("model")
    if not (
        isinstance(model_link, (list, tuple))
        and len(model_link) == 2
        and isinstance(model_link[0], (str, int))
    ):
        return CHECKPOINT_UNKNOWN

    stack = [str(model_link[0])]
    visited: set[str] = set()
    classifications: set[str] = set()
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        node = prompt.get(node_id)
        if not isinstance(node, dict):
            return CHECKPOINT_UNKNOWN
        inputs = node.get("inputs") or {}
        if str(node.get("class_type", "")) == "UNETLoader":
            loader_text = " ".join(
                str(value).lower()
                for value in inputs.values()
                if isinstance(value, str)
            )
            if "ref2va" in loader_text or "ref2v" in loader_text:
                classifications.add(CHECKPOINT_REF2VA)
            elif "fl2va" in loader_text or "fl2v" in loader_text:
                classifications.add(CHECKPOINT_FL2VA)
            else:
                classifications.add(CHECKPOINT_UNKNOWN)
            continue
        for value in inputs.values():
            if (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and isinstance(value[0], (str, int))
                and str(value[0]) in prompt
            ):
                stack.append(str(value[0]))
    if classifications == {CHECKPOINT_REF2VA}:
        return CHECKPOINT_REF2VA
    if classifications == {CHECKPOINT_FL2VA}:
        return CHECKPOINT_FL2VA
    return CHECKPOINT_UNKNOWN
