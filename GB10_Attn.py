# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SOLRICKS
"""GB10-Attn v0.1.2 (single-file ComfyUI node).

Copy this file to ComfyUI/custom_nodes/GB10_Attn.py and restart.

This is a model-scoped *dispatcher*, not a new attention kernel. It uses the
native Sol-Attn kernel in ComfyUI / comfy-kitchen when available. It does not
copy or vendor third-party CUDA or Triton code. It preserves the existing
attention backend (e.g. SageAttention or PyTorch SDPA) for ineligible calls.

Hardware target: NVIDIA GB10 (SM121, CUDA / Linux aarch64). A separate option
permits other NVIDIA devices with compatible native Sol-Attn kernels.

Compatible with models using ComfyUI's generic attention override. For native
MiniMax H3 models, ComfyUI core's H3-specific sparse attention path is used.
Other models can use the generic patch where their attention calls support it.

IMPORTANT: Sparse attention is an approximation. Compare outputs to the dense
reference for identity, text, faces and temporally sensitive content. Do not
stack with another sparse/attention replacement that shadows this patch.

Requires a recent ComfyUI providing:
  comfy_extras.nodes_sparse_attention.apply_block_sparse_attention
  comfy_kitchen.sol_attn_is_available
and a comfy-kitchen build with a native Sol-Attn kernel for your device.
"""

from __future__ import annotations

import logging
import math

import torch

__version__ = "0.1.2"
_LOG = logging.getLogger("GB10Attn")
_MARKER = "_gb10_attn_v1"

# No steps, seeds, model weights, LoRAs, CFG or other non-attention math modified.
# A lower tau preserves more exact KV blocks and usually costs more compute.
_PROFILES = {
    "quality": {
        "tau": 0.90, "min_tokens": 16384, "start": 0.20, "end": 0.95,
        "extra_tokens": 256,
    },
    "balanced": {
        "tau": 1.30, "min_tokens": 12288, "start": 0.10, "end": 1.00,
        "extra_tokens": 256,
    },
    "speed": {
        "tau": 1.60, "min_tokens": 8192, "start": 0.00, "end": 1.00,
        "extra_tokens": 128,
    },
}


def _failure(model, reason: str, strict: bool):
    if strict:
        raise RuntimeError(f"GB10-Attn: {reason}")
    message = f"[GB10-Attn] Original model returned: {reason}"
    _LOG.warning(message)
    return model, message


def _device_info():
    """Return active ComfyUI torch device and hardware label; do not change it."""
    import comfy.model_management as mm

    device = mm.get_torch_device()
    if device.type != "cuda" or not torch.cuda.is_available():
        return device, None, str(device)
    capability = tuple(torch.cuda.get_device_capability(device))
    name = torch.cuda.get_device_name(device)
    return device, capability, f"{name} / SM{capability[0]}{capability[1]}"


class GB10Attn:
    """A per-model opt-in dispatcher to the ComfyUI core Sol-Attn kernels."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "profile": (["balanced", "quality", "speed", "dense_reference"], {
                    "default": "balanced",
                    "tooltip": "Quality is more conservative; speed is more approximate. "
                               "Dense reference returns the original model for A/B comparison.",
                }),
                "device_scope": (["GB10 only (SM121)", "Any supported NVIDIA GPU"], {
                    "default": "GB10 only (SM121)",
                    "tooltip": "Other GPUs work only if a native comfy-kitchen Sol-Attn kernel is available.",
                }),
                "strict": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Raise instead of pass through when the required kernel/API is unavailable. "
                               "Does not turn runtime Sol-Attn exceptions into silent fallback.",
                }),
            },
            "optional": {
                "min_tokens_override": ("INT", {
                    "default": 0, "min": 0, "max": 1048576, "step": 1024,
                    "tooltip": "0 = use profile threshold. Smaller sequences stay on the existing dense backend.",
                }),
                "tau_override": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": "0 = profile default. Higher tau means more approximation and usually more speed.",
                }),
                "dense_blocks": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Optional layer IDs to keep dense, e.g. 0,1,48-49. Applies if layer IDs are available.",
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Log eligible and rejected attention shapes through ComfyUI's native diagnostics.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    FUNCTION = "patch"
    CATEGORY = "SOLRICKS/Optimization"
    DESCRIPTION = (
        "GB10-Attn: single-file, model-scoped attention dispatcher using native ComfyUI/"
        "comfy-kitchen Sol-Attn. Automatically retains the existing dense backend "
        "on short/ineligible attention calls. Quality and speed profiles are approximate."
    )

    def patch(self, model, enabled=True, profile="balanced", device_scope="GB10 only (SM121)",
              strict=False, min_tokens_override=0, tau_override=0.0,
              dense_blocks="", verbose=False):
        if not enabled or profile == "dense_reference":
            return model, ("[GB10-Attn] Disabled / reference: input MODEL unchanged. "
                           "For a true dense baseline, connect an unpatched original MODEL.")

        if profile not in _PROFILES:
            return _failure(model, f"unknown profile: {profile}", strict)

        # Avoid stacking multiple copies of this node, which would add redundant
        # callbacks and could silently change the intended sparse schedule.
        try:
            options = model.model_options.get("transformer_options", {})
            if options.get(_MARKER) is not None:
                return _failure(model, "already patched by GB10-Attn; do not stack nodes", strict)
        except (AttributeError, TypeError):
            return _failure(model, "input is not a ComfyUI MODEL patcher", strict)

        try:
            device, capability, label = _device_info()
        except (ImportError, AttributeError, RuntimeError, AssertionError) as exc:
            return _failure(model, f"cannot inspect CUDA device: {exc}", strict)

        if capability is None:
            return _failure(model, f"CUDA GPU not active ({label})", strict)
        if device_scope == "GB10 only (SM121)" and capability != (12, 1):
            return _failure(model, f"GB10 SM121 required; detected {label}", strict)
        if device_scope not in ("GB10 only (SM121)", "Any supported NVIDIA GPU"):
            return _failure(model, f"unknown device scope: {device_scope}", strict)
        if getattr(torch.version, "hip", None):
            return _failure(model, "this node requires NVIDIA CUDA, not ROCm/HIP", strict)

        # Import lazily: this file must load cleanly even when a ComfyUI wheel is
        # old, or if comfy-kitchen was installed without a compatible CUDA build.
        try:
            import comfy_kitchen as ck
            from comfy_extras.nodes_sparse_attention import (
                apply_block_sparse_attention,
                parse_block_list,
            )
        except (ImportError, AttributeError) as exc:
            return _failure(model, f"update ComfyUI + comfy-kitchen (missing native sparse API: {exc})", strict)

        try:
            native = bool(ck.sol_attn_is_available(device))
        except (RuntimeError, AttributeError, TypeError) as exc:
            return _failure(model, f"Sol-Attn availability check failed: {exc}", strict)
        if not native:
            return _failure(model, f"no native comfy-kitchen Sol-Attn kernel for {label}", strict)

        cfg = _PROFILES[profile]
        min_tokens = int(min_tokens_override) if int(min_tokens_override) > 0 else cfg["min_tokens"]
        tau = float(tau_override) if float(tau_override) > 0 else cfg["tau"]
        if not math.isfinite(tau) or tau <= 0 or tau > 4.0:
            return _failure(model, "tau must be finite and between 0 and 4", strict)

        try:
            # Core creates a model clone and installs hooks on that clone only.
            # Its native implementation preserves H3 text/audio/ref conditioning
            # with exact-KV sink blocks and uses generic hooks for other models.
            patched = apply_block_sparse_attention(
                model,
                tau=tau,
                topk_ratio=0.0,               # Sol-Attn, not SLA/VSA-trained pattern
                vsa=False,
                start_percent=cfg["start"],
                end_percent=cfg["end"],
                min_tokens=min_tokens,
                dense_blocks=parse_block_list(dense_blocks),
                sink_conditioning="exact_kv_and_rows",
                extra_tokens=cfg["extra_tokens"],
                verbose=bool(verbose),
            )
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
            return _failure(model, f"core sparse patch could not be applied: {exc}", strict)

        patched.model_options.setdefault("transformer_options", {})[_MARKER] = {
            "version": __version__, "profile": profile, "tau": tau,
        }
        message = (
            f"[GB10-Attn] Active on {label}; profile={profile}, "
            f"native=ComfyUI Sol-Attn, tau={tau:g}, min_tokens={min_tokens}, "
            f"range={cfg['start']:.0%}-{cfg['end']:.0%}, "
            f"extra_tokens={cfg['extra_tokens']}. "
            "Fallback: existing dense backend for ineligible attention calls. "
            "This is an approximate attention mode, not a guaranteed speedup."
        )
        _LOG.info(message)
        print(message, flush=True)
        return patched, message


NODE_CLASS_MAPPINGS = {"GB10Attn": GB10Attn}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GB10Attn": "GB10-Attn",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
