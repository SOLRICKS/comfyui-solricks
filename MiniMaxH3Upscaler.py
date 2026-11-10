# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SOLRICKS
"""MiniMax H3 Upscaler by SOLRICKS, version 0.2.6.

This is a new implementation of the SOLRICKS node. It does not bundle code
from another custom-node project. ComfyUI's own H3 loader, latent upscaler,
BasicGuider and sampler APIs supply the model implementations at runtime.
The file is distributed under comfyui-solricks' existing MIT license.

Requirements
------------
Use a current ComfyUI with MiniMaxH3LatentUpscaleWithModel and an H3-capable
LatentUpscaleModelLoader. No additional upscaler custom-node pack is needed.
Existing model files are reused, without training:

  models/latent_upscale_models/
      minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors (or bf16)
  models/loras/h3_upscale_refiner_bf16.safetensors
  (the original open_h3_regenerate_2k_r256pp_bf16.safetensors is also supported)

Connect an H3 FL2VA MODEL without generation LoRAs, an H3 video VAE, and the
source IMAGE batch. Source audio, source FPS, H3 audio VAE and text-only H3
conditioning are optional. Audio output is the exact input AUDIO object,
or silence matching the number of output frames and the supplied FPS.
Model weights and ComfyUI keep their own licenses; this file does not
relicense, redistribute or claim authorship of those dependencies.

Implementation
--------------
Frames are fitted to H3's 32-pixel grid, padded to its native time layout,
encoded, enlarged through core, refined through core, decoded, and trimmed.
With source detail protection enabled, the last step compares each decoded
frame with the corresponding input in RGB. A float Lanczos down/up pass
estimates the part of the H3 change already resolved on the input grid.
Subtracting it keeps the source texture while retaining finer H3 detail.
This operates on one frame at a time. An exact size match passes through
the source RGB directly when protection is enabled.
Refinement uses a seeded clip-wide noise field and normalized overlap-add.
"stepped" updates overlapping time/space windows together at every sigma;
"crossfade" refines time windows separately, with joint spatial steps inside
each window. A smaller token budget is retried after a recognized OOM.
This is a distinct implementation; bit-identical output to other plugins
is not claimed. There is no auxiliary face-detector/refinement pass.

This IMAGE node holds source and output frame batches in RAM. Chunking
controls model work, not the storage required by the IMAGE type.

Performance changes keep model precision, sigma schedules, noise seeds,
window geometry and blend arithmetic unchanged. Sampler/preview setup is
reused during a run; unused x0 previews are not retained. Host-side zero
noise and blend workspaces are reused, and decoder context is cropped
before transfer/casting to avoid extra full-resolution temporary frames.

Controls
--------
Choose the upscaler model and refiner LoRA from their selection fields.
The other widgets are megapixels, enable temporal chunking, enable spatial
tiling and preserve source detail. The last control defaults to on. Turn
it off to use the original H3 output, including same-size redraws.
Internal refinement defaults are listed below the imports.
Sparse attention is preferred automatically. Missing/unsupported sparse
support falls back to a clean dense model, including at sampling time.
OOM still follows the smaller-window retry path; unrelated errors propagate.
Only IMAGE and AUDIO are returned. A node-scoped ComfyUI progress bar tracks
the stages and individual refinement steps; sampler terminal bars are disabled.
Cancellation is checked between CPU batches, model calls and sampling steps.
An active GPU kernel/VAE call must return before Python can handle cancellation.
Console output is limited to start/end, cancellation and important fallbacks.
Repeated ComfyUI model-management INFO messages are filtered from console
handlers only while this node runs. The first model message, warnings/errors,
file logs and messages from other execution contexts remain available.

Installation
------------
Place MiniMaxH3Upscaler.py in custom_nodes/comfyui-solricks, and add these
lines to that package's __init__.py before its __all__ assignment:

    try:
        from .MiniMaxH3Upscaler import (
            NODE_CLASS_MAPPINGS as H3_UPSCALER_MAPPINGS,
            NODE_DISPLAY_NAME_MAPPINGS as H3_UPSCALER_NAMES,
        )
        NODE_CLASS_MAPPINGS.update(H3_UPSCALER_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(H3_UPSCALER_NAMES)
    except Exception as e:
        print(f"[SOLRICKS] MiniMaxH3Upscaler failed to load: {type(e).__name__}: {e}")

Restart ComfyUI and search for MiniMax H3 Upscaler.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import logging
import math
import os
import threading
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

__version__ = "0.2.6"
_MODEL_FOLDER = "latent_upscale_models"
_UPSCALE_FILE = "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"
_LORA_FILE = "h3_upscale_refiner_bf16.safetensors"
_LOG = "[SOLRICKS MiniMax H3 Upscaler]"
_LOG_SCOPE = ContextVar("solricks_h3_log_scope", default=None)

# Internal settings: fixed defaults keep the node's user interface compact.
_DENOISE = 0.8
_TILE_TOKENS = 0
_SAMPLER = "euler"
_SEED = 42
_ATTENTION = "sparse"
_CHUNK_JOIN = "stepped"
_REDRAW_START = 0.6
_REDRAW_STEPS = 2


class _ModelInfoFilter(logging.Filter):
    """Deduplicate routine model messages for one execution and one handler."""

    def __init__(self, scope):
        super().__init__()
        self.scope = scope
        self.seen = set()
        self.lock = threading.Lock()

    def filter(self, record):
        if record.levelno != logging.INFO or _LOG_SCOPE.get() is not self.scope:
            return True
        message = record.getMessage().strip()
        if message == "0 models unloaded.":
            return False
        routine = (message.startswith(("Requested to load ", "Model storage policy:"))
                   or (message.startswith("Model ")
                       and " prepared for dynamic VRAM loading." in message))
        if not routine:
            return True
        key = (record.name, message)
        with self.lock:
            if key in self.seen:
                return False
            self.seen.add(key)
        return True


class _QuietModelLogs:
    def __enter__(self):
        self.token = _LOG_SCOPE.set(self)
        self.filters = []
        loggers = [logging.getLogger(), *list(logging.Logger.manager.loggerDict.values())]
        handlers = {handler for logger in loggers if isinstance(logger, logging.Logger)
                    for handler in logger.handlers
                    if isinstance(handler, logging.StreamHandler)
                    and not isinstance(handler, logging.FileHandler)}
        try:
            for handler in handlers:
                message_filter = _ModelInfoFilter(self)
                handler.addFilter(message_filter)
                self.filters.append((handler, message_filter))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exception):
        try:
            for handler, message_filter in self.filters:
                handler.removeFilter(message_filter)
        finally:
            _LOG_SCOPE.reset(self.token)
        return False


def _first(result):
    """Unpack a public V3 NodeOutput, also accepting V1 tuple results."""
    return result.args[0] if hasattr(result, "args") else result[0]


def _model_names():
    import folder_paths

    if _MODEL_FOLDER not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(
            _MODEL_FOLDER, os.path.join(folder_paths.models_dir, _MODEL_FOLDER))
    roots, suffixes = folder_paths.folder_names_and_paths[_MODEL_FOLDER]
    folder_paths.folder_names_and_paths[_MODEL_FOLDER] = (
        roots, set(suffixes).union({".safetensors"}))
    files = [name for name in folder_paths.get_filename_list(_MODEL_FOLDER)
             if name.lower().endswith(".safetensors")
             and "h3" in name.lower() and "upscal" in name.lower()]
    files.sort(key=lambda name: (os.path.basename(name) != _UPSCALE_FILE, name))
    return files or [_UPSCALE_FILE]


def _frame_boundary(token):
    """H3's public time layout: one 1-frame token followed by four 4-frame tokens."""
    groups, remainder = divmod(int(token), 5)
    return groups * 17 + (0, 1, 5, 9, 13)[remainder]


def _canvas(width, height, megapixels):
    source = tuple(max(32, round(side / 32) * 32) for side in (width, height))
    factor = math.sqrt(float(megapixels) * 1048576 / (source[0] * source[1]))
    target = tuple(max(32, round(side * factor / 32) * 32) for side in source)
    if any(out < inp for inp, out in zip(source, target)):
        raise ValueError(
            f"H3 needs an upscale factor >= 1. Source canvas is {source[0]}x{source[1]} "
            f"({source[0] * source[1] / 1048576:.3f} MP); increase megapixels.")
    return source, target


def _fit_input(images, canvas, padded_frames, runtime=None):
    if runtime is not None:
        runtime.check_cancel()
    width, height = canvas
    n = len(images)
    if images.shape[1:3] == (height, width) and padded_frames == n:
        if runtime is not None:
            runtime.progress.update(5)
        return images[..., :3]
    fitted = torch.empty((padded_frames, height, width, 3), dtype=torch.float32, device="cpu")
    for left in range(0, n, 8):
        if runtime is not None:
            runtime.check_cancel()
        piece = images[left:left + 8, ..., :3].detach().to("cpu", torch.float32)
        if piece.shape[1:3] != (height, width):
            piece = F.interpolate(piece.permute(0, 3, 1, 2), (height, width),
                                  mode="bicubic", align_corners=False,
                                  antialias=True).permute(0, 2, 3, 1).clamp_(0, 1)
        fitted[left:left + len(piece)].copy_(piece)
        if runtime is not None:
            runtime.progress.section(0, 5, left + len(piece), n)
    if padded_frames > n:
        fitted[n:].copy_(fitted[n - 1:n].expand(padded_frames - n, -1, -1, -1))
    if runtime is not None:
        runtime.progress.update(5)
    return fitted


def _prompt_only(conditioning):
    if conditioning is None or len(conditioning) == 0:
        return [[torch.zeros(1, 1, 5120),
                 {"minimax_token_tags": torch.ones(1, dtype=torch.int64)}]]
    forbidden = {"minimax_refs", "minimax_keyframes", "minimax_visual_cond_noise_aug",
                 "minimax_audio_cond_noise_aug", "control", "concat_latent_image", "concat_mask"}
    return [[embedding, {key: value for key, value in metadata.items() if key not in forbidden}]
            for embedding, metadata in conditioning]


def _refine_schedule(start, steps):
    """Even flow-time intervals under the refiner's shift-6 noise mapping."""
    raw_start = float(start) / (6.0 - 5.0 * float(start))
    unshifted = torch.linspace(raw_start, 0.0, int(steps) + 1, dtype=torch.float32)
    sigmas = (unshifted * 6).div_(unshifted * 5 + 1)
    sigmas[0], sigmas[-1] = float(start), 0.0
    return sigmas


def _overlapping_intervals(total, span, overlap, quantum=1):
    """Cover a dimension, keeping starts aligned and avoiding very short final pieces."""
    if total <= span:
        return [(0, total)]
    stride = max(quantum, ((span - overlap) // quantum) * quantum)
    starts = list(range(0, total - span + 1, stride))
    final = ((total - span) // quantum) * quantum
    if final > starts[-1]:
        starts.append(final)
    intervals = [(start, min(start + span, total)) for start in starts]
    intervals[-1] = (intervals[-1][0], total)
    return intervals


def _blend_axis(length):
    # A strictly positive triangular window has no uncovered/zero-weight edges.
    return 1.0 - torch.linspace(-0.95, 0.95, int(length), dtype=torch.float32).abs()


@dataclass(frozen=True)
class _Region:
    begin: int
    end: int
    top: int
    bottom: int
    left: int
    right: int

    @property
    def index(self):
        return (slice(None), slice(None), slice(self.begin, self.end),
                slice(self.top, self.bottom), slice(self.left, self.right))

    def weight(self):
        return (_blend_axis(self.end - self.begin)[None, None, :, None, None]
                * _blend_axis(self.bottom - self.top)[None, None, None, :, None]
                * _blend_axis(self.right - self.left)[None, None, None, None, :])


def _space_regions(begin, end, height, width, budget, extra, tiling):
    cells = max(16, (int(budget) - int(extra)) // max(1, end - begin)) * 4
    if not tiling or height * width <= cells:
        return [_Region(begin, end, 0, height, 0, width)]
    ratio = width / height
    tile_h = min(height, max(16, 2 * int(math.sqrt(cells / ratio) / 2)))
    tile_w = min(width, max(16, 2 * int(math.sqrt(cells * ratio) / 2)))
    while tile_h * tile_w > cells and max(tile_h, tile_w) > 16:
        if tile_w >= tile_h and tile_w > 16:
            tile_w -= 2
        elif tile_h > 16:
            tile_h -= 2
        else:
            break
    ys = _overlapping_intervals(height, tile_h, min(8, tile_h // 2), quantum=2)
    xs = _overlapping_intervals(width, tile_w, min(8, tile_w // 2), quantum=2)
    return [_Region(begin, end, y0, y1, x0, x1) for y0, y1 in ys for x0, x1 in xs]


def _time_regions(tokens, height, width, budget, chunking):
    if not chunking:
        return [(0, tokens)]
    tokens_per_frame = max(1, (height // 2) * (width // 2))
    span = max(20, min(40, 5 * (int(budget) // tokens_per_frame // 5)))
    return _overlapping_intervals(tokens, span, 10, quantum=5)


def _out_of_memory(error):
    if isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError)):
        return True
    message = str(error).lower()
    return "out of memory" in message or "cuda_error_out_of_memory" in message


class _NodeProgress:
    """One UI progress bar; no terminal output or process-wide progress changes."""

    def __init__(self, check_cancel, node_id=None):
        from comfy.utils import ProgressBar

        self.check_cancel = check_cancel
        self.bar = ProgressBar(1000, node_id=node_id)
        self.value = -1
        self.update(0)

    def update(self, percent, preview=None):
        self.check_cancel()
        value = max(self.value, min(1000, max(0, round(percent * 10))))
        if value != self.value or preview is not None:
            self.value = value
            self.bar.update_absolute(value, 1000, preview)

    def section(self, begin, end, done, total, preview=None):
        self.update(begin + (end - begin) * min(1.0, done / max(1, total)), preview)


class _Runtime:
    """ComfyUI adapters; no imported custom-node implementations or vendored networks."""

    def __init__(self, node_id=None):
        import comfy.model_management as management
        from comfy_extras.nodes_minimax_h3 import temporal_shape, MiniMaxH3LatentUpscaleWithModel
        from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader

        self.mm = management
        self.clock = temporal_shape
        self.upscale_node = MiniMaxH3LatentUpscaleWithModel
        self.loader = LatentUpscaleModelLoader
        self.progress = _NodeProgress(self.check_cancel, node_id)
        self._notices = set()
        self._refine_done = 0
        self._refine_total = 1
        self._clear_sampler()

    def _clear_sampler(self):
        # This cache lasts only for one model stage. No GPU masks or x0
        # tensors are kept between calls, or after the stage is offloaded.
        self._guider = None
        self._sampler_key = None
        self._previewer = None
        self._previewer_ready = False
        self._mask_key = None
        self._mask_values = None
        self._zero_video = None

    def release(self):
        self._clear_sampler()
        self.mm.unload_all_models()
        self.mm.soft_empty_cache()

    def check_cancel(self):
        self.mm.throw_exception_if_processing_interrupted()

    def notice(self, key, message):
        if key not in self._notices:
            self._notices.add(key)
            print(f"{_LOG} {message}", flush=True)

    def begin_refinement(self, total):
        self.check_cancel()
        self._refine_done = 0
        self._refine_total = max(1, total)
        self.progress.update(45)

    def budget(self, requested):
        if requested > 0:
            return int(requested)
        device = self.mm.get_torch_device()
        if device.type != "cuda":
            return 16384
        gigabytes = torch.cuda.get_device_properties(device).total_memory / 1073741824
        return max(12288, min(65536, int(gigabytes * 2048) // 1024 * 1024))

    def enlarge(self, video, filename, target, allow_chunks):
        from comfy.nested_tensor import NestedTensor

        self.check_cancel()
        if (video.shape[4] * 16, video.shape[3] * 16) == target:
            self.progress.update(40)
            return video, "unchanged latent dimensions"
        self.release()
        self.check_cancel()
        patcher = _first(self.loader.execute(filename))
        self.check_cancel()

        def call(part):
            self.check_cancel()
            # The spatial-upscale core node passes this dummy audio through.
            payload = {"samples": NestedTensor((part, torch.zeros(1, 32, 2, 1)))}
            result = _first(self.upscale_node.execute(patcher, payload, *target))
            self.check_cancel()
            return result["samples"].tensors[0].cpu().float()

        try:
            result = call(video)
            self.progress.update(40)
            return result, "core latent upscale: whole clip"
        except Exception as error:
            if not _out_of_memory(error) or not allow_chunks:
                raise
            self.check_cancel()
            self.notice("upscale_memory", "VRAM limit reached; splitting latent upscale into time sections.")
        finally:
            self.release()

        for span in (30, 15, 5):
            self.check_cancel()
            if video.shape[2] <= span:
                continue
            intervals = _overlapping_intervals(video.shape[2], span, min(5, span - 1), quantum=5)
            accum = torch.zeros((1, 24, video.shape[2], target[1] // 16, target[0] // 16))
            norm = torch.zeros(1, 1, video.shape[2], 1, 1)
            try:
                for index, (begin, end) in enumerate(intervals, 1):
                    part = call(video[:, :, begin:end].contiguous())
                    weight = _blend_axis(end - begin)[None, None, :, None, None]
                    accum[:, :, begin:end].add_(part * weight)
                    norm[:, :, begin:end].add_(weight)
                    self.progress.section(25, 40, index, len(intervals))
                return accum.div_(norm), f"core latent upscale: {len(intervals)} overlapping sections"
            except Exception as error:
                if not _out_of_memory(error):
                    raise
            finally:
                self.release()
        raise RuntimeError("H3 latent upscale could not fit even a short section in memory.")

    def audio_latent(self, audio, audio_vae, wanted, fps):
        self.check_cancel()
        if audio is None or audio_vae is None:
            return torch.zeros(1, 32, 2, wanted), "silent conditioning"
        import comfy.audio

        source_rate = int(audio["sample_rate"])
        vae_rate = int(getattr(audio_vae, "audio_sample_rate", 44100))
        waveform = audio["waveform"][:1]
        # Only conditioning is retimed to H3's 24-fps clock. Output stays original.
        conditioning_rate = max(1, round(vae_rate * fps / 24))
        if source_rate != conditioning_rate:
            waveform = comfy.audio.resample(waveform, source_rate, conditioning_rate)
        self.check_cancel()
        encoded = audio_vae.encode(waveform.movedim(1, -1)).cpu().float()
        self.check_cancel()
        if encoded.ndim != 4 or encoded.shape[:3] != (1, 32, 2):
            raise ValueError("audio_vae must be an H3 audio VAE ([1,32,2,time] output).")
        fitted = torch.zeros(1, 32, 2, wanted)
        count = min(wanted, encoded.shape[-1])
        fitted[..., :count] = encoded[..., :count]
        return fitted, f"soundtrack conditioning at {fps:g} fps"

    def sampler_call(self, model, prompt, sampler, sigma, video, audio, noise, seed):
        from comfy.nested_tensor import NestedTensor
        from comfy_extras.nodes_custom_sampler import BasicGuider
        import latent_preview

        self.check_cancel()
        if (self._sampler_key is None or self._sampler_key[0] is not model
                or self._sampler_key[1] is not prompt):
            self._clear_sampler()
            self._guider = _first(BasicGuider.execute(model, prompt))
            self._sampler_key = (model, prompt)
        guider = self._guider
        mask_key = (tuple(video.shape), video.dtype, tuple(audio.shape), audio.dtype)
        if mask_key != self._mask_key:
            self._mask_key = mask_key
            self._mask_values = (
                torch.ones(1, 1, 1, video.shape[3], video.shape[4], device="cpu"),
                torch.zeros_like(audio, device="cpu"))
            self._zero_video = None
        video_mask, audio_zero = self._mask_values
        if noise is None:
            if self._zero_video is None:
                self._zero_video = torch.zeros_like(video, device="cpu")
            noise = self._zero_video
        samples = NestedTensor((video, audio))
        noise_pair = NestedTensor((noise, audio_zero))
        # ComfyUI replaces entries of the mask's tensor list while preparing
        # GPU masks. Always pass a fresh wrapper around the cached CPU tensors.
        mask = NestedTensor((video_mask, audio_zero))
        if not self._previewer_ready:
            patcher = guider.model_patcher
            self._previewer = latent_preview.get_previewer(
                patcher.load_device, patcher.model.latent_format)
            self._previewer_ready = True
        job_steps = len(sigma) - 1
        completed = self._refine_done

        def callback(step, x0, x, total_steps):
            self.check_cancel()
            preview = None
            if self._previewer is not None:
                if getattr(x0, "is_nested", False):
                    x0 = x0.tensors[0]
                preview = self._previewer.decode_latent_to_preview_image("JPEG", x0)
            self.progress.section(45, 90, completed + min(step + 1, job_steps),
                                  self._refine_total, preview)

        output = guider.sample(noise_pair, samples, sampler, sigma,
                               denoise_mask=mask, callback=callback,
                               disable_pbar=True, seed=seed)
        self.check_cancel()
        self._refine_done += job_steps
        self.progress.section(45, 90, self._refine_done, self._refine_total)
        host_video = output.tensors[0].cpu()
        del output, samples, noise_pair, mask
        return host_video.float()


def _prepare_model(model, filename, strength, attention, runtime=None):
    import nodes

    if runtime is not None:
        runtime.check_cancel()
    refined = model.clone()
    if filename and filename != "None":
        refined = nodes.LoraLoaderModelOnly().load_lora_model_only(refined, filename, strength)[0]
    if runtime is not None:
        runtime.check_cancel()
    # Use optional registered patches through their public node API when present.
    optional = (
        ("H3AdaLNLoRAFix", {"mode": "port"}),
        ("MiniMaxLowVRAMAttention", {"head_chunks": 4}),
        ("MiniMaxChunkFeedForward", {"chunks": 2, "seq_threshold": 4096}),
    )
    for name, arguments in optional:
        if runtime is not None:
            runtime.check_cancel()
        implementation = nodes.NODE_CLASS_MAPPINGS.get(name)
        if implementation is not None:
            try:
                refined = _first(implementation.execute(model=refined, **arguments))
            except (AttributeError, TypeError) as error:
                logging.getLogger(__name__).debug("Optional %s patch skipped: %s", name, error)
    dense_model = refined
    applied_attention = "dense"
    if attention == "sparse":
        try:
            from comfy_extras.nodes_sparse_attention import apply_block_sparse_attention, parse_block_list
            refined = apply_block_sparse_attention(
                refined, tau=1.3, topk_ratio=0.0, vsa=False, start_percent=0.0,
                end_percent=1.0, min_tokens=8192, dense_blocks=parse_block_list(""),
                sink_conditioning="exact_kv_and_rows", extra_tokens=256, verbose=False)
            applied_attention = "sparse"
        except (ImportError, AttributeError, TypeError, NotImplementedError, RuntimeError) as error:
            if _out_of_memory(error):
                raise
            if runtime is not None:
                runtime.check_cancel()
                runtime.notice("sparse", "Sparse attention unavailable; using dense.")
            logging.getLogger(__name__).debug("Sparse attention setup failed: %s", error)
    if runtime is not None:
        runtime.check_cancel()
    return refined, applied_attention, dense_model


def _sparse_unavailable(error):
    """Recognize unavailable sparse backends without hiding OOM/data errors."""
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _out_of_memory(current):
            return False
        message = str(current).lower()
        marker = any(word in message for word in (
            "sparse", "sol_attn", "sol-attn", "solattn", "comfy_kitchen"))
        unsupported = any(word in message for word in (
            "not available", "unavailable", "not supported", "unsupported", "not implemented",
            "not installed", "no module", "cannot import", "could not import", "missing",
            "no kernel", "not compiled", "requires", "unexpected keyword", "no attribute"))
        if marker and (unsupported or isinstance(current, (ImportError, NotImplementedError))):
            return True
        # Backend exceptions may carry only a generic CUDA support message.
        trace = current.__traceback__
        while trace is not None:
            filename = trace.tb_frame.f_code.co_filename.lower()
            if unsupported and any(part in filename for part in (
                    "nodes_sparse_attention", "sol_attn", "sol-attn", "comfy_kitchen")):
                return True
            trace = trace.tb_next
        current = current.__cause__ or current.__context__
    return False


class _Refinement:
    def __init__(self, runtime, model, prompt, sampler, schedule, audio, seed):
        self.runtime, self.model, self.prompt = runtime, model, prompt
        self.sampler, self.schedule = sampler, schedule
        self.audio, self.seed = audio, seed
        self.extra = sum(embedding.shape[1] for embedding, _ in prompt) + 1024
        self._noise = None
        self._blend_scratch = None

    def _add_weighted(self, destination, values, weight):
        self.runtime.check_cancel()
        # Separate multiply then add preserves the previous blend arithmetic.
        # The reusable CPU workspace avoids a fresh product tensor per job.
        if self._blend_scratch is None or self._blend_scratch.shape != values.shape:
            self._blend_scratch = torch.empty_like(values, device="cpu")
        torch.mul(values, weight, out=self._blend_scratch)
        destination.add_(self._blend_scratch)

    def _job(self, current, noise, region, sigmas, clock_offset):
        self.runtime.check_cancel()
        global_begin, global_end = region.begin + clock_offset, region.end + clock_offset
        a0 = round(_frame_boundary(global_begin) * 40 / 24)
        a1 = round(_frame_boundary(global_end) * 40 / 24)
        soundtrack = self.audio[..., a0:a1].contiguous()
        return self.runtime.sampler_call(
            self.model, self.prompt, self.sampler, sigmas,
            current[region.index].contiguous(), soundtrack,
            None if noise is None else noise[region.index].contiguous(), self.seed)

    def _joint(self, source, noise, regions, clock_offset=0):
        if len(regions) == 1:
            return self._job(source, noise, regions[0], self.schedule, clock_offset)
        weights = [(region, region.weight()) for region in regions]
        normalizer = torch.zeros((1, 1, *source.shape[2:]), dtype=torch.float32)
        for region, weight in weights:
            self.runtime.check_cancel()
            normalizer[region.index].add_(weight)
        if not bool((normalizer > 0).all()):
            raise RuntimeError("Internal H3 window plan left uncovered latent cells.")
        current = source
        buffers = []
        for step in range(len(self.schedule) - 1):
            self.runtime.check_cancel()
            slot = step % 2
            if slot == len(buffers):
                buffers.append(torch.empty_like(source, dtype=torch.float32))
            accum = buffers[slot]
            accum.zero_()
            step_noise = noise if step == 0 else None
            for region, weight in weights:
                output = self._job(current, step_noise, region,
                                   self.schedule[step:step + 2], clock_offset)
                self._add_weighted(accum[region.index], output, weight)
            current = accum.div_(normalizer)
        return current

    def run(self, source, budget, chunking, tiling, join):
        self.runtime.check_cancel()
        total, height, width = source.shape[2:]
        if self._noise is None or self._noise.shape != source.shape:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            self._noise = torch.randn(source.shape, generator=generator, dtype=torch.float32)
        self.runtime.check_cancel()
        noise = self._noise
        intervals = _time_regions(total, height, width, budget, chunking)
        if join == "stepped" or len(intervals) == 1:
            regions = [region for begin, end in intervals for region in
                       _space_regions(begin, end, height, width, budget, self.extra, tiling)]
            self.runtime.begin_refinement(len(regions) * (len(self.schedule) - 1))
            return self._joint(source, noise, regions), len(intervals), len(regions)
        plans = [(begin, end, _space_regions(0, end - begin, height, width,
                                            budget, self.extra, tiling))
                 for begin, end in intervals]
        self.runtime.begin_refinement(sum(len(regions) for _, _, regions in plans)
                                      * (len(self.schedule) - 1))
        accum = torch.zeros_like(source)
        norm = torch.zeros(1, 1, total, 1, 1)
        jobs = 0
        for begin, end, regions in plans:
            self.runtime.check_cancel()
            result = self._joint(source[:, :, begin:end], noise[:, :, begin:end], regions, begin)
            weight = _blend_axis(end - begin)[None, None, :, None, None]
            self._add_weighted(accum[:, :, begin:end], result, weight)
            norm[:, :, begin:end].add_(weight)
            jobs += len(regions)
        return accum.div_(norm), len(intervals), jobs


def _resize_float_plane(plane, size):
    # Pillow mode F retains signed residuals and avoids an 8-bit round trip.
    return np.asarray(Image.fromarray(plane).resize(size, Image.Resampling.LANCZOS))


def _preserve_source_frame(generated, source):
    """Remove the part of the H3 change that survives a round trip to the source grid."""
    source_size = (source.shape[1], source.shape[0])
    target_size = (generated.shape[1], generated.shape[0])
    for channel in range(3):
        target = generated[..., channel]
        change = target - _resize_float_plane(source[..., channel], target_size)
        # For delta = H3 - baseline, keep delta - up(down(delta)).
        # Filtering delta also leaves an already correct baseline untouched.
        source_change = _resize_float_plane(change, source_size)
        del change
        target -= _resize_float_plane(source_change, target_size)
    np.clip(generated, 0.0, 1.0, out=generated)


def _output_audio(audio, frame_count, fps):
    if audio is not None:
        return audio
    sample_rate = 44100
    return {"waveform": torch.zeros(1, 2, round(frame_count / fps * sample_rate),
                                    dtype=torch.float32, device="cpu"),
            "sample_rate": sample_rate}


def _decode_frames(runtime, vae, video, count, source_images=None):
    runtime.progress.update(90)
    height, width = video.shape[3] * 16, video.shape[4] * 16
    output = torch.empty(count, height, width, 3, dtype=torch.float32, device="cpu")
    written = 0
    # Chunk boundaries are complete 17-frame / 5-token groups. Neighbouring
    # groups are decoded as context, then discarded before writing the output.
    for begin in range(0, video.shape[2], 30):
        runtime.check_cancel()
        end = min(video.shape[2], begin + 30)
        context_begin = max(0, begin - 5)
        context_end = min(video.shape[2], end + 7)
        skip = _frame_boundary(begin) - _frame_boundary(context_begin)
        keep = min(_frame_boundary(end) - _frame_boundary(begin), count - written)
        if keep <= 0:
            break
        frames = vae.decode(video[:, :, context_begin:context_end].contiguous())
        runtime.check_cancel()
        frames = frames.reshape(-1, *frames.shape[-3:])
        if len(frames) < skip + keep:
            raise RuntimeError("The connected VAE decoded fewer frames than the H3 time layout requires.")
        # Transfer only the kept RGB frames; cast on the CPU rather than
        # creating a full-resolution float32 temporary on the GPU.
        kept = frames[skip:skip + keep, ..., :3].cpu()
        del frames
        output[written:written + keep].copy_(kept)
        del kept
        if source_images is not None:
            for index in range(written, written + keep):
                runtime.check_cancel()
                source_frame = source_images[index, ..., :3].detach().to(device="cpu", dtype=torch.float32)
                _preserve_source_frame(output[index].numpy(), source_frame.numpy())
                runtime.progress.section(90, 99, index + 1, count)
        written += keep
        runtime.progress.section(90, 99, written, count)
    if written != count:
        raise RuntimeError(f"Expected {count} output frames; decoded {written}.")
    return output


class SolricksMiniMaxH3Upscaler:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        loras = list(dict.fromkeys([*folder_paths.get_filename_list("loras"), "None"]))
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Plain MiniMax H3 FL2VA diffusion model."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE with a decoder."}),
                "images": ("IMAGE",),
                "upscale_model": (_model_names(),),
                "megapixels": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 16.0, "step": 0.05}),
                "lora": (loras, {"default": _LORA_FILE if _LORA_FILE in loras else loras[0]}),
                "enable temporal chunking": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Allow overlapping time windows for long clips or limited VRAM."}),
                "enable spatial tiling": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Allow overlapping image tiles when a frame does not fit in VRAM."}),
            },
            "optional": {
                "conditioning": ("CONDITIONING", {"tooltip": "Optional H3 text prompt describing the clip."}),
                "audio": ("AUDIO", {"tooltip": "Original soundtrack; returned unchanged."}),
                "fps": ("FLOAT", {"forceInput": True, "tooltip": "Source frame rate; defaults to 24."}),
                "audio_vae": ("VAE", {"tooltip": "H3 audio VAE for soundtrack conditioning."}),
                "preserve source detail": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Retain source texture and tone, and use finer detail from H3. At matching dimensions this returns the source RGB directly."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "run"
    CATEGORY = "CustomPostProcess"
    DESCRIPTION = "MiniMax H3 latent enlargement and windowed refinement with optional source texture protection."

    @torch.inference_mode()
    def run(self, model, vae, images, upscale_model, megapixels, lora,
            conditioning=None, audio=None, fps=None, audio_vae=None, unique_id=None, **controls):
        # Widget keys contain spaces so their visible labels are exactly the
        # requested names; ComfyUI passes them as ordinary keyword entries.
        chunking = bool(controls.get("enable temporal chunking", True))
        tiling = bool(controls.get("enable spatial tiling", True))
        with _QuietModelLogs():
            return self._execute(
                model, vae, images, upscale_model, megapixels, _DENOISE, lora,
                _TILE_TOKENS, _SAMPLER, redraw_seed=_SEED, attention=_ATTENTION,
                chunk_join=_CHUNK_JOIN, chunking=chunking, tiling=tiling,
                redraw_start=_REDRAW_START, redraw_steps=str(_REDRAW_STEPS),
                conditioning=conditioning, audio=audio, fps=fps, audio_vae=audio_vae,
                node_id=unique_id, preserve_source_detail=bool(controls.get("preserve source detail", True)))

    @torch.inference_mode()
    def _execute(self, model, vae, images, upscale_model, megapixels, denoise, lora,
            tile_tokens, sampler_name, redraw_seed=42, attention="sparse",
            chunk_join="stepped", chunking=True, tiling=True, redraw_start=0.6,
            redraw_steps="2", audio=None, audio_vae=None, conditioning=None, fps=None,
            node_id=None, preserve_source_detail=False):
        import folder_paths
        import comfy.samplers

        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("Expected IMAGE [frames,height,width,RGB].")
        if any(size <= 0 for size in images.shape[:3]):
            raise ValueError("The IMAGE batch is empty.")
        source_fps = 24.0 if fps is None else float(fps)
        numbers = (source_fps, float(megapixels), float(denoise), float(redraw_start))
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("FPS and numeric settings must be finite.")
        if source_fps <= 0 or float(megapixels) <= 0 or not 0 <= float(denoise) <= 1:
            raise ValueError("FPS and megapixels must be positive; denoise must be between zero and one.")
        if not 0 < float(redraw_start) < 1:
            raise ValueError("redraw_start must be between zero and one, excluding both endpoints.")
        steps = int(str(redraw_steps).split()[0])
        if steps not in (2, 3) or int(tile_tokens) < 0:
            raise ValueError("Choose two or three redraw steps and a non-negative tile budget.")
        input_size = (int(images.shape[2]), int(images.shape[1]))
        canvas, target = _canvas(*input_size, megapixels)
        frame_count = len(images)
        if preserve_source_detail and input_size == target:
            import comfy.model_management as management

            progress = _NodeProgress(management.throw_exception_if_processing_interrupted, node_id)
            result = images[..., :3].detach().to(device="cpu", dtype=torch.float32)
            soundtrack = _output_audio(audio, frame_count, source_fps)
            progress.update(100)
            print(f"{_LOG} Source detail: returning {frame_count} original RGB frames at "
                  f"{target[0]}x{target[1]}.", flush=True)
            return result, soundtrack
        _model_names()
        if canvas != target:
            folder_paths.get_full_path_or_raise(_MODEL_FOLDER, upscale_model)
        if float(denoise) > 0 and lora and lora != "None":
            folder_paths.get_full_path_or_raise("loras", lora)
        try:
            runtime = _Runtime(node_id)
        except (ImportError, AttributeError) as error:
            raise RuntimeError("Update ComfyUI: this SOLRICKS node requires the built-in H3 latent upscaler.") from error
        started = time.perf_counter()
        padded, latent_steps, audio_steps = runtime.clock(frame_count)
        protection = "on" if preserve_source_detail else "off"
        print(f"{_LOG} Starting: {frame_count} frames, "
              f"{images.shape[2]}x{images.shape[1]} -> {target[0]}x{target[1]}, "
              f"source detail={protection}.", flush=True)
        try:
            runtime.check_cancel()
            fitted = _fit_input(images, canvas, padded, runtime)
            runtime.check_cancel()
            encoded = vae.encode(fitted[..., :3]).cpu().float()
            del fitted
            runtime.progress.update(20)
            if encoded.ndim != 5 or encoded.shape[:3] != (1, 24, latent_steps):
                raise ValueError("The video VAE must produce native MiniMax H3 latents [1,24,time,height,width].")
            # Complete VAE work before loading the latent upscaler and DiT.
            if float(denoise) > 0:
                audio_z, _ = runtime.audio_latent(audio, audio_vae, audio_steps, source_fps)
            runtime.progress.update(25)
            enlarged, _ = runtime.enlarge(encoded, upscale_model, target, bool(chunking))
            del encoded
            if float(denoise) > 0:
                runtime.release()
                runtime.check_cancel()
                refined_model, attention_note, dense_model = _prepare_model(
                    model, lora, float(denoise), attention, runtime=runtime)
                runtime.progress.update(45)
                prompt = _prompt_only(conditioning)
                sampler = comfy.samplers.sampler_object(sampler_name)
                schedule = _refine_schedule(redraw_start, steps)

                def make_refiner():
                    return _Refinement(runtime, refined_model, prompt, sampler,
                                       schedule, audio_z, int(redraw_seed))

                refiner = make_refiner()
                budget = runtime.budget(int(tile_tokens))
                oom_retries = 0
                try:
                    while True:
                        runtime.check_cancel()
                        try:
                            enlarged, _, _ = refiner.run(
                                enlarged, budget, bool(chunking), bool(tiling), chunk_join)
                            break
                        except Exception as error:
                            runtime.check_cancel()
                            if attention_note == "sparse" and _sparse_unavailable(error):
                                runtime.notice("sparse", "Sparse attention unavailable; restarting with dense.")
                                logging.getLogger(__name__).debug("Sparse sampling failed: %s", error)
                                runtime.release()
                                refined_model = dense_model
                                attention_note = "dense (sparse unavailable)"
                                refiner = make_refiner()
                                continue
                            if not _out_of_memory(error):
                                raise
                            smaller = max(4096, (budget // 2 // 1024) * 1024)
                            if (not chunking and not tiling) or smaller >= budget or oom_retries >= 6:
                                raise RuntimeError("H3 redraw ran out of memory. Enable chunking/tiling or reduce output megapixels.") from error
                            budget = smaller
                            oom_retries += 1
                            runtime.notice("refine_memory", "VRAM limit reached; retrying with smaller windows.")
                            runtime.release()
                finally:
                    del refiner, refined_model, dense_model, audio_z
                    runtime.release()
            runtime.check_cancel()
            result = _decode_frames(runtime, vae, enlarged, frame_count,
                                    source_images=images if preserve_source_detail else None)
            audio = _output_audio(audio, frame_count, source_fps)
            runtime.progress.update(100)
            print(f"{_LOG} Completed: {frame_count} frames, {target[0]}x{target[1]}, "
                  f"{time.perf_counter() - started:.1f}s.", flush=True)
            return result, audio
        except runtime.mm.InterruptProcessingException:
            print(f"{_LOG} Cancelled.", flush=True)
            raise
        finally:
            runtime.release()


NODE_CLASS_MAPPINGS = {
    "SolricksMiniMaxH3Upscaler": SolricksMiniMaxH3Upscaler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SolricksMiniMaxH3Upscaler": "MiniMax H3 Upscaler",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
