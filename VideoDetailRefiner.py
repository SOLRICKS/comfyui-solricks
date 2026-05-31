# Video Detail Refiner
# v0.2.1
# SOLRICKS


import logging
import os
import threading
import torch
import torch.nn.functional as F


try:
    from safetensors.torch import load_file
except ImportError:  # ComfyUI normally includes safetensors, but keep import safe.
    load_file = None


try:
    from .model import DLAANet
    from .utils import rgb_luma, clamp01
except ImportError:
    from model import DLAANet
    from utils import rgb_luma, clamp01


try:
    import comfy.model_management as mm
except ImportError:
    mm = None


try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None


log = logging.getLogger("VideoDetailRefiner")


MIN_EFFECT_STRENGTH = 1e-5
OOM_TILE_STEPS = (1536, 1024, 512, 256)


def load_torch_weights(path, device):
    """Load legacy .pth weights safely when the installed torch supports weights_only."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


# The presets are intentionally separate from VideoTAADLAA presets.
# This node is a detail/texture refiner, not a full anti-aliasing pipeline.
RESTORE_PRESETS = {
    "Photo": {
        "model_weight": 0.48,
        "detail_gain": 0.44,
        "texture_gain": 0.34,
        "temporal_gain": 0.00,
        "detail_limit": 0.018,
        "texture_limit": 0.014,
        "edge_threshold": 0.056,
        "highlight_protect": 0.76,
        "chroma_guard": 0.42,
        "fine_line_strength": 0.18,
        "dehalo_strength": 0.050,
        "micro_contrast_strength": 0.035,
        "motion_suppression": 0.00,
    },
    "Video Balanced": {
        "model_weight": 0.55,
        "detail_gain": 0.55,
        "texture_gain": 0.45,
        "temporal_gain": 0.25,
        "detail_limit": 0.020,
        "texture_limit": 0.018,
        "edge_threshold": 0.055,
        "highlight_protect": 0.66,
        "chroma_guard": 0.36,
        "fine_line_strength": 0.22,
        "dehalo_strength": 0.060,
        "micro_contrast_strength": 0.040,
        "motion_suppression": 0.85,
    },
    "Video High Detail": {
        "model_weight": 0.72,
        "detail_gain": 0.78,
        "texture_gain": 0.65,
        "temporal_gain": 0.20,
        "detail_limit": 0.026,
        "texture_limit": 0.022,
        "edge_threshold": 0.050,
        "highlight_protect": 0.60,
        "chroma_guard": 0.44,
        "fine_line_strength": 0.32,
        "dehalo_strength": 0.080,
        "micro_contrast_strength": 0.052,
        "motion_suppression": 0.75,
    },
    "Soft": {
        "model_weight": 0.36,
        "detail_gain": 0.34,
        "texture_gain": 0.24,
        "temporal_gain": 0.38,
        "detail_limit": 0.014,
        "texture_limit": 0.011,
        "edge_threshold": 0.070,
        "highlight_protect": 0.78,
        "chroma_guard": 0.56,
        "fine_line_strength": 0.16,
        "dehalo_strength": 0.045,
        "micro_contrast_strength": 0.026,
        "motion_suppression": 0.95,
    },
    "Performance": {
        "model_weight": 0.00,
        "detail_gain": 0.42,
        "texture_gain": 0.00,
        "temporal_gain": 0.18,
        "detail_limit": 0.016,
        "texture_limit": 0.000,
        "edge_threshold": 0.064,
        "highlight_protect": 0.72,
        "chroma_guard": 0.30,
        "fine_line_strength": 0.14,
        "dehalo_strength": 0.040,
        "micro_contrast_strength": 0.022,
        "motion_suppression": 0.80,
    },
}


PRESET_ALIASES = {
    # Step 1 / older workflow names
    "Balanced": "Video Balanced",
    "High Detail": "Video High Detail",
    "Soft Restore": "Soft",
    # extra forgiving aliases
    "Video Detail": "Video High Detail",
    "Detail": "Video High Detail",
    "Photo Restore": "Photo",
}


class VideoDetailRefiner:
    def __init__(self):
        self.model_cache = {}
        self._model_lock = threading.Lock()
        self._warned = set()
        self._warning_lock = threading.Lock()
        self._sobel_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "preset": ([
                    "Photo",
                    "Video Balanced",
                    "Video High Detail",
                    "Soft",
                    "Performance",
                ],),
                "detail_strength": (
                    "FLOAT",
                    {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.05},
                ),
                "texture_strength": (
                    "FLOAT",
                    {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.05},
                ),
                "temporal_stability": (
                    "FLOAT",
                    {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "CustomPostProcess"

    def _warn_once(self, key, message):
        with self._warning_lock:
            if key in self._warned:
                return
            self._warned.add(key)
        log.warning(message)

    def _normalize_preset(self, preset):
        preset = PRESET_ALIASES.get(preset, preset)
        return preset if preset in RESTORE_PRESETS else "Video Balanced"

    def _resolve_run_mode(self, preset, frame_count):
        """Make image/video behavior explicit and safe for old workflows."""
        is_single_image = frame_count == 1

        if is_single_image:
            # Single images should default to the Photo tuning unless the user explicitly
            # picked Performance/Soft. Old workflows with Video presets stay safe.
            if preset in ("Video Balanced", "Video High Detail"):
                return "image", "Photo"
            return "image", preset

        # Multi-frame batches should use video tuning. If Photo is used on a video,
        # fall back to Video Balanced so temporal stabilization is enabled.
        if preset == "Photo":
            return "video", "Video Balanced"
        return "video", preset

    def _get_device(self):
        if mm is not None:
            return mm.get_torch_device()
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _vram_mb(self, device):
        try:
            torch_device = torch.device(device)
        except Exception:
            return 0

        if torch_device.type != "cuda" or not torch.cuda.is_available():
            return 0

        try:
            device_index = torch_device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            return torch.cuda.get_device_properties(device_index).total_memory // (1024 * 1024)
        except Exception:
            return 0

    def _tile_size_for_vram(self, vram_mb):
        # Match the TAA node style but keep CPU/VRAM conservative.
        if vram_mb <= 0:
            return 512
        if vram_mb <= 8192:
            return 512
        if vram_mb <= 16384:
            return 1024
        return 1536

    def _tile_retry_sequence(self, initial_tile_size):
        initial_tile_size = int(initial_tile_size)
        sequence = []

        if initial_tile_size not in OOM_TILE_STEPS:
            sequence.append(initial_tile_size)

        for tile in OOM_TILE_STEPS:
            if tile <= initial_tile_size and tile not in sequence:
                sequence.append(tile)

        if 256 not in sequence:
            sequence.append(256)

        return sequence

    def _load_texture_model(self, device):
        key = str(device)
        with self._model_lock:
            if key in self.model_cache:
                return self.model_cache[key]

            base_path = os.path.dirname(os.path.realpath(__file__))
            safetensors_path = os.path.join(base_path, "DLAATexture.safetensors")
            pth_path = os.path.join(base_path, "DLAATexture.pth")

            model_path = None
            if os.path.exists(safetensors_path):
                model_path = safetensors_path
            elif os.path.exists(pth_path):
                model_path = pth_path

            if model_path is None:
                self._warn_once(
                    "missing_texture_model",
                    "[Video Detail Refiner] DLAATexture model not found. Running classical detail pass only.",
                )
                self.model_cache[key] = None
                return None

            try:
                net = DLAANet().to(device)

                if model_path.endswith(".safetensors"):
                    if load_file is None:
                        raise ImportError("safetensors is not available")
                    state_dict = load_file(model_path, device=str(device))
                else:
                    state_dict = load_torch_weights(model_path, device)

                if isinstance(state_dict, dict):
                    if "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
                    elif "params_ema" in state_dict:
                        state_dict = state_dict["params_ema"]
                    elif "params" in state_dict:
                        state_dict = state_dict["params"]
                    elif "model" in state_dict:
                        state_dict = state_dict["model"]

                state_dict = {
                    name: tensor
                    for name, tensor in state_dict.items()
                    if name != "jitter_offsets"
                }

                load_result = net.load_state_dict(state_dict, strict=False)
                missing = [name for name in load_result.missing_keys if name != "jitter_offsets"]

                if missing:
                    log.warning("[Video Detail Refiner] Texture model missing keys: %s", missing)
                if load_result.unexpected_keys:
                    log.warning(
                        "[Video Detail Refiner] Texture model unexpected keys: %s",
                        load_result.unexpected_keys,
                    )

                net = net.float().eval()
                self.model_cache[key] = net

                n_params = sum(p.numel() for p in net.parameters())
                log.info("[Video Detail Refiner] Loaded %s", os.path.basename(model_path))
                log.info("[Video Detail Refiner] Texture model parameters: %.2fM", n_params / 1e6)

                return net

            except Exception as e:
                self._warn_once(
                    "texture_model_load_failed",
                    f"[Video Detail Refiner] Texture model could not be loaded: {type(e).__name__}: {e}",
                )
                self.model_cache[key] = None
                return None

    def _get_sobel_kernels(self, device, dtype):
        cache_key = (str(device), dtype)
        if cache_key not in self._sobel_cache:
            sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                device=device,
                dtype=dtype,
            ).view(1, 1, 3, 3)
            sobel_y = torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                device=device,
                dtype=dtype,
            ).view(1, 1, 3, 3)
            self._sobel_cache[cache_key] = (sobel_x, sobel_y)
        return self._sobel_cache[cache_key]

    def _luma_edge(self, image, net=None):
        luma = rgb_luma(image)

        if net is not None:
            sobel_x = net.sobel_x.to(device=image.device, dtype=image.dtype)
            sobel_y = net.sobel_y.to(device=image.device, dtype=image.dtype)
        else:
            sobel_x, sobel_y = self._get_sobel_kernels(image.device, image.dtype)

        sx = F.conv2d(luma, sobel_x, padding=1)
        sy = F.conv2d(luma, sobel_y, padding=1)
        edge = torch.sqrt(sx * sx + sy * sy + 1e-6)
        return luma, edge

    def _tile_weight_map(self, th, tw, overlap, device, dtype):
        w_y = torch.ones(th, device=device, dtype=dtype)
        w_x = torch.ones(tw, device=device, dtype=dtype)
        ramp = min(overlap, th // 4, tw // 4)

        if ramp > 0:
            values = torch.linspace(
                1.0 / (ramp + 1),
                ramp / (ramp + 1),
                ramp,
                device=device,
                dtype=dtype,
            )
            w_y[:ramp] = values
            w_y[-ramp:] = torch.flip(values, dims=[0])
            w_x[:ramp] = values
            w_x[-ramp:] = torch.flip(values, dims=[0])

        return torch.minimum(w_y.view(1, 1, th, 1), w_x.view(1, 1, 1, tw))

    def _tiled_forward(self, net, image, tile_size=1024, overlap=32):
        if net is None:
            return image

        tile_size = int(tile_size)
        overlap = int(overlap)

        if tile_size <= 0:
            raise ValueError(f"Invalid tile_size: {tile_size}")
        if overlap < 0:
            raise ValueError(f"Invalid overlap: {overlap}")
        if overlap * 2 >= tile_size:
            raise ValueError(f"Invalid tiling settings: tile_size={tile_size}, overlap={overlap}")

        b, c, h, w = image.shape
        if h <= tile_size and w <= tile_size:
            return torch.clamp(net(image), 0.0, 1.0)

        step = tile_size - overlap * 2
        if step <= 0:
            raise ValueError(f"Invalid tiling settings: tile_size={tile_size}, overlap={overlap}")

        out = torch.zeros_like(image)
        weight = torch.zeros(b, 1, h, w, device=image.device, dtype=image.dtype)
        weight_cache = {}

        y0 = 0
        while y0 < h:
            y1 = min(y0 + tile_size, h)
            y0c = max(0, y1 - tile_size)

            x0 = 0
            while x0 < w:
                x1 = min(x0 + tile_size, w)
                x0c = max(0, x1 - tile_size)

                tile = image[:, :, y0c:y1, x0c:x1]
                tile_out = net(tile)
                th, tw = tile.shape[2], tile.shape[3]

                cache_key = (th, tw, overlap, image.device, image.dtype)
                if cache_key not in weight_cache:
                    weight_cache[cache_key] = self._tile_weight_map(
                        th,
                        tw,
                        overlap,
                        image.device,
                        image.dtype,
                    )

                w_map = weight_cache[cache_key]
                out[:, :, y0c:y1, x0c:x1] += tile_out * w_map
                weight[:, :, y0c:y1, x0c:x1] += w_map

                if x1 == w:
                    break
                x0 += step

            if y1 == h:
                break
            y0 += step

        return torch.clamp(out / weight.clamp(min=1e-6), 0.0, 1.0)

    def _run_model_with_retry(self, net, image, initial_tile_size, frame_index=None):
        if net is None:
            return None, initial_tile_size

        last_error = None
        for tile_size in self._tile_retry_sequence(initial_tile_size):
            try:
                model_out = self._tiled_forward(net, image, tile_size=tile_size, overlap=32)
                return model_out, tile_size

            except torch.OutOfMemoryError as e:
                last_error = e
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log.warning(
                    "[Video Detail Refiner] OOM at tile size %d%s, retrying smaller tile.",
                    tile_size,
                    f" on frame {frame_index}" if frame_index is not None else "",
                )

            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                last_error = e
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log.warning(
                    "[Video Detail Refiner] OOM at tile size %d%s, retrying smaller tile.",
                    tile_size,
                    f" on frame {frame_index}" if frame_index is not None else "",
                )

        self._warn_once(
            "texture_model_oom_skip",
            "[Video Detail Refiner] Texture model pass failed even at tile size 256. Skipping model pass.",
        )
        if last_error is not None:
            log.debug("[Video Detail Refiner] Final OOM error: %r", last_error)
        return None, 256

    def _motion_protect(self, cfg, temporal_motion, run_mode):
        if run_mode != "video":
            return 1.0
        return 1.0 - min(max(float(temporal_motion) * cfg["motion_suppression"], 0.0), 0.95)

    def _classical_detail_pass(self, image, cfg, detail_strength, temporal_motion, run_mode):
        detail_strength = max(0.0, min(float(detail_strength), 2.0))
        if detail_strength <= MIN_EFFECT_STRENGTH:
            return image

        luma, edge = self._luma_edge(image)

        local_3 = F.avg_pool2d(
            F.pad(image, [1, 1, 1, 1], mode="reflect"),
            3,
            stride=1,
        )
        local_7 = F.avg_pool2d(
            F.pad(image, [3, 3, 3, 3], mode="reflect"),
            7,
            stride=1,
        )

        fine_residual = image - local_3
        soft_residual = image - local_7
        residual = fine_residual * 0.72 + soft_residual * 0.28

        detail_energy = rgb_luma(fine_residual.abs())
        detail_mask = torch.sigmoid((detail_energy - 0.0045) * 80.0)
        edge_mask = torch.sigmoid((edge - cfg["edge_threshold"]) * 16.0)
        highlight_mask = torch.sigmoid((luma - 0.82) * 12.0)

        highlight_protect = 1.0 - highlight_mask * cfg["highlight_protect"]
        motion_protect = self._motion_protect(cfg, temporal_motion, run_mode)

        mask = (
            detail_mask *
            (0.62 + edge_mask * 0.38) *
            highlight_protect *
            motion_protect
        ).clamp(0.0, 1.0)

        residual = residual.clamp(-cfg["detail_limit"], cfg["detail_limit"])
        out = image + residual * mask * cfg["detail_gain"] * detail_strength
        return torch.clamp(out, 0.0, 1.0)

    def _micro_contrast_lite(self, image, cfg, detail_strength, temporal_motion, run_mode):
        strength = cfg["micro_contrast_strength"] * max(0.0, min(float(detail_strength), 2.0))
        if strength <= MIN_EFFECT_STRENGTH:
            return image

        radius = 5
        pad = radius // 2
        local_avg = F.avg_pool2d(
            F.pad(image, [pad, pad, pad, pad], mode="reflect"),
            radius,
            stride=1,
        )

        residual = (image - local_avg).clamp(-0.014, 0.014)
        luma = rgb_luma(image)
        detail_energy = rgb_luma(residual.abs())

        detail_mask = torch.sigmoid((detail_energy - 0.0055) * 80.0)
        highlight_mask = torch.sigmoid((luma - 0.84) * 12.0)
        protect = 1.0 - highlight_mask * cfg["highlight_protect"]
        motion_protect = self._motion_protect(cfg, temporal_motion, run_mode)

        out = image + residual * detail_mask * protect * motion_protect * strength
        return torch.clamp(out, 0.0, 1.0)

    def _fine_line_recovery(self, image, cfg, detail_strength, temporal_motion, run_mode):
        strength = cfg["fine_line_strength"] * max(0.0, min(float(detail_strength), 2.0))
        if strength <= MIN_EFFECT_STRENGTH:
            return image

        luma, edge = self._luma_edge(image)
        local_avg = F.avg_pool2d(image, 3, stride=1, padding=1)
        detail = image - local_avg
        detail_energy = rgb_luma(detail.abs())

        edge_mask = torch.sigmoid((edge - cfg["edge_threshold"] * 0.92) * 18.0)
        detail_mask = torch.sigmoid((detail_energy - 0.0060) * 90.0)
        dark_line_bias = torch.sigmoid((0.48 - luma) * 8.0)
        highlight_mask = torch.sigmoid((luma - 0.82) * 12.0)

        line_mask = (
            edge_mask *
            detail_mask *
            (0.45 + dark_line_bias * 0.55) *
            (1.0 - highlight_mask * cfg["highlight_protect"])
        ).clamp(0.0, 1.0)

        motion_protect = self._motion_protect(cfg, temporal_motion, run_mode)
        recovery = detail.clamp(-cfg["detail_limit"] * 0.75, cfg["detail_limit"] * 0.75)

        out = image + recovery * line_mask * strength * motion_protect
        return torch.clamp(out, 0.0, 1.0)

    def _dehalo_lite(self, image, cfg, detail_strength):
        strength = cfg["dehalo_strength"] * max(0.0, min(float(detail_strength), 2.0))
        if strength <= MIN_EFFECT_STRENGTH:
            return image

        luma, edge = self._luma_edge(image)
        edge_mask = torch.sigmoid((edge - cfg["edge_threshold"] * 1.35) * 14.0)

        local_avg = F.avg_pool2d(
            F.pad(image, [2, 2, 2, 2], mode="reflect"),
            5,
            stride=1,
        )
        halo_residual = image - local_avg

        bright_halo = halo_residual.clamp(min=0.0)
        dark_halo = (-halo_residual).clamp(min=0.0)

        dark_protect = torch.sigmoid((0.22 - luma) * 12.0)
        light_protect = torch.sigmoid((luma - 0.78) * 12.0)

        bright_reduce = bright_halo * edge_mask * strength * (1.0 - light_protect * 0.55)
        dark_restore = dark_halo * edge_mask * strength * 0.35 * (1.0 - dark_protect * 0.35)

        out = image - bright_reduce + dark_restore
        return torch.clamp(out, 0.0, 1.0)

    def _texture_model_pass(self, net, image, cfg, texture_strength, tile_size, temporal_motion, run_mode, frame_index):
        texture_strength = max(0.0, min(float(texture_strength), 2.0))
        if (
            net is None or
            texture_strength <= MIN_EFFECT_STRENGTH or
            cfg["model_weight"] <= MIN_EFFECT_STRENGTH or
            cfg["texture_gain"] <= MIN_EFFECT_STRENGTH
        ):
            return image, tile_size

        model_out, used_tile = self._run_model_with_retry(
            net,
            image,
            initial_tile_size=tile_size,
            frame_index=frame_index,
        )

        if model_out is None:
            return image, used_tile

        if model_out.shape != image.shape:
            self._warn_once(
                "texture_model_shape_mismatch",
                "[Video Detail Refiner] Texture model output shape mismatch. Skipping model pass.",
            )
            return image, used_tile

        # Keep global brightness stable before extracting the generated texture.
        model_out = model_out - model_out.mean(dim=(1, 2, 3), keepdim=True) + image.mean(dim=(1, 2, 3), keepdim=True)

        blur_kernel = 5 if run_mode == "image" else 7
        model_blur = F.avg_pool2d(
            F.pad(model_out, [blur_kernel // 2] * 4, mode="reflect"),
            blur_kernel,
            stride=1,
        )
        texture_delta = model_out - model_blur

        luma, edge = self._luma_edge(image, net)
        highlight_mask = torch.sigmoid((luma - 0.82) * 12.0)
        edge_mask = torch.sigmoid((edge - cfg["edge_threshold"]) * 16.0)
        motion_protect = self._motion_protect(cfg, temporal_motion, run_mode)

        texture_delta = texture_delta.clamp(-cfg["texture_limit"], cfg["texture_limit"])
        mask = (
            (0.34 + edge_mask * 0.66) *
            (1.0 - highlight_mask * cfg["highlight_protect"]) *
            motion_protect
        ).clamp(0.0, 1.0)

        out = image + texture_delta * mask * cfg["model_weight"] * cfg["texture_gain"] * texture_strength
        return torch.clamp(out, 0.0, 1.0), used_tile

    def _chroma_guard(self, image, source, strength):
        strength = clamp01(strength)
        if strength <= MIN_EFFECT_STRENGTH:
            return image

        source_luma = rgb_luma(source)
        out_luma = rgb_luma(image)

        source_chroma = source - source_luma
        out_chroma = image - out_luma

        chroma_delta = out_chroma - source_chroma
        chroma_energy = rgb_luma(chroma_delta.abs())
        chroma_mask = torch.sigmoid((chroma_energy - 0.010) * 80.0)

        guarded_chroma = torch.lerp(out_chroma, source_chroma, chroma_mask * strength)
        out = out_luma + guarded_chroma

        # Preserve restored luma target after chroma.
        luma_error = rgb_luma(out) - out_luma
        out = out - luma_error
        return torch.clamp(out, 0.0, 1.0)

    def _temporal_stabilize_detail(self, current, previous, cfg, temporal_stability):
        temporal_stability = max(0.0, min(float(temporal_stability), 2.0))
        if previous is None or previous.shape != current.shape or temporal_stability <= MIN_EFFECT_STRENGTH:
            return current

        current_base = F.avg_pool2d(current, 3, stride=1, padding=1)
        previous_base = F.avg_pool2d(previous, 3, stride=1, padding=1)

        current_detail = current - current_base
        previous_detail = previous - previous_base

        base_motion = torch.abs(rgb_luma(current_base) - rgb_luma(previous_base))
        stable_mask = (1.0 - torch.clamp(base_motion / 0.080, 0.0, 1.0)).clamp(0.0, 1.0)

        detail_energy = rgb_luma(current_detail.abs())
        detail_mask = torch.sigmoid((detail_energy - 0.006) * 80.0)

        blend = (
            stable_mask *
            detail_mask *
            cfg["temporal_gain"] *
            temporal_stability
        ).clamp(0.0, 0.45)

        stabilized_detail = torch.lerp(current_detail, previous_detail, blend)
        return torch.clamp(current_base + stabilized_detail, 0.0, 1.0)

    def _estimate_motion(self, current, previous):
        if previous is None or previous.shape != current.shape:
            return 0.0
        return min(max(torch.abs(rgb_luma(current) - rgb_luma(previous)).mean().item() * 10.0, 0.0), 1.0)

    def _process_frame(
        self,
        rgb,
        previous_out,
        net,
        cfg,
        detail_strength,
        texture_strength,
        temporal_stability,
        tile_size,
        run_mode,
        frame_index,
    ):
        source = rgb
        temporal_motion = self._estimate_motion(rgb, previous_out) if run_mode == "video" else 0.0

        out = self._classical_detail_pass(rgb, cfg, detail_strength, temporal_motion, run_mode)
        out = self._micro_contrast_lite(out, cfg, detail_strength, temporal_motion, run_mode)
        out = self._fine_line_recovery(out, cfg, detail_strength, temporal_motion, run_mode)
        out = self._dehalo_lite(out, cfg, detail_strength)
        out, used_tile = self._texture_model_pass(
            net,
            out,
            cfg,
            texture_strength,
            tile_size,
            temporal_motion,
            run_mode,
            frame_index,
        )
        out = self._chroma_guard(out, source, cfg["chroma_guard"])

        if run_mode == "video":
            out = self._temporal_stabilize_detail(out, previous_out, cfg, temporal_stability)

        return torch.clamp(out, 0.0, 1.0), used_tile

    def execute(self, images, preset, detail_strength=1.0, texture_strength=1.0, temporal_stability=1.0):
        detail_strength = max(0.0, min(float(detail_strength), 2.0))
        texture_strength = max(0.0, min(float(texture_strength), 2.0))
        temporal_stability = max(0.0, min(float(temporal_stability), 2.0))

        preset = self._normalize_preset(preset)

        if len(images.shape) != 4:
            raise ValueError("VideoDetailRefiner expects an IMAGE tensor with shape [B, H, W, C].")

        frame_count, height, width, channels = images.shape
        if channels < 3:
            raise ValueError("VideoDetailRefiner expects RGB or RGBA images.")

        run_mode, effective_preset = self._resolve_run_mode(preset, frame_count)
        cfg = RESTORE_PRESETS.get(effective_preset, RESTORE_PRESETS["Video Balanced"])

        device = self._get_device()
        vram_mb = self._vram_mb(device)
        tile_size = self._tile_size_for_vram(vram_mb)

        net = None
        if cfg["model_weight"] > MIN_EFFECT_STRENGTH and texture_strength > MIN_EFFECT_STRENGTH:
            net = self._load_texture_model(device)

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "[Video Detail Refiner] mode=%s preset=%s effective=%s frames=%d size=%dx%d device=%s vram=%dMB tile=%d",
                run_mode,
                preset,
                effective_preset,
                frame_count,
                width,
                height,
                device,
                vram_mb,
                tile_size,
            )

        # Process RGB, preserve alpha channels
        extra_channels = None
        if channels > 3:
            extra_channels = images[:, :, :, 3:].detach().cpu()

        out_channels = 3 if extra_channels is None else 3 + extra_channels.shape[-1]
        out_tensor = torch.empty((frame_count, height, width, out_channels), dtype=images.dtype, device="cpu")

        progress = ProgressBar(frame_count) if ProgressBar is not None else None
        previous_out = None
        current_tile_size = tile_size

        with torch.inference_mode():
            for i in range(frame_count):
                if mm is not None and hasattr(mm, "throw_exception_if_processing_interrupted"):
                    mm.throw_exception_if_processing_interrupted()

                rgb = images[i:i + 1].to(device).permute(0, 3, 1, 2).float()[:, :3]

                restored, used_tile = self._process_frame(
                    rgb=rgb,
                    previous_out=previous_out,
                    net=net,
                    cfg=cfg,
                    detail_strength=detail_strength,
                    texture_strength=texture_strength,
                    temporal_stability=temporal_stability,
                    tile_size=current_tile_size,
                    run_mode=run_mode,
                    frame_index=i,
                )

                if used_tile < current_tile_size:
                    # Keep the successful smaller tile for subsequent frames in the same run.
                    current_tile_size = used_tile

                previous_out = restored.detach() if run_mode == "video" else None

                frame_out = restored.permute(0, 2, 3, 1).detach().cpu()
                if frame_out.dtype != out_tensor.dtype:
                    frame_out = frame_out.to(out_tensor.dtype)

                if extra_channels is not None:
                    frame_extra = extra_channels[i:i + 1]
                    if frame_extra.dtype != frame_out.dtype:
                        frame_extra = frame_extra.to(frame_out.dtype)
                    frame_out = torch.cat((frame_out, frame_extra), dim=-1)

                out_tensor[i:i + 1].copy_(frame_out)

                if progress is not None:
                    progress.update(1)

                if mm is not None and i > 0 and i % 50 == 0:
                    mm.soft_empty_cache()

        return (out_tensor,)


NODE_CLASS_MAPPINGS = {
    "VideoDetailRefiner": VideoDetailRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoDetailRefiner": "✨ Video Detail Refiner",
}
