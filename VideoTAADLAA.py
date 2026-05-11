# Video TAA + DLAA Inference
# v0.2.0
# SOLRICKS
# AI-assisted


import os
import logging
import threading
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


try:
    from .model import DLAANet
    from .utils import rgb_luma, clamp01
    from .taa import TAAState
    from .config import NODE_DEFAULTS, NODE_DEFAULT_FIELDS, INTERNAL_TUNING
    from .presets import (
        PRESETS,
        AUTO_STATIC,
        AUTO_BALANCED,
        AUTO_MOTION,
        MOTION_SUPPRESSION,
        PRESET_MODEL_WEIGHT,
        TEXTURE_PRESETS,
    )
except ImportError:
    from model import DLAANet
    from utils import rgb_luma, clamp01
    from taa import TAAState
    from config import NODE_DEFAULTS, NODE_DEFAULT_FIELDS, INTERNAL_TUNING
    from presets import (
        PRESETS,
        AUTO_STATIC,
        AUTO_BALANCED,
        AUTO_MOTION,
        MOTION_SUPPRESSION,
        PRESET_MODEL_WEIGHT,
        TEXTURE_PRESETS,
    )

try:
    import comfy.model_management as mm
except ImportError:
    mm = None

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None


log = logging.getLogger("VideoTAADLAA")

MIN_EFFECT_STRENGTH = 1e-5


def load_torch_weights(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


# Internal tuning values from config.py.
TUNING = INTERNAL_TUNING

FINE_LINE_DARK_SLOPE = TUNING["fine_line_dark_slope"]

SPECULAR_CLIP_LUMA = TUNING["specular_clip_luma"]
SPECULAR_CLIP_SLOPE = TUNING["specular_clip_slope"]
SPECULAR_EDGE_THRESHOLD = TUNING["specular_edge_threshold"]
SPECULAR_EDGE_SLOPE = TUNING["specular_edge_slope"]
SPECULAR_HIGHLIGHT_SUPPRESSION = TUNING["specular_highlight_suppression"]

MICRO_DETAIL_THRESHOLD = TUNING["micro_detail_threshold"]
MICRO_DETAIL_SLOPE = TUNING["micro_detail_slope"]
MICRO_HIGHLIGHT_LUMA = TUNING["micro_highlight_luma"]
MICRO_HIGHLIGHT_SLOPE = TUNING["micro_highlight_slope"]

DEHALO_DARK_PROTECT_LUMA = TUNING["dehalo_dark_protect_luma"]
DEHALO_LIGHT_PROTECT_LUMA = TUNING["dehalo_light_protect_luma"]
DEHALO_PROTECT_SLOPE = TUNING["dehalo_protect_slope"]

CHROMA_SATURATION_SLOPE = TUNING["chroma_saturation_slope"]
CHROMA_FRINGE_THRESHOLD_SCALE = TUNING["chroma_fringe_threshold_scale"]
CHROMA_FRINGE_SLOPE = TUNING["chroma_fringe_slope"]
CHROMA_DARK_PROTECT_SLOPE = TUNING["chroma_dark_protect_slope"]

TEMPORAL_SPEC_DETAIL_SLOPE = TUNING["temporal_spec_detail_slope"]
TEMPORAL_SPEC_MOTION_SCALE = TUNING["temporal_spec_motion_scale"]

LOCAL_TONEMAP_HIGHLIGHT_LUMA = TUNING["local_tonemap_highlight_luma"]
LOCAL_TONEMAP_HIGHLIGHT_SLOPE = TUNING["local_tonemap_highlight_slope"]
LOCAL_TONEMAP_SHADOW_SLOPE = TUNING["local_tonemap_shadow_slope"]
LOCAL_TONEMAP_RATIO_MIN = TUNING["local_tonemap_ratio_min"]
LOCAL_TONEMAP_RATIO_MAX = TUNING["local_tonemap_ratio_max"]

JITTER_DAMPING_MIN = TUNING["jitter_damping_min"]

FUR_DETAIL_SLOPE = TUNING["fur_detail_slope"]


class VideoTAADLAA:
    def __init__(self):
        self.net_cache = {}
        self._net_lock = threading.Lock()

        self.texture_net_cache = {}
        self._texture_lock = threading.Lock()

        self._seen_warnings = set()
        self._warning_lock = threading.Lock()

        self._load_defaults()

        self.preset_model_weight = PRESET_MODEL_WEIGHT
        self.texture_presets = TEXTURE_PRESETS

    def _load_defaults(self):
        for name in NODE_DEFAULT_FIELDS:
            setattr(self, name, NODE_DEFAULTS[name])

    def _first_time(self, key):
        with self._warning_lock:
            if key in self._seen_warnings:
                return False

            self._seen_warnings.add(key)
            return True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "preset": ([
                    "Auto",
                    "Performance",
                    "Balanced",
                    "High Detail",
                ],),
                "dlaa_intensity": ("FLOAT", {
                    "default": 1.00,
                    "min": 0.00,
                    "max": 2.00,
                    "step": 0.05,
                }),
                "texture_intensity": ("FLOAT", {
                    "default": 1.00,
                    "min": 0.00,
                    "max": 2.00,
                    "step": 0.05,
                }),
                "motion_stability": ("FLOAT", {
                    "default": 1.00,
                    "min": 0.50,
                    "max": 2.00,
                    "step": 0.05,
                }),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION     = "execute"
    CATEGORY     = "CustomPostProcess"

    def _net(self, device):
        key = str(device)
        
        # main model
        with self._net_lock:
            if key in self.net_cache:
                return self.net_cache[key]
                
            net = DLAANet().to(device)
            
            base_path = os.path.dirname(os.path.realpath(__file__))
            
            safetensors_path = os.path.join(base_path, "DLAANet.safetensors")
            pth_path = os.path.join(base_path, "DLAANet.pth")

            if os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path, device=str(device))
                log.info("[DLAA] Loaded DLAANet.safetensors")
            elif os.path.exists(pth_path):
                state_dict = load_torch_weights(pth_path, device)
                log.info("[DLAA] Loaded DLAANet.pth")
            else:
                raise FileNotFoundError(
                    f"[DLAA] Model not found. Expected: {safetensors_path} or {pth_path}"
                )
                
            state_dict = {
                param_name: tensor
                for param_name, tensor in state_dict.items()
                if param_name != "jitter_offsets"
            }
                
            load_result = net.load_state_dict(state_dict, strict=False)
            
            missing_keys = [
                name for name in load_result.missing_keys
                if name != "jitter_offsets"
            ]

            if missing_keys:
                log.warning("[DLAA] Main model missing keys: %s", missing_keys)
                
            if load_result.unexpected_keys:
                log.warning("[DLAA] Main model unexpected keys: %s", load_result.unexpected_keys)
                
            net = net.float()
            net.eval()
            
            n_params = sum(p.numel() for p in net.parameters())
            log.info(f"[DLAA] Main model parameters: {n_params / 1e6:.2f}M")
            
            self.net_cache[key] = net
            
            return net
            
    def _texture_net(self, device):
        texture_enabled = getattr(
            self,
            "texture_pass_enabled",
            NODE_DEFAULTS.get("texture_pass_enabled", True)
        )

        if not texture_enabled:
            return None
            
        key = str(device)
        
        with self._texture_lock:
            if key in self.texture_net_cache:
                return self.texture_net_cache[key]
                
            base_path = os.path.dirname(os.path.realpath(__file__))
            
            safetensors_path = os.path.join(base_path, "DLAATexture.safetensors")
            pth_path = os.path.join(base_path, "DLAATexture.pth")
            
            if os.path.exists(safetensors_path):
                model_path = safetensors_path
            elif os.path.exists(pth_path):
                model_path = pth_path
            else:
                if self._first_time("texture_missing"):
                    log.info("[DLAA] Texture model not found, skipping texture pass.")
                return None
                
            try:
                net = DLAANet().to(device)

                if model_path.endswith(".safetensors"):
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
                    param_name: tensor
                    for param_name, tensor in state_dict.items()
                    if param_name != "jitter_offsets"
                }
                    
                load_result = net.load_state_dict(state_dict, strict=False)

                missing_keys = [
                    name for name in load_result.missing_keys
                    if name != "jitter_offsets"
                ]

                if missing_keys:
                    log.warning("[DLAA] Texture model missing keys: %s", missing_keys)
                    
                if load_result.unexpected_keys:
                    log.warning("[DLAA] Texture model unexpected keys: %s", load_result.unexpected_keys)
                
                net = net.float()
                net.eval()
                
                n_params = sum(p.numel() for p in net.parameters())
                
                self.texture_net_cache[key] = net
                
                log.info("[DLAA] Loaded %s", os.path.basename(model_path))
                log.info(f"[DLAA] Texture model parameters: {n_params / 1e6:.2f}M")

                return net
                
            except Exception as e:
                if self._first_time("texture_load_error"):
                    import traceback
                    log.error(f"[DLAA] Texture model exception:\n{traceback.format_exc()}")
                    log.warning(
                        f"[DLAA] Texture model could not be loaded, skipping texture pass: {type(e).__name__}: {e!r}"
                    )
                return None

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
                dtype=dtype
            )

            w_y[:ramp] = values
            w_y[-ramp:] = torch.flip(values, dims=[0])

            w_x[:ramp] = values
            w_x[-ramp:] = torch.flip(values, dims=[0])

        return torch.minimum(
            w_y.view(1, 1, th, 1),
            w_x.view(1, 1, 1, tw)
        )

    def _tiled_forward(self, net, image: torch.Tensor, tile_size: int = 512, overlap: int = 32) -> torch.Tensor:

        tile_size = int(tile_size)
        overlap = int(overlap)

        if tile_size <= 0:
            raise ValueError(f"Invalid tile_size: {tile_size}")

        if overlap < 0:
            raise ValueError(f"Invalid overlap: {overlap}")

        if overlap * 2 >= tile_size:
            raise ValueError(
                f"Invalid tiling settings: tile_size={tile_size}, overlap={overlap}"
            )

        # Tile-based inference with border blending.
        B, C, H, W = image.shape

        if H <= tile_size and W <= tile_size:
            return torch.clamp(net(image), 0.0, 1.0)


        step = tile_size - overlap * 2


        if step <= 0:
            raise ValueError(
                f"Invalid tiling settings: tile_size={tile_size}, overlap={overlap}"
            )
            
        out = torch.zeros_like(image)
        weight = torch.zeros(B, 1, H, W, device=image.device, dtype=image.dtype)
        weight_cache = {}
        
        y0 = 0
        while y0 < H:
            y1    = min(y0 + tile_size, H)
            y0_c  = max(0, y1 - tile_size)

            x0 = 0
            while x0 < W:
                x1    = min(x0 + tile_size, W)
                x0_c  = max(0, x1 - tile_size)

                tile = image[:, :, y0_c:y1, x0_c:x1]
                dlaa_out_tile = net(tile)

                tile_h = tile.shape[2]
                tile_w = tile.shape[3]

                cache_key = (tile_h, tile_w, overlap, image.device, image.dtype)

                if cache_key not in weight_cache:
                    weight_cache[cache_key] = self._tile_weight_map(
                        tile_h,
                        tile_w,
                        overlap,
                        image.device,
                        image.dtype,
                    )

                w_map = weight_cache[cache_key]

                out[:, :, y0_c:y1, x0_c:x1]    += dlaa_out_tile * w_map
                weight[:, :, y0_c:y1, x0_c:x1] += w_map

                if x1 == W:
                    break
                x0 += step

            if y1 == H:
                break
            y0 += step

        return torch.clamp(out / weight.clamp(min=1e-6), 0.0, 1.0)

    def _jitter_count(self, net):
        offsets = getattr(net, "jitter_offsets", None)

        if offsets is None or offsets.shape[0] == 0:
            return 0

        return offsets.shape[0]

    def _jitter(self, image, idx, scale, net):
        if scale < 1e-5:
            return image

        offsets = getattr(net, "jitter_offsets", None)

        if offsets is None or offsets.shape[0] == 0:
            return image

        off = offsets[idx % offsets.shape[0]].to(device=image.device, dtype=image.dtype)

        B, C, H, W = image.shape

        theta = torch.eye(
            2,
            3,
            device=image.device,
            dtype=image.dtype,
        ).unsqueeze(0).repeat(B, 1, 1)

        theta[:, 0, 2] = off[0] * scale / W
        theta[:, 1, 2] = off[1] * scale / H

        grid = F.affine_grid(theta, image.shape, align_corners=False)

        return F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=False,
        )

    def _luma_edge(self, image, net):
        # luma + Sobel edge map
        luma = rgb_luma(image)
        sx = F.conv2d(luma, net.sobel_x, padding=1)
        sy = F.conv2d(luma, net.sobel_y, padding=1)
        edge = torch.sqrt(sx * sx + sy * sy + 1e-6)

        return luma, edge

    def _edge_aa(self, image, thr, blur_radius, net, strength=1.0):
        # edge AA
        strength = clamp01(strength)

        if blur_radius <= 0 or strength <= MIN_EFFECT_STRENGTH:
            return image

        _, edge = self._luma_edge(image, net)

        mask = torch.sigmoid((edge - thr) * self.edge_aa_slope)
        mask = mask * strength

        blurred = F.avg_pool2d(
            F.pad(image, [blur_radius] * 4, mode="reflect"),
            blur_radius * 2 + 1,
            stride=1,
        )

        return image * (1.0 - mask) + blurred * mask

    def _fine_line_aa(self, image, net, strength):
        # thin-line shimmer control
        strength = clamp01(strength)

        if strength <= MIN_EFFECT_STRENGTH:
            return image

        luma, edge = self._luma_edge(image, net)

        dark_mask = torch.sigmoid(
            (self.detail_fine_line_dark_threshold - luma) * FINE_LINE_DARK_SLOPE
        )

        edge_mask = torch.sigmoid(
            (edge - self.detail_fine_line_edge_threshold) * self.edge_aa_slope
        )

        local_avg = F.avg_pool2d(image, 3, stride=1, padding=1)
        fine_detail = rgb_luma((image - local_avg).abs())

        detail_mask = torch.sigmoid(
            (fine_detail - self.detail_shimmer_threshold) *
            self.detail_shimmer_slope
        )

        line_mask = (dark_mask * edge_mask * detail_mask).clamp(0.0, 1.0)

        blurred = F.avg_pool2d(
            F.pad(image, [1, 1, 1, 1], mode="reflect"),
            3,
            stride=1,
        )

        blur_strength = clamp01(self.detail_fine_line_blur_strength)

        aa_target = torch.lerp(
            image,
            blurred,
            blur_strength,
        )

        blend = (line_mask * strength).clamp(0.0, 1.0)

        return torch.lerp(image, aa_target, blend).clamp(0.0, 1.0)

    def _specular_detail(self, image, net, highlight_mask, strength):
        # controlled specular detail
        strength = clamp01(strength)

        if strength <= MIN_EFFECT_STRENGTH:
            return image

        luma, edge = self._luma_edge(image, net)

        local_avg = F.avg_pool2d(
            F.pad(image, [1, 1, 1, 1], mode="reflect"),
            3,
            stride=1,
        )

        spec_residual = (image - local_avg).clamp(min=0.0)
        spec_residual = spec_residual.clamp(max=self.detail_specular_limit)

        bright_mask = torch.sigmoid(
            (luma - self.detail_specular_threshold) *
            self.detail_specular_slope
        )

        clip_protect = 1.0 - torch.sigmoid(
            (luma - SPECULAR_CLIP_LUMA) * SPECULAR_CLIP_SLOPE
        )

        edge_mask = torch.sigmoid(
            (edge - SPECULAR_EDGE_THRESHOLD) * SPECULAR_EDGE_SLOPE
        )
        edge_mix = torch.lerp(
            torch.ones_like(edge_mask),
            edge_mask,
            self.detail_specular_edge_boost,
        )

        spec_mask = (bright_mask * clip_protect * edge_mix).clamp(0.0, 1.0)

        if highlight_mask is not None:
            spec_mask = spec_mask * (1.0 - highlight_mask * SPECULAR_HIGHLIGHT_SUPPRESSION)

        return (image + spec_residual * spec_mask * strength).clamp(0.0, 1.0)

    def _micro_contrast(self, image, highlight_mask, strength):
        strength = clamp01(strength)

        if strength <= MIN_EFFECT_STRENGTH:
            return image

        radius = int(self.detail_micro_contrast_radius)
        radius = max(3, radius)

        if radius % 2 == 0:
            radius += 1

        pad = radius // 2

        local_avg = F.avg_pool2d(
            F.pad(image, [pad, pad, pad, pad], mode="reflect"),
            radius,
            stride=1,
        )

        residual = image - local_avg
        residual = residual.clamp(
            -self.detail_micro_contrast_limit,
            self.detail_micro_contrast_limit,
        )

        luma = rgb_luma(image)
        detail_energy = rgb_luma(residual.abs())

        detail_mask = torch.sigmoid(
            (detail_energy - MICRO_DETAIL_THRESHOLD) * MICRO_DETAIL_SLOPE
        )

        if highlight_mask is not None:
            protect = 1.0 - highlight_mask * self.detail_micro_contrast_highlight_protect
        else:
            protect = (
                1.0 -
                torch.sigmoid(
                    (luma - MICRO_HIGHLIGHT_LUMA) * MICRO_HIGHLIGHT_SLOPE
                ) *
                self.detail_micro_contrast_highlight_protect
            )

        out = image + residual * detail_mask * protect * strength

        return torch.clamp(out, 0.0, 1.0)

    def _edge_dehalo(
        self,
        image: torch.Tensor,
        net,
        strength: float,
    ) -> torch.Tensor:
        strength = clamp01(strength)

        if strength <= MIN_EFFECT_STRENGTH:
            return image

        luma, edge = self._luma_edge(image, net)

        edge_mask = torch.sigmoid(
            (edge - self.detail_dehalo_threshold) * self.edge_sharp_slope
        )

        local_avg = F.avg_pool2d(
            F.pad(image, [2, 2, 2, 2], mode="reflect"),
            5,
            stride=1,
        )

        halo_residual = image - local_avg

        bright_halo = halo_residual.clamp(min=0.0)
        dark_halo = (-halo_residual).clamp(min=0.0)

        dark_protect = torch.sigmoid(
            (DEHALO_DARK_PROTECT_LUMA - luma) * DEHALO_PROTECT_SLOPE
        )
        light_protect = torch.sigmoid(
            (luma - DEHALO_LIGHT_PROTECT_LUMA) * DEHALO_PROTECT_SLOPE
        )

        bright_reduce = bright_halo * edge_mask * strength * (
            1.0 - light_protect * self.detail_dehalo_light_protect
        )

        dark_reduce = dark_halo * edge_mask * strength * (
            1.0 - dark_protect * self.detail_dehalo_dark_protect
        )

        out = image - bright_reduce + dark_reduce

        return torch.clamp(out, 0.0, 1.0)

    def _chroma_edge_cleanup(
        self,
        image: torch.Tensor,
        net,
        strength: float,
    ) -> torch.Tensor:
        strength = clamp01(strength)

        if strength <= MIN_EFFECT_STRENGTH:
            return image

        luma, edge = self._luma_edge(image, net)

        edge_mask = torch.sigmoid(
            (edge - self.detail_chroma_edge_threshold) * self.edge_aa_slope
        )

        chroma = image - luma

        chroma_blur = F.avg_pool2d(
            F.pad(chroma, [1, 1, 1, 1], mode="reflect"),
            3,
            stride=1,
        )

        chroma_delta = chroma_blur - chroma
        chroma_residual = rgb_luma(chroma_delta.abs())
        chroma_amount = rgb_luma(chroma.abs())

        chroma_mask = torch.sigmoid(
            (chroma_amount - self.detail_chroma_saturation_threshold) *
            CHROMA_SATURATION_SLOPE
        )

        fringe_mask = torch.sigmoid(
            (
                chroma_residual -
                self.detail_chroma_saturation_threshold * CHROMA_FRINGE_THRESHOLD_SCALE
            ) * CHROMA_FRINGE_SLOPE
        )

        dark_protect = torch.sigmoid(
            (luma - self.detail_chroma_dark_protect) * CHROMA_DARK_PROTECT_SLOPE
        )

        mask = (edge_mask * chroma_mask * fringe_mask * dark_protect).clamp(0.0, 1.0)

        chroma_delta = chroma_delta.clamp(
            -self.detail_chroma_cleanup_limit,
            self.detail_chroma_cleanup_limit,
        )

        clean_chroma = chroma + chroma_delta * mask * strength
        out = luma + clean_chroma

        # keep luma stable after chroma cleanup
        luma_error = rgb_luma(out) - luma
        out = out - luma_error

        return torch.clamp(out, 0.0, 1.0)

    def _subpixel_edge_reconstruction(
        self,
        image: torch.Tensor,
        net,
        motion_gate: float,
        strength: float,
    ) -> torch.Tensor:
        strength = clamp01(strength)

        if strength <= MIN_EFFECT_STRENGTH:
            return image
            
        luma = rgb_luma(image)

        sx = F.conv2d(luma, net.sobel_x, padding=1)
        sy = F.conv2d(luma, net.sobel_y, padding=1)
        edge = torch.sqrt(sx * sx + sy * sy + 1e-6)

        edge_mask = torch.sigmoid(
            (edge - self.detail_subpixel_edge_threshold) *
            self.detail_subpixel_edge_slope
        )

        B, C, H, W = image.shape
        
        # sample along the local edge normal
        nx = sx / edge.clamp(min=1e-6)
        ny = sy / edge.clamp(min=1e-6)

        sample_scale = float(self.detail_subpixel_sample_scale)

        offset_x = nx.squeeze(1) * sample_scale * (2.0 / max(W - 1, 1))
        offset_y = ny.squeeze(1) * sample_scale * (2.0 / max(H - 1, 1))

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=image.device, dtype=image.dtype),
            torch.linspace(-1.0, 1.0, W, device=image.device, dtype=image.dtype),
            indexing="ij",
        )

        base_grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

        offset = torch.stack((offset_x, offset_y), dim=-1)

        grid_pos = base_grid + offset
        grid_neg = base_grid - offset

        sample_pos = F.grid_sample(
            image,
            grid_pos,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=False,
        )

        sample_neg = F.grid_sample(
            image,
            grid_neg,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=False,
        )
        
        # limit the reconstruction delta
        reconstructed = (sample_pos + sample_neg) * 0.5

        delta = reconstructed - image
        delta = delta.clamp(
            -self.detail_subpixel_delta_limit,
            self.detail_subpixel_delta_limit,
        )

        stable_motion = 1.0 - min(
            max(float(motion_gate) * self.detail_subpixel_motion_protect, 0.0),
            1.0,
        )

        blend = (edge_mask * strength * stable_motion).clamp(
            0.0,
            self.detail_subpixel_blend_limit,
        )

        out = image + delta * blend

        return torch.clamp(out, 0.0, 1.0)

    def _temporal_specular_stabilizer(
        self,
        image: torch.Tensor,
        previous: torch.Tensor,
        motion_gate: float,
        strength: float,
    ) -> torch.Tensor:
        strength = clamp01(strength)

        if strength <= MIN_EFFECT_STRENGTH:
            return image

        if previous is None or previous.shape != image.shape:
            return image

        luma = rgb_luma(image)
        prev_luma = rgb_luma(previous)

        local_avg = F.avg_pool2d(
            F.pad(image, [1, 1, 1, 1], mode="reflect"),
            3,
            stride=1,
        )

        prev_local_avg = F.avg_pool2d(
            F.pad(previous, [1, 1, 1, 1], mode="reflect"),
            3,
            stride=1,
        )

        # stabilize bright high-frequency residuals
        current_spec = (image - local_avg).clamp(min=0.0)
        previous_spec = (previous - prev_local_avg).clamp(min=0.0)

        spec_energy = rgb_luma(current_spec)

        bright_mask = torch.sigmoid(
            (luma - self.detail_specular_temporal_threshold) *
            self.detail_specular_temporal_slope
        )

        spec_mask = torch.sigmoid(
            (spec_energy - self.detail_specular_temporal_detail_threshold) *
            TEMPORAL_SPEC_DETAIL_SLOPE
        )

        # avoid pulling moving highlights from history
        local_motion = torch.abs(luma - prev_luma)
        local_stable = 1.0 - torch.clamp(
            local_motion / TEMPORAL_SPEC_MOTION_SCALE,
            0.0,
            1.0,
        )

        global_stable = 1.0 - min(
            max(float(motion_gate) * self.detail_specular_temporal_motion_protect, 0.0),
            1.0,
        )

        blend = (
            bright_mask *
            spec_mask *
            local_stable *
            global_stable *
            strength
        ).clamp(0.0, self.detail_specular_temporal_blend_limit)

        stabilized_spec = torch.lerp(
            current_spec,
            previous_spec,
            blend,
        )

        delta = stabilized_spec - current_spec
        delta = delta.clamp(
            -self.detail_specular_temporal_delta_limit,
            self.detail_specular_temporal_delta_limit,
        )

        out = image + delta

        return torch.clamp(out, 0.0, 1.0)

    def _local_tone_mapping(
        self,
        image: torch.Tensor,
        highlight_mask: torch.Tensor,
        motion_gate: float,
        strength: float,
    ) -> torch.Tensor:
        # local luma tone adjust
        strength = clamp01(strength)

        if strength <= MIN_EFFECT_STRENGTH:
            return image

        radius = int(self.detail_local_tonemap_radius)
        radius = max(3, radius)

        if radius % 2 == 0:
            radius += 1

        pad = radius // 2

        luma = rgb_luma(image)

        local_avg = F.avg_pool2d(
            F.pad(luma, [pad, pad, pad, pad], mode="reflect"),
            radius,
            stride=1,
        )

        local_contrast = luma - local_avg
        local_contrast = local_contrast.clamp(
            -self.detail_local_tonemap_limit,
            self.detail_local_tonemap_limit,
        )

        if highlight_mask is not None:
            highlight_protect = 1.0 - highlight_mask * self.detail_local_tonemap_highlight_protect
        else:
            highlight_protect = (
                1.0 -
                torch.sigmoid(
                    (luma - LOCAL_TONEMAP_HIGHLIGHT_LUMA) *
                    LOCAL_TONEMAP_HIGHLIGHT_SLOPE
                ) *
                self.detail_local_tonemap_highlight_protect
            )

        shadow_mask = torch.sigmoid(
            (self.detail_local_tonemap_shadow_threshold - luma) *
            LOCAL_TONEMAP_SHADOW_SLOPE
        )

        shadow_lift = (
            shadow_mask *
            self.detail_local_tonemap_shadow_lift *
            highlight_protect
        )

        motion_stable = 1.0 - min(
            max(float(motion_gate) * self.detail_local_tonemap_motion_protect, 0.0),
            1.0,
        )

        luma_delta = (
            local_contrast * highlight_protect +
            shadow_lift
        ) * strength * motion_stable

        luma_delta = luma_delta.clamp(
            -self.detail_local_tonemap_limit,
            self.detail_local_tonemap_limit,
        )

        # preserve RGB balance with a luma ratio
        target_luma = (luma + luma_delta).clamp(0.0, 1.0)

        ratio = target_luma / luma.clamp(min=1e-6)
        ratio = ratio.clamp(LOCAL_TONEMAP_RATIO_MIN, LOCAL_TONEMAP_RATIO_MAX)
        out = image * ratio

        return torch.clamp(out, 0.0, 1.0)

    def _fur_hair_stabilizer(
        self,
        image: torch.Tensor,
        previous: torch.Tensor,
        net,
        motion_gate: float,
        strength: float,
    ) -> torch.Tensor:
        # stable fur/hair detail reuse
        strength = clamp01(strength)

        if strength <= MIN_EFFECT_STRENGTH:
            return image

        if previous is None or previous.shape != image.shape:
            return image

        _, edge = self._luma_edge(image, net)

        local_avg = F.avg_pool2d(
            F.pad(image, [1, 1, 1, 1], mode="reflect"),
            3,
            stride=1,
        )
        prev_local_avg = F.avg_pool2d(
            F.pad(previous, [1, 1, 1, 1], mode="reflect"),
            3,
            stride=1,
        )

        fine_detail = rgb_luma((image - local_avg).abs())
        prev_detail = previous - prev_local_avg
        current_detail = image - local_avg

        edge_mask = torch.sigmoid(
            (edge - self.detail_fur_edge_threshold) * self.edge_aa_slope
        )

        detail_mask = torch.sigmoid(
            (fine_detail - self.detail_fur_detail_threshold) * FUR_DETAIL_SLOPE
        )

        stable_motion = 1.0 - min(
            max(float(motion_gate) * self.detail_fur_motion_protect, 0.0),
            1.0,
        )

        fur_mask = (edge_mask * detail_mask * stable_motion).clamp(0.0, 1.0)

        blend = (fur_mask * strength).clamp(0.0, self.detail_fur_blend_limit)

        stabilized_detail = torch.lerp(
            current_detail,
            prev_detail,
            blend,
        )

        base = local_avg
        out = base + stabilized_detail

        return torch.clamp(out, 0.0, 1.0)

    def _temporal_refine(self, current, previous, strength=0.35, motion_threshold=0.08):
        # temporal shimmer control
        if previous is None:
            return current

        if previous.shape != current.shape:
            return current

        curr_luma = rgb_luma(current)
        prev_luma = rgb_luma(previous)

        motion = torch.abs(curr_luma - prev_luma)
        motion = torch.clamp(motion / motion_threshold, 0.0, 1.0)

        blend_mask = (1.0 - motion) * strength

        refined = current * (1.0 - blend_mask) + previous * blend_mask
        return refined.clamp(0.0, 1.0)

    def _stabilize_fine_detail(
        self,
        current,
        previous,
        strength,
        motion_threshold,
    ):
        # stable fine-detail reuse
        if previous is None:
            return current

        if previous.shape != current.shape:
            return current

        strength = clamp01(strength)

        if strength <= MIN_EFFECT_STRENGTH:
            return current

        current_base = F.avg_pool2d(current, 3, stride=1, padding=1)
        previous_base = F.avg_pool2d(previous, 3, stride=1, padding=1)

        current_detail = current - current_base
        previous_detail = previous - previous_base

        current_base_luma = rgb_luma(current_base)
        previous_base_luma = rgb_luma(previous_base)

        base_motion = torch.abs(current_base_luma - previous_base_luma)
        stable_area = 1.0 - torch.clamp(
            base_motion / motion_threshold,
            0.0,
            1.0,
        )

        detail_energy = rgb_luma(current_detail.abs())

        line_mask = torch.sigmoid(
            (detail_energy - self.detail_shimmer_threshold) *
            self.detail_shimmer_slope
        )

        blend_mask = (line_mask * stable_area * strength).clamp(
            0.0,
            self.detail_shimmer_max_blend,
        )

        stabilized_detail = torch.lerp(
            current_detail,
            previous_detail,
            blend_mask,
        )

        return torch.clamp(current_base + stabilized_detail, 0.0, 1.0)

    def _resolve_frame_config(self, preset, is_single_image, rgb, taa):
        if is_single_image:
            frame_cfg = PRESETS["Photo"]

        elif preset == "Auto":
            if taa.history is not None and taa.history.shape == rgb.shape:
                scene_motion = torch.abs(rgb - taa.history).mean().item()
            else:
                scene_motion = self.auto_default_scene_motion

            if scene_motion < self.auto_static_motion_threshold:
                frame_cfg = AUTO_STATIC
            elif scene_motion < self.auto_balanced_motion_threshold:
                frame_cfg = AUTO_BALANCED
            else:
                frame_cfg = AUTO_MOTION

        else:
            frame_cfg = PRESETS.get(preset, PRESETS["Balanced"])

        cfg = frame_cfg.copy()
        return cfg

    def _apply_texture_pass(
        self,
        texture_net,
        dlaa_out,
        tile_size,
        motion_gate,
        dark_mask,
        highlight_mask,
        preset,
        texture_intensity=1.00,
        motion_stability=1.00,
        frame_index=None
    ):
        if texture_net is None:
            return dlaa_out
            
        if texture_intensity <= 1e-5:
            return dlaa_out
            
        texture_cfg = self.texture_presets.get(preset, self.texture_presets["Balanced"])
        if not texture_cfg.get("enabled", True):
            return dlaa_out
            
        try:
            gen_out = self._tiled_forward(
                texture_net,
                dlaa_out,
                tile_size=tile_size,
                overlap=self.texture_tile_overlap,
            )

        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self._first_time("texture_oom"):
                log.warning("[DLAA] Texture pass ran out of VRAM, skipping texture pass.")

            return dlaa_out

        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self._first_time("texture_oom"):
                log.warning("[DLAA] Texture pass ran out of VRAM, skipping texture pass.")

            return dlaa_out
        
        if gen_out.shape != dlaa_out.shape:
            log.warning(
                "[DLAA] Texture model output size does not match input, skipping texture pass."
            )
            return dlaa_out
            
        gray = rgb_luma(dlaa_out)
        
        edge_x = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        edge_y = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
        
        edge_x = F.pad(edge_x, (0, 1, 0, 0))
        edge_y = F.pad(edge_y, (0, 0, 0, 1))
        
        thin_edge_mask = (edge_x + edge_y).clamp(0.0, 1.0)
        thin_edge_mask = torch.sigmoid(
            (thin_edge_mask - texture_cfg["edge_threshold"]) * texture_cfg["edge_slope"]
        )
        thin_edge_mask = F.avg_pool2d(thin_edge_mask, kernel_size=3, stride=1, padding=1)
        
        # isolate texture residual
        blur_kernel = int(texture_cfg.get("blur_kernel", 3))
        blur_kernel = max(3, blur_kernel)
        
        if blur_kernel % 2 == 0:
            blur_kernel += 1

        gen_blur = F.avg_pool2d(
            gen_out,
            kernel_size=blur_kernel,
            stride=1,
            padding=blur_kernel // 2,
        )
        texture_delta = gen_out - gen_blur
        
        debug_stats = log.isEnabledFor(logging.DEBUG)
        raw_gen_delta = 0.0
        if debug_stats:
            raw_gen_delta = texture_delta.abs().mean().item()
            
        dark_base = texture_cfg["dark_base"]
        texture_strength = texture_cfg["strength"] * texture_intensity
        texture_limit = texture_cfg["limit"] * texture_intensity
        motion_suppression = min(texture_cfg["motion_suppression"] * motion_stability, 0.98)
        line_suppression = min(texture_cfg["line_suppression"] * motion_stability, 0.95)

        texture_mask = dark_base + (1.0 - dark_base) * dark_mask
        texture_mask = texture_mask * (
            1.0 - highlight_mask * texture_cfg["highlight_suppression"]
        )
        texture_mask = texture_mask * (
            1.0 - motion_gate * motion_suppression
        )
        texture_mask = texture_mask * (
            1.0 - thin_edge_mask * line_suppression
        )
        
        texture_delta = texture_delta * texture_mask
        texture_delta = texture_delta.clamp(
            -texture_limit,
            texture_limit
        )
        
        out = dlaa_out + texture_delta * texture_strength
        
        final_texture_delta = 0.0
        
        if debug_stats:
            final_texture_delta = (out - dlaa_out).abs().mean().item()
            
        if (
            debug_stats and
            frame_index is not None and
            self.texture_log_interval > 0 and
            frame_index % self.texture_log_interval == 0
        ):
            log.debug(
                "[DLAA] texture frame=%d raw_delta=%.6f final_delta=%.6f",
                frame_index,
                raw_gen_delta,
                final_texture_delta
            )
            
        return torch.clamp(out, 0.0, 1.0)

    def _normalize_run_inputs(
        self,
        preset,
        dlaa_intensity,
        texture_intensity,
        motion_stability,
    ):
        # Preset aliases and old workflow names
        preset_aliases = {
            "High Detail": "Detail",
            "Sharp": "Detail",
            "Cinematic": "Smooth",
        }

        preset = preset_aliases.get(preset, preset)

        dlaa_intensity = max(0.0, min(float(dlaa_intensity), 2.0))
        texture_intensity = max(0.0, min(float(texture_intensity), 2.0))
        motion_stability = max(0.5, min(float(motion_stability), 2.0))

        return preset, dlaa_intensity, texture_intensity, motion_stability

    def _get_device(self):
        if mm is not None:
            return mm.get_torch_device()
            
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _frame_blur_radius(self, preset, is_single_image):
        if is_single_image:
            if preset == "Photo":
                return 1
            return 0

        return 1

    def _frame_edge_aa_strength(self, preset, is_single_image):
        if is_single_image:
            if preset == "Photo":
                return self.photo_edge_aa_strength
            return 0.0

        if preset == "Detail":
            return self.detail_edge_aa_strength

        if preset == "Photo":
            return self.photo_edge_aa_strength

        return 1.0

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
        # safe fallback for CPU/unknown VRAM
        if vram_mb <= 0:
            return 512
            
        if vram_mb <= 8192:
            return 512
            
        if vram_mb <= 16384:
            return 1024
            
        return 1536

    def _log_tiling(self, height, width, tile_size):
        if height <= tile_size and width <= tile_size:
            return
            
        tile_step = tile_size - 64
        tile_count = ((height + tile_step - 1) // tile_step) * ((width + tile_step - 1) // tile_step)
        log.debug(f"[DLAA] Tiled inference: {tile_count} tiles")

    def _load_frame(self, images, frame_index, device):
        img = images[frame_index:frame_index + 1].to(device).permute(0, 3, 1, 2).float()
        return img[:, :3]

    def _texture_net_for_run(self, device, preset, texture_intensity):
        texture_cfg = self.texture_presets.get(preset, self.texture_presets["Balanced"])
        
        texture_enabled = getattr(
            self,
            "texture_pass_enabled",
            NODE_DEFAULTS.get("texture_pass_enabled", True)
        )

        if not texture_enabled:
            return None

        if not texture_cfg.get("enabled", True):
            return None

        if texture_intensity <= 1e-5:
            return None

        return self._texture_net(device)

    def _frame_params(self, frame_cfg, dlaa_intensity):
        params = frame_cfg.copy()

        scaled_params = (
            "detail_boost",
            "edge_boost",
            "edge_sharp_strength",
            "micro_limit",
        )

        for name in scaled_params:
            params[name] *= dlaa_intensity

        return params

    def _apply_jitter_and_taa(
        self,
        rgb,
        taa,
        net,
        motion_sensitivity,
        preset_jitter_scale,
        taa_strength,
        blur_radius,
        edge_aa_strength,
    ):
        motion_gate = 0.0
        fid = taa.frame_id
        jitter_count = self._jitter_count(net)

        if jitter_count > 0:
            taa.frame_id = (fid + 1) % jitter_count
        else:
            taa.frame_id = fid + 1
        
        # damp jitter on motion
        adaptive_jitter_scale = preset_jitter_scale
        
        if taa.history is not None and taa.history.shape == rgb.shape:
            motion_estimate = torch.abs(rgb - taa.history).mean()
            motion_value = float(motion_estimate.item())
            
            motion_gate = min(
                max(motion_value * self.motion_gate_scale, 0.0),
                1.0,
            )
            
            jitter_damping = min(
                max(1.0 - motion_value * self.jitter_motion_damping, JITTER_DAMPING_MIN),
                1.0,
            )
            
            adaptive_jitter_scale = preset_jitter_scale * jitter_damping
            
        rgb = self._jitter(rgb, fid, adaptive_jitter_scale, net)
        rgb = self._edge_aa(
            rgb,
            self.edge_threshold,
            blur_radius,
            net,
            edge_aa_strength,
        )
        
        taa_out = taa.update(rgb, self.taa_alpha, motion_sensitivity)
        rgb = torch.lerp(rgb, taa_out, taa_strength)
        
        return rgb, motion_gate

    def _run_dlaa_with_retry(self, net, rgb, tile_size, debug_stats, frame_index):
        tile = tile_size
        min_tile_size = 128
        dlaa_out = None
        last_oom_error = None
        
        while tile >= min_tile_size:
            try:
                dlaa_out = self._tiled_forward(
                    net,
                    rgb,
                    tile_size=tile,
                    overlap=32,
                )
                
                if debug_stats and frame_index == 0:
                    raw_model_delta = (dlaa_out - rgb).abs().mean().item()
                    log.debug(f"[DLAA] raw_model_delta={raw_model_delta:.6f}")
                    
                break
                
            except torch.OutOfMemoryError as e:
                last_oom_error = e

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                next_tile_size = tile // 2
                
                if next_tile_size < min_tile_size:
                    log.error(
                        "DLAA tiled inference failed at minimum tile size %d.",
                        tile,
                    )
                    raise last_oom_error
                    
                log.warning(
                    "Out of VRAM at tile size %d, retrying with %d.",
                    tile,
                    next_tile_size,
                )
                
                tile = next_tile_size
                
        if dlaa_out is None:
            raise RuntimeError("DLAA tiled inference failed.")
            
        return dlaa_out, tile

    def _apply_model_residual(
        self,
        rgb,
        dlaa_out,
        preset,
        debug_stats,
        frame_index,
    ):
        # match output mean before blending
        dlaa_mean = dlaa_out.mean(dim=(1, 2, 3), keepdim=True)
        rgb_mean = rgb.mean(dim=(1, 2, 3), keepdim=True)
        dlaa_out = dlaa_out - dlaa_mean + rgb_mean
        
        preset_model_weight = self.preset_model_weight.get(preset, 1.00)
        
        model_delta = dlaa_out - rgb
        
        model_delta_value = None
        if debug_stats:
            model_delta_value = model_delta.abs().mean().item()
            
        dlaa_out = torch.clamp(
            rgb + model_delta * self.model_weight * preset_model_weight,
            0.0,
            1.0,
        )
        
        if debug_stats and frame_index == 0:
            log.debug(f"[DLAA] model_delta_first={model_delta_value:.6f}")
            
        return dlaa_out, model_delta_value

    def _apply_highlight_preblend(self, dlaa_out, rgb, preset):
        luma = rgb_luma(dlaa_out)
        highlight_mask = torch.sigmoid((luma - self.highlight_threshold) * self.highlight_slope)
        
        highlight_pre_blend = self.highlight_pre_blend * (
            self.detail_highlight_pre_scale if preset == "Detail" else 1.0
        )
        dlaa_out = torch.lerp(dlaa_out, rgb, highlight_mask * highlight_pre_blend)
        dlaa_out = torch.clamp(dlaa_out, 0.0, 1.0)
        
        return dlaa_out, highlight_mask

    def _apply_motion_suppression(
        self,
        preset,
        detail_boost,
        edge_boost,
        micro_limit,
        motion_gate,
        motion_stability,
    ):
        motion_cfg = MOTION_SUPPRESSION["Detail"] if preset == "Detail" else MOTION_SUPPRESSION["Default"]
        
        detail_boost *= (1.0 - motion_gate * motion_cfg["detail"] * motion_stability)
        edge_boost *= (1.0 - motion_gate * motion_cfg["edge"] * motion_stability)
        micro_limit *= (1.0 - motion_gate * motion_cfg["micro"] * motion_stability)
        
        return detail_boost, edge_boost, micro_limit

    def _apply_detail_pass(
        self,
        dlaa_out,
        net,
        preset,
        detail_boost,
        edge_boost,
        micro_limit,
        edge_sharp_strength,
        highlight_mask,
    ):
        
        local_avg_rgb = F.avg_pool2d(dlaa_out, 3, stride=1, padding=1)
        fine_detail_rgb = dlaa_out - local_avg_rgb
        
        luma = rgb_luma(dlaa_out)
        fine_detail = rgb_luma(fine_detail_rgb)
        
        texture_dark_mask = torch.clamp(
            (luma - self.detail_dark_luma_start) / self.detail_dark_luma_range,
            0.0,
            1.0,
        )
        
        if preset == "Detail":
            dark_mask = texture_dark_mask
            micro_limit = micro_limit * (
                self.detail_dark_mix_base + self.detail_dark_mix_scale * dark_mask
            )
        else:
            dark_mask = 1.0
            
        detail_strength = fine_detail.abs().mean(dim=(1, 2, 3), keepdim=True)
        detail_scale = (
            self.detail_base_scale + (self.detail_ref_scale / (detail_strength + 1e-6))
        ).clamp(self.detail_min_scale, self.detail_max_scale)
        
        fine_detail = fine_detail * torch.sigmoid(fine_detail * detail_scale)
        fine_detail = fine_detail.clamp(-self.fine_detail_limit, self.fine_detail_limit)
        
        local_detail = F.avg_pool2d(fine_detail.abs(), 7, stride=1, padding=3)
        global_detail = fine_detail.abs().mean(dim=(1, 2, 3), keepdim=True)
        
        edge = torch.sqrt(
            F.conv2d(luma, net.sobel_x, padding=1) ** 2 +
            F.conv2d(luma, net.sobel_y, padding=1) ** 2 +
            1e-6
        )
        
        edge_detail_weight = torch.sigmoid(
            (edge - self.edge_sharp_threshold) * self.edge_sharp_slope
        )
        
        detail_gain = (global_detail / (local_detail + 1e-6)).clamp(
            self.detail_min_gain,
            self.detail_max_gain,
        )
        detail_gain = detail_gain * (1.0 - local_detail.clamp(0.0, 0.5))
        detail_gain = detail_gain * (1.0 + edge_detail_weight * self.detail_edge_boost)
        detail_gain = detail_gain * (1.0 - highlight_mask * self.detail_highlight_suppression)
        
        micro_detail = fine_detail_rgb * detail_gain * detail_boost * dark_mask
        micro_detail = micro_detail.clamp(-micro_limit, micro_limit)
        dlaa_out = dlaa_out + micro_detail
        
        edge_mask = torch.sigmoid(
            (edge - self.edge_sharp_threshold) * self.edge_sharp_slope
        )
        edge_detail = fine_detail_rgb * edge_mask * (1.0 - highlight_mask)
        
        edge_boosted = edge_detail * edge_sharp_strength * edge_boost * dark_mask
        edge_boosted = edge_boosted.clamp(
            -micro_limit * self.edge_detail_limit_scale,
            micro_limit * self.edge_detail_limit_scale,
        )
        dlaa_out = dlaa_out + edge_boosted
        
        return dlaa_out, texture_dark_mask

    def _apply_tone_and_color_pass(
        self,
        dlaa_out,
        rgb,
        preset,
        highlight_mask,
        tone_strength,
        luma_boost_mult,
        saturation_boost_mult,
    ):
        tone_mapped = dlaa_out / (dlaa_out + self.tone_curve_bias)
        dlaa_out = torch.lerp(dlaa_out, tone_mapped, highlight_mask * tone_strength)
        
        luma = rgb_luma(dlaa_out)
        luma_boost = (
            self.luma_boost_base *
            luma_boost_mult *
            (1.0 - highlight_mask * self.luma_highlight_protect)
        )
        dlaa_out = dlaa_out * (1.0 + luma_boost)
        
        mean_rgb = dlaa_out.mean(dim=1, keepdim=True)
        saturation_boost = (
            self.saturation_boost_base *
            saturation_boost_mult *
            (1.0 - highlight_mask * self.saturation_highlight_protect)
        )
        dlaa_out = mean_rgb + (dlaa_out - mean_rgb) * (1.0 + saturation_boost)

        highlight_post_blend = self.highlight_post_blend * (
            self.detail_highlight_post_scale if preset == "Detail" else 1.0
        )
        dlaa_out = torch.lerp(dlaa_out, rgb, highlight_mask * highlight_post_blend)
        
        return torch.clamp(dlaa_out, 0.0, 1.0)

    def _apply_final_temporal_and_blend(
        self,
        rgb,
        dlaa_out,
        prev_dlaa_output,
        preset,
        temporal_strength,
        motion_threshold,
        dlaa_strength,
    ):
        if prev_dlaa_output is not None:
            if prev_dlaa_output.shape != dlaa_out.shape:
                prev_dlaa_output = None

        if preset == "Detail":
            dlaa_out = self._stabilize_fine_detail(
                dlaa_out,
                prev_dlaa_output,
                strength=self.detail_shimmer_strength,
                motion_threshold=motion_threshold,
            )

        dlaa_out = self._temporal_refine(
            dlaa_out,
            prev_dlaa_output,
            strength=temporal_strength,
            motion_threshold=motion_threshold,
        )
        
        prev_dlaa_output = dlaa_out.detach()
        dlaa_out = torch.clamp(dlaa_out, 0.0, 1.0)
        
        blend_weight = dlaa_strength * self.dlaa_blend_scale
        
        if preset == "Detail":
            blend_weight = min(blend_weight * self.detail_blend_boost, 1.0)
            
        rgb = torch.lerp(rgb, dlaa_out, blend_weight)
        
        return rgb, prev_dlaa_output

    def _apply_dlaa_pipeline(
        self,
        rgb,
        net,
        texture_net,
        preset,
        tile_size,
        motion_gate,
        detail_boost,
        edge_boost,
        temporal_strength,
        micro_limit,
        luma_boost_mult,
        saturation_boost_mult,
        motion_threshold,
        dlaa_strength,
        tone_strength,
        edge_sharp_strength,
        motion_stability,
        texture_intensity,
        prev_dlaa_output,
        debug_stats,
        frame_index,
    ):
        # main DLAA pipeline
        
        if dlaa_strength <= 0.0:
            return rgb, prev_dlaa_output, None

        is_detail = preset == "Detail"
        is_performance = preset == "Performance"
        perf_scale = 0.75 if is_performance else 1.0

        dlaa_out, tile = self._run_dlaa_with_retry(
            net,
            rgb,
            tile_size,
            debug_stats,
            frame_index,
        )
        
        dlaa_out, model_delta_value = self._apply_model_residual(
            rgb,
            dlaa_out,
            preset,
            debug_stats,
            frame_index,
        )
        
        dlaa_out, highlight_mask = self._apply_highlight_preblend(
            dlaa_out,
            rgb,
            preset,
        )
        
        detail_boost, edge_boost, micro_limit = self._apply_motion_suppression(
            preset,
            detail_boost,
            edge_boost,
            micro_limit,
            motion_gate,
            motion_stability,
        )
        
        if preset == "Photo":
            luma = rgb_luma(dlaa_out)
            
            texture_dark_mask = torch.clamp(
                (luma - self.detail_dark_luma_start) / self.detail_dark_luma_range,
                0.0,
                1.0,
            )
        else:
            dlaa_out, texture_dark_mask = self._apply_detail_pass(
                dlaa_out,
                net,
                preset,
                detail_boost,
                edge_boost,
                micro_limit,
                edge_sharp_strength,
                highlight_mask,
            )
            
        if is_detail or is_performance:
            dlaa_out = self._fine_line_aa(
                dlaa_out,
                net,
                self.detail_fine_line_aa_strength * perf_scale,
            )
            
        dlaa_out = self._apply_texture_pass(
            texture_net=texture_net,
            dlaa_out=dlaa_out,
            tile_size=tile,
            motion_gate=motion_gate,
            dark_mask=texture_dark_mask,
            highlight_mask=highlight_mask,
            preset=preset,
            texture_intensity=texture_intensity,
            motion_stability=motion_stability,
            frame_index=frame_index,
        )
        
        if is_detail:
            dlaa_out = self._specular_detail(
                dlaa_out,
                net,
                highlight_mask,
                self.detail_specular_strength,
            )

        if is_detail or is_performance:
            dlaa_out = self._micro_contrast(
                dlaa_out,
                highlight_mask,
                self.detail_micro_contrast_strength * perf_scale,
            )

            dlaa_out = self._edge_dehalo(
                dlaa_out,
                net,
                self.detail_dehalo_strength * perf_scale,
            )

            dlaa_out = self._chroma_edge_cleanup(
                dlaa_out,
                net,
                self.detail_chroma_cleanup_strength * perf_scale,
            )

        if is_detail:
            dlaa_out = self._subpixel_edge_reconstruction(
                dlaa_out,
                net,
                motion_gate,
                self.detail_subpixel_edge_strength,
            )

            dlaa_out = self._local_tone_mapping(
                dlaa_out,
                highlight_mask,
                motion_gate,
                self.detail_local_tonemap_strength,
            )
            
        dlaa_out = self._apply_tone_and_color_pass(
            dlaa_out,
            rgb,
            preset,
            highlight_mask,
            tone_strength,
            luma_boost_mult,
            saturation_boost_mult,
        )
        
        if preset == "Detail":
            dlaa_out = self._temporal_specular_stabilizer(
                dlaa_out,
                prev_dlaa_output,
                motion_gate,
                self.detail_specular_temporal_strength,
            )
            
            dlaa_out = self._fur_hair_stabilizer(
                dlaa_out,
                prev_dlaa_output,
                net,
                motion_gate,
                self.detail_fur_stabilizer_strength,
            )
            
        rgb, prev_dlaa_output = self._apply_final_temporal_and_blend(
            rgb,
            dlaa_out,
            prev_dlaa_output,
            preset,
            temporal_strength,
            motion_threshold,
            dlaa_strength,
        )
        
        return rgb, prev_dlaa_output, model_delta_value

    # Main node entry
    def execute(
        self,
        images,
        preset,
        dlaa_intensity=1.00,
        texture_intensity=1.00,
        motion_stability=1.00,
        detail_intensity=None,
        **kwargs,
    ):
        # Legacy workflow fallback
        if detail_intensity is not None:
            dlaa_intensity = detail_intensity
        preset, dlaa_intensity, texture_intensity, motion_stability = self._normalize_run_inputs(
            preset,
            dlaa_intensity,
            texture_intensity,
            motion_stability,
        )
        
        device = self._get_device()
        
        B, H, W, C = images.shape
        is_single_image = (B == 1)
        run_preset = "Photo" if is_single_image else preset
        
        # process RGB, preserve extra channels
        extra_channels = None
        if C > 3:
            extra_channels = images[:, :, :, 3:].detach().cpu()

        out_channels = 3 if extra_channels is None else 3 + extra_channels.shape[-1]
        
        blur_radius = self._frame_blur_radius(run_preset, is_single_image)
        edge_aa_strength = self._frame_edge_aa_strength(
            run_preset,
            is_single_image,
        )
        
        # per-run temporal state
        taa = TAAState()
        prev_dlaa_output = None
        
        net = self._net(device)
        texture_net = self._texture_net_for_run(device, run_preset, texture_intensity)
        
        out_tensor = torch.empty((B, H, W, out_channels), dtype=images.dtype, device="cpu")
        delta_sum = 0.0
        delta_count = 0
        debug_stats = log.isEnabledFor(logging.DEBUG)
        
        vram_mb = self._vram_mb(device)
        tile_size = self._tile_size_for_vram(vram_mb)

        if debug_stats:
            log.debug(
                "[DLAA] device=%s vram=%dMB tile_size=%d",
                device,
                vram_mb,
                tile_size,
            )

        self._log_tiling(H, W, tile_size)
        
        progress = ProgressBar(B) if ProgressBar is not None else None
        
        with torch.inference_mode():
            for i in range(B):
                # allow ComfyUI interruption
                if mm is not None and hasattr(mm, "throw_exception_if_processing_interrupted"):
                    mm.throw_exception_if_processing_interrupted()

                rgb = self._load_frame(images, i, device)
                
                frame_cfg = self._resolve_frame_config(
                    run_preset,
                    is_single_image,
                    rgb,
                    taa,
                )
                params = self._frame_params(frame_cfg, dlaa_intensity)
                
                rgb, motion_gate = self._apply_jitter_and_taa(
                    rgb,
                    taa,
                    net,
                    params["motion_sensitivity"],
                    params["jitter_scale"],
                    params["taa_strength"],
                    blur_radius,
                    edge_aa_strength,
                )
                
                rgb, prev_dlaa_output, model_delta_value = self._apply_dlaa_pipeline(
                    rgb=rgb,
                    net=net,
                    texture_net=texture_net,
                    preset=run_preset,
                    tile_size=tile_size,
                    motion_gate=motion_gate,
                    detail_boost=params["detail_boost"],
                    edge_boost=params["edge_boost"],
                    temporal_strength=params["temporal_strength"],
                    micro_limit=params["micro_limit"],
                    luma_boost_mult=params["luma_boost_mult"],
                    saturation_boost_mult=params["saturation_boost_mult"],
                    motion_threshold=params["motion_threshold"],
                    dlaa_strength=params["dlaa_strength"],
                    tone_strength=params["tone_strength"],
                    edge_sharp_strength=params["edge_sharp_strength"],
                    motion_stability=motion_stability,
                    texture_intensity=texture_intensity,
                    prev_dlaa_output=prev_dlaa_output,
                    debug_stats=debug_stats,
                    frame_index=i,
                )
                
                if debug_stats and model_delta_value is not None:
                    delta_sum += model_delta_value
                    delta_count += 1
                    
                rgb = torch.clamp(rgb, 0.0, 1.0)
                frame_out = rgb.permute(0, 2, 3, 1).detach().cpu()

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

        if debug_stats and delta_count > 0:
            log.debug(f"[DLAA] model_delta_avg={delta_sum / delta_count:.6f}")

        return (out_tensor,)


NODE_CLASS_MAPPINGS        = {"VideoTAADLAA": VideoTAADLAA}
NODE_DISPLAY_NAME_MAPPINGS = {"VideoTAADLAA": "🎮 Video Anti-Aliasing (TAA + DLAA)"}