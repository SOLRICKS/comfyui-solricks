# Video TAA + DLAA Inference
# v0.2.2
# SOLRICKS
# AI-assisted


from __future__ import annotations

import logging
import math
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from safetensors import SafetensorError
from safetensors.torch import load_file

try:
    from .config import INTERNAL_TUNING, NODE_DEFAULTS
    from .model import DLAANet
    from .presets import (
        AUTO_BALANCED,
        AUTO_MOTION,
        AUTO_STATIC,
        MOTION_SUPPRESSION,
        PRESET_MODEL_WEIGHT,
        PRESETS,
        TEXTURE_PRESETS,
    )
    from .taa import TAAState
    from .utils import LUMA_WEIGHTS, clamp01
except ImportError:
    from config import INTERNAL_TUNING, NODE_DEFAULTS
    from model import DLAANet
    from presets import (
        AUTO_BALANCED,
        AUTO_MOTION,
        AUTO_STATIC,
        MOTION_SUPPRESSION,
        PRESET_MODEL_WEIGHT,
        PRESETS,
        TEXTURE_PRESETS,
    )
    from taa import TAAState
    from utils import LUMA_WEIGHTS, clamp01

try:
    import comfy.model_management as mm
    import comfy.model_patcher as model_patcher
except ImportError:
    mm = None
    model_patcher = None

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None


log = logging.getLogger("VideoTAADLAA")


def rgb_luma(image: torch.Tensor) -> torch.Tensor:
    r, g, b = LUMA_WEIGHTS
    luma = torch.add(image[:, 0:1] * r, image[:, 1:2], alpha=g)
    return torch.add(luma, image[:, 2:3], alpha=b)


def load_checkpoint(stem: str) -> DLAANet:
    directory = Path(__file__).resolve().parent
    path = directory / f"{stem}.safetensors"
    if path.is_file():
        weights = load_file(str(path), device="cpu")
    else:
        path = directory / f"{stem}.pth"
        if not path.is_file():
            raise FileNotFoundError(
                f"[DLAA] Expected {stem}.safetensors or {stem}.pth in {directory}"
            )
        weights = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(weights, Mapping):
            for key in ("state_dict", "params_ema", "params", "model"):
                if isinstance(weights.get(key), Mapping):
                    weights = weights[key]
                    break
    if not isinstance(weights, Mapping):
        raise ValueError(f"[DLAA] {path.name} must contain a model state dictionary")
    net = DLAANet().float().cpu()
    # Older checkpoints have a different jitter table. This buffer is not a learned weight.
    state = dict(weights)
    state["jitter_offsets"] = net.jitter_offsets
    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(
            f"[DLAA] Incompatible weights in {path.name}: {error}"
        ) from error
    log.info("[DLAA] Loaded %s", path.name)
    return net.eval()


SETTINGS = MappingProxyType(
    {
        "tile_overlap": 32,
        "scene_cut_min_change": 0.18,
        **NODE_DEFAULTS,
        **INTERNAL_TUNING,
    }
)


@dataclass(frozen=True)
class FrameParams:
    preset: str
    detail_boost: float
    edge_boost: float
    temporal_strength: float
    micro_limit: float
    luma_boost_mult: float
    saturation_boost_mult: float
    motion_threshold: float
    taa_strength: float
    dlaa_strength: float
    tone_strength: float
    edge_sharp_strength: float
    motion_sensitivity: float


@dataclass
class RunContext:
    tile_size: int
    tile_overlap: int
    motion_stability: float
    texture_intensity: float
    debug_stats: bool
    sobel_kernel: torch.Tensor
    grid_cache: dict = field(default_factory=dict)


@dataclass
class ManagedModel:
    net: DLAANet
    patcher: Optional[Any]


@dataclass(frozen=True)
class ClipHistory:
    source: torch.Tensor
    taa: torch.Tensor
    detail: torch.Tensor
    frames: int
    tile_size: int
    settings: tuple[str, float, float, float]

    def on(self, device: torch.device | str, *, copy: bool = False) -> ClipHistory:
        def move(frame: torch.Tensor) -> torch.Tensor:
            return frame.detach().to(
                device=device, copy=copy, memory_format=torch.contiguous_format
            )

        return ClipHistory(
            move(self.source),
            move(self.taa),
            move(self.detail),
            self.frames,
            self.tile_size,
            self.settings,
        )


@dataclass(frozen=True)
class CameraMotion:
    amount: float
    dx: float = 0.0
    dy: float = 0.0
    matched: bool = False


@dataclass(frozen=True)
class AlignedHistory:
    taa: torch.Tensor
    detail: torch.Tensor
    confidence: torch.Tensor


def pixel_grid(image: torch.Tensor, cache: Optional[dict]) -> torch.Tensor:
    b, _, h, w = image.shape
    key = ("pixels", h, w, image.device, image.dtype)
    if cache is not None and key in cache:
        grid = cache[key]
    else:
        yy, xx = torch.meshgrid(
            (torch.arange(h, device=image.device, dtype=image.dtype) + 0.5) * (2.0 / h)
            - 1,
            (torch.arange(w, device=image.device, dtype=image.dtype) + 0.5) * (2.0 / w)
            - 1,
            indexing="ij",
        )
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
        if cache is not None:
            cache[key] = grid
    return grid.expand(b, -1, -1, -1)


def refine_translation(
    current: torch.Tensor,
    previous: torch.Tensor,
    offset: torch.Tensor,
    scale: float,
    cache: dict,
) -> torch.Tensor:
    """Fit the thumbnail displacement to sparse samples at the source resolution."""
    h, w = current.shape[-2:]
    key = ("pan_samples", h, w, current.device, current.dtype)
    if key not in cache:

        def axis(length: int) -> torch.Tensor:
            count = min(length, 32)
            return (
                (torch.arange(count, device=current.device) + 0.5) * (length / count)
            ).long()

        y, x = torch.meshgrid(axis(h), axis(w), indexing="ij")
        units = current.new_tensor((2 / w, 2 / h))
        grid = (torch.stack((x, y), -1) + 0.5) * units - 1
        radius = min(8, math.ceil(scale / 2))
        whole = torch.arange(
            -radius, radius + 1, device=current.device, dtype=current.dtype
        )
        fraction = torch.linspace(
            -0.5, 0.5, 9, device=current.device, dtype=current.dtype
        )
        cache[key] = (
            y,
            x,
            units,
            grid,
            torch.cartesian_prod(whole, whole),
            torch.cartesian_prod(fraction, fraction),
        )
    y, x, units, grid, whole, fraction = cache[key]
    samples = current[0, :, y, x].unsqueeze(1)

    def best(center: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        shifts = center + offsets
        grids = grid - (shifts * units)[:, None, None, :]
        values = F.grid_sample(previous, grids.flatten(0, 1)[None], align_corners=False)
        values = values.reshape(3, *grids.shape[:3])
        valid = (grids.abs() <= 1 - units * 0.5 + 1e-6).all(dim=-1)
        count = valid.sum((-2, -1))
        # A moving foreground object should not dominate the camera fit.
        error = (values - samples).abs().mean(0).clamp_max(0.12)
        cost = (error * valid).sum((-2, -1)) / count.clamp_min(1)
        cost = cost.masked_fill(count == 0, torch.inf) + offsets.square().sum(-1) * 1e-7
        return shifts.index_select(0, cost.argmin().view(1))[0]

    center = offset.round()
    if scale > 1:
        center = best(center, whole)
    return best(center, fraction)


def measure_camera_motion(
    current: torch.Tensor, previous: torch.Tensor, cache: dict
) -> CameraMotion:
    """Estimate camera translation by phase correlation of the two source frames."""
    raw_change = (current - previous).abs().mean()
    h, w = current.shape[-2:]
    scale = min(1.0, 512 / max(h, w))
    sh, sw = max(1, round(h * scale)), max(1, round(w * scale))
    if min(sh, sw) < 4:
        amount = float(raw_change.item())
        return CameraMotion(amount, matched=amount == 0)

    def thumbnail(frame: torch.Tensor) -> torch.Tensor:
        if (sh, sw) != (h, w):
            frame = F.interpolate(frame, size=(sh, sw), mode="area")
        return rgb_luma(frame)[0, 0]

    pair = torch.stack((thumbnail(current), thumbnail(previous)))
    # Use the CPU Fourier transform for backends with limited FFT support.
    if pair.device.type not in {"cpu", "cuda"}:
        pair = pair.cpu()
    # The taper keeps newly exposed borders from determining the displacement.
    key = ("hann", sh, sw, pair.device, pair.dtype)
    if key not in cache:
        wy = torch.hann_window(sh, periodic=False, device=pair.device, dtype=pair.dtype)
        wx = torch.hann_window(sw, periodic=False, device=pair.device, dtype=pair.dtype)
        cache[key] = wy[:, None] * wx[None, :]
    spectrum = torch.fft.rfft2((pair - pair.mean((-2, -1), keepdim=True)) * cache[key])
    cross = spectrum[0] * spectrum[1].conj()
    corr = torch.fft.irfft2(cross / cross.abs().clamp_min(1e-8), s=(sh, sw))
    flat = corr.flatten()
    index = flat.argmax().view(1)
    y, x = index // sw, index % sw
    peak = flat.gather(0, index)

    def at(py: torch.Tensor, px: torch.Tensor) -> torch.Tensor:
        return flat.gather(0, (py % sh) * sw + (px % sw))

    def subpixel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        curvature = (left + right - 2 * peak).clamp_max(-1e-8)
        return (0.5 * (left - right) / curvature).clamp(-0.5, 0.5)

    dx = (
        torch.where(x > sw // 2, x - sw, x) + subpixel(at(y, x - 1), at(y, x + 1))
    ) * (w / sw)
    dy = (
        torch.where(y > sh // 2, y - sh, y) + subpixel(at(y - 1, x), at(y + 1, x))
    ) * (h / sh)
    displacement = refine_translation(
        current,
        previous,
        torch.cat((dx, dy)).to(current.device),
        max(h / sh, w / sw),
        cache,
    )
    rms = corr.square().mean().sqrt()
    # Auto and the temporal filters share one transfer from the device.
    scores = torch.stack((peak[0], rms)).to(current.device)
    amount, dx, dy, peak, rms = torch.cat(
        (raw_change.view(1), displacement, scores)
    ).tolist()
    if amount == 0.0:
        return CameraMotion(0.0, matched=True)
    matched = peak > max(0.10, 5.5 * rms) and abs(dx) < w / 3 and abs(dy) < h / 3
    return CameraMotion(amount, dx, dy, matched)


def align_history(
    current: torch.Tensor,
    history: ClipHistory,
    motion: CameraMotion,
    sensitivity: float,
    cache: dict,
) -> Optional[AlignedHistory]:
    if not motion.matched:
        return None
    h, w = current.shape[-2:]
    source, taa, detail = history.source, history.taa, history.detail
    if motion.dx or motion.dy:
        grid = pixel_grid(current, cache) - current.new_tensor(
            (2 * motion.dx / w, 2 * motion.dy / h)
        )
        warped = F.grid_sample(
            torch.cat((source, taa, detail), dim=1),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        source, taa, detail = warped.split(3, dim=1)
        valid = (
            (grid[..., 0].abs() <= 1 - 1 / w + 1e-6)
            & (grid[..., 1].abs() <= 1 - 1 / h + 1e-6)
        ).unsqueeze(1)
    else:
        valid = None
    # Moving objects and newly exposed pixels cannot borrow the camera's motion.
    error = (source - current).abs().amax(dim=1, keepdim=True)
    local_error = F.avg_pool2d(error, 3, stride=1, padding=1, count_include_pad=False)
    luma = rgb_luma(current)
    local_range = F.max_pool2d(luma, 3, 1, 1) + F.max_pool2d(-luma, 3, 1, 1)
    tolerance = (2.0 / 255.0 + 0.25 * local_range).clamp(max=sensitivity)
    confidence = (1 - torch.maximum(error, local_error) / tolerance).clamp(0, 1)
    if valid is not None:
        confidence = confidence * valid
    return AlignedHistory(taa, detail, confidence)


class ImageFeatures:
    """Luma, edges and local averages for one pass input."""

    def __init__(
        self,
        image: torch.Tensor,
        net: Optional[DLAANet],
        sobel_kernel: Optional[torch.Tensor] = None,
    ) -> None:
        self.image = image
        self.net = net
        self.sobel_kernel = sobel_kernel
        self._luma: Optional[torch.Tensor] = None
        self._luma_edge: Optional[tuple] = None
        self._blurs: dict[int, torch.Tensor] = {}

    def luma(self) -> torch.Tensor:
        if self._luma is None:
            self._luma = rgb_luma(self.image)
        return self._luma

    def luma_edge(self, return_grad: bool = False) -> tuple:
        if self._luma_edge is None:
            kernel = self.sobel_kernel
            if kernel is None:
                kernel = torch.cat((self.net.sobel_x, self.net.sobel_y), dim=0)
            gradients = F.conv2d(
                F.pad(self.luma(), (1, 1, 1, 1), mode="replicate"), kernel
            )
            sx, sy = gradients[:, :1], gradients[:, 1:2]
            edge = torch.sqrt(sx * sx + sy * sy + 1e-6)
            self._luma_edge = (self.luma(), edge, sx, sy)
        return self._luma_edge if return_grad else self._luma_edge[:2]

    def blur(self, kernel_size: int) -> torch.Tensor:
        if kernel_size not in self._blurs:
            pad = kernel_size // 2
            mode = "reflect" if min(self.image.shape[-2:]) > pad else "replicate"
            self._blurs[kernel_size] = F.avg_pool2d(
                F.pad(self.image, [pad] * 4, mode=mode), kernel_size, stride=1
            )
        return self._blurs[kernel_size]


def image_features(
    image: torch.Tensor, net: Optional[DLAANet], cached: Optional[ImageFeatures] = None
) -> ImageFeatures:
    if cached is not None and cached.image is image and cached.net is net:
        return cached
    kernel = cached.sobel_kernel if cached is not None and cached.net is net else None
    return ImageFeatures(image, net, kernel)


def is_oom(error: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", None)
    return (oom_type is not None and isinstance(error, oom_type)) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def clear_device_cache() -> None:
    if mm is not None:
        mm.soft_empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or overlap < 0 or overlap * 2 >= tile_size:
        raise ValueError(
            f"Invalid tiling settings: tile_size={tile_size}, overlap={overlap}"
        )
    last = max(0, length - tile_size)
    starts = list(range(0, last + 1, tile_size - overlap * 2))
    if starts[-1] != last:
        starts.append(last)
    return starts


class VideoTAADLAA:
    def __init__(self) -> None:
        self.net_cache: dict[str, ManagedModel] = {}
        self._net_lock = threading.Lock()

        self.texture_net_cache: dict[str, ManagedModel] = {}
        self._texture_lock = threading.Lock()

        self._seen_warnings = set()
        self._warning_lock = threading.Lock()

        self.settings: Mapping[str, float | int | bool] = SETTINGS

        self.preset_model_weight = PRESET_MODEL_WEIGHT
        self.texture_presets = TEXTURE_PRESETS

    def _managed_model(self, net: DLAANet, device: torch.device) -> ManagedModel:
        patcher = None
        if mm is not None and model_patcher is not None:
            # Ordinary torch layers require a fully loaded, non-dynamic patcher.
            patcher_class = getattr(
                model_patcher, "CoreModelPatcher", model_patcher.ModelPatcher
            )
            patcher = patcher_class(
                net, load_device=device, offload_device=torch.device("cpu")
            )
        return ManagedModel(net, patcher)

    def _activate_models(
        self,
        models: tuple[Optional[ManagedModel], ...],
        device: torch.device,
        height: int,
        width: int,
        tile_size: int,
    ) -> None:
        patchers = [
            entry.patcher
            for entry in models
            if entry is not None and entry.patcher is not None
        ]
        if patchers:
            # Conservative working-space estimate for the CNN and full-frame FX.
            tile_pixels = min(height, tile_size) * min(width, tile_size)
            memory_required = tile_pixels * 192 * 4 * 6 + height * width * 3 * 4 * 24
            mm.load_models_gpu(
                patchers, memory_required=memory_required, force_full_load=True
            )
        for entry in models:
            if entry is not None and entry.patcher is None:
                entry.net.to(device)

    def _release_unmanaged_models(
        self, models: tuple[Optional[ManagedModel], ...]
    ) -> None:
        # ComfyUI owns registered model residency/accounting. Standalone usage
        # has no manager, so return its cached models to CPU even on failure.
        for entry in models:
            if entry is not None and entry.patcher is None:
                entry.net.to("cpu")

    def _first_time(self, key: str) -> bool:
        with self._warning_lock:
            if key in self._seen_warnings:
                return False

            self._seen_warnings.add(key)
            return True

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "preset": (
                    [
                        "Auto",
                        "Performance",
                        "Balanced",
                        "High Detail",
                    ],
                ),
                "dlaa_intensity": (
                    "FLOAT",
                    {
                        "default": 1.00,
                        "min": 0.00,
                        "max": 2.00,
                        "step": 0.05,
                    },
                ),
                "texture_intensity": (
                    "FLOAT",
                    {
                        "default": 1.00,
                        "min": 0.00,
                        "max": 2.00,
                        "step": 0.05,
                    },
                ),
                "motion_stability": (
                    "FLOAT",
                    {
                        "default": 1.00,
                        "min": 0.50,
                        "max": 2.00,
                        "step": 0.05,
                    },
                ),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "CustomPostProcess"

    def _net(self, device: torch.device) -> ManagedModel:
        key = str(device)
        with self._net_lock:
            if key not in self.net_cache:
                self.net_cache[key] = self._managed_model(
                    load_checkpoint("DLAANet"), device
                )
            return self.net_cache[key]

    def _texture_net(self, device: torch.device) -> Optional[ManagedModel]:
        key = str(device)
        with self._texture_lock:
            if key not in self.texture_net_cache:
                try:
                    net = load_checkpoint("DLAATexture")
                except FileNotFoundError:
                    if self._first_time("texture_missing"):
                        log.info("[DLAA] No texture checkpoint; skipping texture.")
                    return None
                except (
                    ValueError,
                    RuntimeError,
                    SafetensorError,
                    OSError,
                    EOFError,
                    pickle.UnpicklingError,
                ):
                    if self._first_time("texture_load_error"):
                        log.exception(
                            "[DLAA] Cannot use the optional texture checkpoint"
                        )
                    return None
                self.texture_net_cache[key] = self._managed_model(net, device)
            return self.texture_net_cache[key]

    def _tile_weight_map(
        self, th: int, tw: int, overlap: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        w_y = torch.ones(th, device=device, dtype=dtype)
        w_x = torch.ones(tw, device=device, dtype=dtype)

        ramp = min(overlap, th // 4, tw // 4)

        if ramp > 0:
            values = torch.linspace(
                1.0 / (ramp + 1), ramp / (ramp + 1), ramp, device=device, dtype=dtype
            )

            w_y[:ramp] = values
            w_y[-ramp:] = torch.flip(values, dims=[0])

            w_x[:ramp] = values
            w_x[-ramp:] = torch.flip(values, dims=[0])

        return torch.minimum(w_y.view(1, 1, th, 1), w_x.view(1, 1, 1, tw))

    def _tiled_forward(
        self,
        net: Optional[DLAANet],
        image: torch.Tensor,
        tile_size: int = 512,
        overlap: int = 32,
    ) -> torch.Tensor:
        tile_size, overlap = int(tile_size), int(overlap)
        B, C, H, W = image.shape
        rows = tile_starts(H, tile_size, overlap)
        columns = tile_starts(W, tile_size, overlap)
        if len(rows) == len(columns) == 1:
            return net(image).clamp(0.0, 1.0)

        out = torch.zeros_like(image)
        weight = torch.zeros(B, 1, H, W, device=image.device, dtype=image.dtype)
        weight_cache = {}
        for y0 in rows:
            y1 = min(y0 + tile_size, H)
            for x0 in columns:
                x1 = min(x0 + tile_size, W)
                tile = image[:, :, y0:y1, x0:x1]
                tile_out = net(tile)
                tile_h, tile_w = tile.shape[-2:]
                key = (tile_h, tile_w)
                if key not in weight_cache:
                    weight_cache[key] = self._tile_weight_map(
                        tile_h, tile_w, overlap, image.device, image.dtype
                    )
                w_map = weight_cache[key]
                out[:, :, y0:y1, x0:x1] += tile_out * w_map
                weight[:, :, y0:y1, x0:x1] += w_map
        return (out / weight.clamp(min=1e-6)).clamp(0.0, 1.0)

    def _luma_edge(
        self,
        image: torch.Tensor,
        net: Optional[DLAANet],
        return_grad: bool = False,
        features: Optional[ImageFeatures] = None,
    ) -> tuple:
        return image_features(image, net, features).luma_edge(return_grad)

    def _edge_aa(
        self,
        image: torch.Tensor,
        thr: float,
        blur_radius: int,
        net: Optional[DLAANet],
        strength: float = 1.0,
        features: Optional[ImageFeatures] = None,
    ) -> torch.Tensor:
        strength = clamp01(strength)

        if blur_radius <= 0 or strength == 0.0:
            return image

        features = image_features(image, net, features)
        _, edge = features.luma_edge()

        mask = torch.sigmoid((edge - thr) * self.settings["edge_aa_slope"])
        mask = mask * strength

        blurred = features.blur(blur_radius * 2 + 1)

        return image * (1.0 - mask) + blurred * mask

    def _fine_line_aa(
        self,
        image: torch.Tensor,
        net: Optional[DLAANet],
        strength: float,
        features: Optional[ImageFeatures] = None,
    ) -> torch.Tensor:
        strength = clamp01(strength)

        if strength == 0.0:
            return image

        features = image_features(image, net, features)
        luma, edge = features.luma_edge()

        dark_mask = torch.sigmoid(
            (self.settings["detail_fine_line_dark_threshold"] - luma)
            * self.settings["fine_line_dark_slope"]
        )

        edge_mask = torch.sigmoid(
            (edge - self.settings["detail_fine_line_edge_threshold"])
            * self.settings["edge_aa_slope"]
        )

        local_avg = features.blur(3)
        fine_detail = rgb_luma((image - local_avg).abs())

        detail_mask = torch.sigmoid(
            (fine_detail - self.settings["detail_shimmer_threshold"])
            * self.settings["detail_shimmer_slope"]
        )

        line_mask = (dark_mask * edge_mask * detail_mask).clamp(0.0, 1.0)

        blur_strength = clamp01(self.settings["detail_fine_line_blur_strength"])

        aa_target = torch.lerp(
            image,
            local_avg,
            blur_strength,
        )

        blend = (line_mask * strength).clamp(0.0, 1.0)

        return torch.lerp(image, aa_target, blend).clamp(0.0, 1.0)

    def _specular_detail(
        self,
        image: torch.Tensor,
        net: Optional[DLAANet],
        highlight_mask: Optional[torch.Tensor],
        strength: float,
        features: Optional[ImageFeatures] = None,
    ) -> torch.Tensor:
        strength = clamp01(strength)

        if strength == 0.0:
            return image

        features = image_features(image, net, features)
        luma, edge = features.luma_edge()

        local_avg = features.blur(3)

        spec_residual = (image - local_avg).clamp(min=0.0)
        spec_residual = spec_residual.clamp(max=self.settings["detail_specular_limit"])

        bright_mask = torch.sigmoid(
            (luma - self.settings["detail_specular_threshold"])
            * self.settings["detail_specular_slope"]
        )

        clip_protect = 1.0 - torch.sigmoid(
            (luma - self.settings["specular_clip_luma"])
            * self.settings["specular_clip_slope"]
        )

        edge_mask = torch.sigmoid(
            (edge - self.settings["specular_edge_threshold"])
            * self.settings["specular_edge_slope"]
        )
        edge_mix = torch.lerp(
            torch.ones_like(edge_mask),
            edge_mask,
            self.settings["detail_specular_edge_boost"],
        )

        spec_mask = (bright_mask * clip_protect * edge_mix).clamp(0.0, 1.0)

        if highlight_mask is not None:
            spec_mask = spec_mask * (
                1.0 - highlight_mask * self.settings["specular_highlight_suppression"]
            )

        return (image + spec_residual * spec_mask * strength).clamp(0.0, 1.0)

    def _micro_contrast(
        self,
        image: torch.Tensor,
        highlight_mask: Optional[torch.Tensor],
        strength: float,
        net: Optional[DLAANet] = None,
        features: Optional[ImageFeatures] = None,
    ) -> torch.Tensor:
        strength = clamp01(strength)

        if strength == 0.0:
            return image

        radius = int(self.settings["detail_micro_contrast_radius"])
        radius = max(3, radius)

        if radius % 2 == 0:
            radius += 1

        features = image_features(image, net, features)
        local_avg = features.blur(radius)

        residual = image - local_avg
        residual = residual.clamp(
            -self.settings["detail_micro_contrast_limit"],
            self.settings["detail_micro_contrast_limit"],
        )

        detail_energy = rgb_luma(residual.abs())

        detail_mask = torch.sigmoid(
            (detail_energy - self.settings["micro_detail_threshold"])
            * self.settings["micro_detail_slope"]
        )

        if highlight_mask is not None:
            protect = (
                1.0
                - highlight_mask
                * self.settings["detail_micro_contrast_highlight_protect"]
            )
        else:
            luma = features.luma()
            protect = (
                1.0
                - torch.sigmoid(
                    (luma - self.settings["micro_highlight_luma"])
                    * self.settings["micro_highlight_slope"]
                )
                * self.settings["detail_micro_contrast_highlight_protect"]
            )

        out = image + residual * detail_mask * protect * strength

        return torch.clamp(out, 0.0, 1.0)

    def _edge_dehalo(
        self,
        image: torch.Tensor,
        net: Optional[DLAANet],
        strength: float,
        features: Optional[ImageFeatures] = None,
    ) -> torch.Tensor:
        strength = clamp01(strength)

        if strength == 0.0:
            return image

        features = image_features(image, net, features)
        luma, edge = features.luma_edge()

        edge_mask = torch.sigmoid(
            (edge - self.settings["detail_dehalo_threshold"])
            * self.settings["edge_sharp_slope"]
        )

        local_avg = features.blur(5)

        halo_residual = image - local_avg

        bright_halo = halo_residual.clamp(min=0.0)
        dark_halo = (-halo_residual).clamp(min=0.0)

        dark_protect = torch.sigmoid(
            (self.settings["dehalo_dark_protect_luma"] - luma)
            * self.settings["dehalo_protect_slope"]
        )
        light_protect = torch.sigmoid(
            (luma - self.settings["dehalo_light_protect_luma"])
            * self.settings["dehalo_protect_slope"]
        )

        bright_reduce = (
            bright_halo
            * edge_mask
            * strength
            * (1.0 - light_protect * self.settings["detail_dehalo_light_protect"])
        )

        dark_reduce = (
            dark_halo
            * edge_mask
            * strength
            * (1.0 - dark_protect * self.settings["detail_dehalo_dark_protect"])
        )

        out = image - bright_reduce + dark_reduce

        return torch.clamp(out, 0.0, 1.0)

    def _chroma_edge_cleanup(
        self,
        image: torch.Tensor,
        net: Optional[DLAANet],
        strength: float,
        features: Optional[ImageFeatures] = None,
    ) -> torch.Tensor:
        strength = clamp01(strength)

        if strength == 0.0:
            return image

        luma, edge = image_features(image, net, features).luma_edge()

        edge_mask = torch.sigmoid(
            (edge - self.settings["detail_chroma_edge_threshold"])
            * self.settings["edge_aa_slope"]
        )

        chroma = image - luma

        chroma_blur = ImageFeatures(chroma, net).blur(3)

        chroma_delta = chroma_blur - chroma
        chroma_residual = rgb_luma(chroma_delta.abs())
        chroma_amount = rgb_luma(chroma.abs())

        chroma_mask = torch.sigmoid(
            (chroma_amount - self.settings["detail_chroma_saturation_threshold"])
            * self.settings["chroma_saturation_slope"]
        )

        fringe_mask = torch.sigmoid(
            (
                chroma_residual
                - self.settings["detail_chroma_saturation_threshold"]
                * self.settings["chroma_fringe_threshold_scale"]
            )
            * self.settings["chroma_fringe_slope"]
        )

        dark_protect = torch.sigmoid(
            (luma - self.settings["detail_chroma_dark_protect"])
            * self.settings["chroma_dark_protect_slope"]
        )

        mask = (edge_mask * chroma_mask * fringe_mask * dark_protect).clamp(0.0, 1.0)

        chroma_delta = chroma_delta.clamp(
            -self.settings["detail_chroma_cleanup_limit"],
            self.settings["detail_chroma_cleanup_limit"],
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
        net: Optional[DLAANet],
        motion_gate: float,
        strength: float,
        features: Optional[ImageFeatures] = None,
        grid_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        strength = clamp01(strength)

        if strength == 0.0:
            return image

        sample_scale = float(self.settings["detail_subpixel_sample_scale"])
        if sample_scale == 0.0:
            return image

        _, edge, sx, sy = image_features(image, net, features).luma_edge(
            return_grad=True
        )

        edge_mask = torch.sigmoid(
            (edge - self.settings["detail_subpixel_edge_threshold"])
            * self.settings["detail_subpixel_edge_slope"]
        )

        B, C, H, W = image.shape

        nx = sx / edge.clamp(min=1e-6)
        ny = sy / edge.clamp(min=1e-6)

        offset_x = nx.squeeze(1) * sample_scale * (2.0 / W)
        offset_y = ny.squeeze(1) * sample_scale * (2.0 / H)

        base_grid = pixel_grid(image, grid_cache)

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

        reconstructed = (sample_pos + sample_neg) * 0.5

        delta = reconstructed - image
        delta = delta.clamp(
            -self.settings["detail_subpixel_delta_limit"],
            self.settings["detail_subpixel_delta_limit"],
        )

        stable_motion = 1.0 - min(
            max(
                float(motion_gate) * self.settings["detail_subpixel_motion_protect"],
                0.0,
            ),
            1.0,
        )

        blend = (edge_mask * strength * stable_motion).clamp(
            0.0,
            self.settings["detail_subpixel_blend_limit"],
        )

        out = image + delta * blend

        return torch.clamp(out, 0.0, 1.0)

    def _temporal_specular_stabilizer(
        self,
        image: torch.Tensor,
        previous: Optional[torch.Tensor],
        motion_gate: float,
        strength: float,
        net: Optional[DLAANet] = None,
        features: Optional[ImageFeatures] = None,
        previous_features: Optional[ImageFeatures] = None,
    ) -> torch.Tensor:
        strength = clamp01(strength)

        if strength == 0.0:
            return image

        if previous is None or previous.shape != image.shape:
            return image

        features = image_features(image, net, features)
        luma = features.luma()
        previous_features = image_features(previous, net, previous_features)
        prev_luma = previous_features.luma()

        local_avg = features.blur(3)

        prev_local_avg = previous_features.blur(3)

        current_spec = (image - local_avg).clamp(min=0.0)
        previous_spec = (previous - prev_local_avg).clamp(min=0.0)

        spec_energy = rgb_luma(current_spec)

        bright_mask = torch.sigmoid(
            (luma - self.settings["detail_specular_temporal_threshold"])
            * self.settings["detail_specular_temporal_slope"]
        )

        spec_mask = torch.sigmoid(
            (spec_energy - self.settings["detail_specular_temporal_detail_threshold"])
            * self.settings["temporal_spec_detail_slope"]
        )

        # avoid pulling moving highlights from history
        local_motion = torch.abs(luma - prev_luma)
        local_stable = 1.0 - torch.clamp(
            local_motion / self.settings["temporal_spec_motion_scale"],
            0.0,
            1.0,
        )

        global_stable = 1.0 - min(
            max(
                float(motion_gate)
                * self.settings["detail_specular_temporal_motion_protect"],
                0.0,
            ),
            1.0,
        )

        blend = (
            bright_mask * spec_mask * local_stable * global_stable * strength
        ).clamp(0.0, self.settings["detail_specular_temporal_blend_limit"])

        stabilized_spec = torch.lerp(
            current_spec,
            previous_spec,
            blend,
        )

        delta = stabilized_spec - current_spec
        delta = delta.clamp(
            -self.settings["detail_specular_temporal_delta_limit"],
            self.settings["detail_specular_temporal_delta_limit"],
        )

        out = image + delta

        return torch.clamp(out, 0.0, 1.0)

    def _local_tone_mapping(
        self,
        image: torch.Tensor,
        highlight_mask: torch.Tensor,
        motion_gate: float,
        strength: float,
        net: Optional[DLAANet] = None,
        features: Optional[ImageFeatures] = None,
    ) -> torch.Tensor:
        strength = clamp01(strength)

        if strength == 0.0:
            return image

        radius = int(self.settings["detail_local_tonemap_radius"])
        radius = max(3, radius)

        if radius % 2 == 0:
            radius += 1

        luma = image_features(image, net, features).luma()
        local_avg = ImageFeatures(luma, net).blur(radius)

        local_contrast = luma - local_avg
        local_contrast = local_contrast.clamp(
            -self.settings["detail_local_tonemap_limit"],
            self.settings["detail_local_tonemap_limit"],
        )

        if highlight_mask is not None:
            highlight_protect = (
                1.0
                - highlight_mask
                * self.settings["detail_local_tonemap_highlight_protect"]
            )
        else:
            highlight_protect = (
                1.0
                - torch.sigmoid(
                    (luma - self.settings["local_tonemap_highlight_luma"])
                    * self.settings["local_tonemap_highlight_slope"]
                )
                * self.settings["detail_local_tonemap_highlight_protect"]
            )

        shadow_mask = torch.sigmoid(
            (self.settings["detail_local_tonemap_shadow_threshold"] - luma)
            * self.settings["local_tonemap_shadow_slope"]
        )

        shadow_lift = (
            shadow_mask
            * self.settings["detail_local_tonemap_shadow_lift"]
            * highlight_protect
        )

        motion_stable = 1.0 - min(
            max(
                float(motion_gate)
                * self.settings["detail_local_tonemap_motion_protect"],
                0.0,
            ),
            1.0,
        )

        luma_delta = (
            (local_contrast * highlight_protect + shadow_lift)
            * strength
            * motion_stable
        )

        luma_delta = luma_delta.clamp(
            -self.settings["detail_local_tonemap_limit"],
            self.settings["detail_local_tonemap_limit"],
        )

        # preserve RGB balance with a luma ratio
        target_luma = (luma + luma_delta).clamp(0.0, 1.0)

        ratio = target_luma / luma.clamp(min=1e-6)
        ratio = ratio.clamp(
            self.settings["local_tonemap_ratio_min"],
            self.settings["local_tonemap_ratio_max"],
        )
        out = image * ratio

        return torch.clamp(out, 0.0, 1.0)

    def _fur_hair_stabilizer(
        self,
        image: torch.Tensor,
        previous: Optional[torch.Tensor],
        net: Optional[DLAANet],
        motion_gate: float,
        strength: float,
        features: Optional[ImageFeatures] = None,
        previous_features: Optional[ImageFeatures] = None,
    ) -> torch.Tensor:
        strength = clamp01(strength)

        if strength == 0.0:
            return image

        if previous is None or previous.shape != image.shape:
            return image

        features = image_features(image, net, features)
        _, edge = features.luma_edge()

        local_avg = features.blur(3)
        prev_local_avg = image_features(previous, net, previous_features).blur(3)

        fine_detail = rgb_luma((image - local_avg).abs())
        prev_detail = previous - prev_local_avg
        current_detail = image - local_avg

        edge_mask = torch.sigmoid(
            (edge - self.settings["detail_fur_edge_threshold"])
            * self.settings["edge_aa_slope"]
        )

        detail_mask = torch.sigmoid(
            (fine_detail - self.settings["detail_fur_detail_threshold"])
            * self.settings["fur_detail_slope"]
        )

        stable_motion = 1.0 - min(
            max(float(motion_gate) * self.settings["detail_fur_motion_protect"], 0.0),
            1.0,
        )

        fur_mask = (edge_mask * detail_mask * stable_motion).clamp(0.0, 1.0)

        blend = (fur_mask * strength).clamp(
            0.0, self.settings["detail_fur_blend_limit"]
        )

        stabilized_detail = torch.lerp(
            current_detail,
            prev_detail,
            blend,
        )

        base = local_avg
        out = base + stabilized_detail

        return torch.clamp(out, 0.0, 1.0)

    def _temporal_refine(
        self,
        current: torch.Tensor,
        previous: Optional[torch.Tensor],
        strength: float = 0.35,
        motion_threshold: float = 0.08,
        net: Optional[DLAANet] = None,
        features: Optional[ImageFeatures] = None,
        previous_features: Optional[ImageFeatures] = None,
    ) -> torch.Tensor:
        if strength == 0.0:
            return current
        if previous is None:
            return current

        if previous.shape != current.shape:
            return current

        curr_luma = image_features(current, net, features).luma()
        prev_luma = image_features(previous, net, previous_features).luma()

        motion = torch.abs(curr_luma - prev_luma)
        motion = torch.clamp(motion / motion_threshold, 0.0, 1.0)

        blend_mask = (1.0 - motion) * strength

        refined = current * (1.0 - blend_mask) + previous * blend_mask
        return refined.clamp(0.0, 1.0)

    def _stabilize_fine_detail(
        self,
        current: torch.Tensor,
        previous: Optional[torch.Tensor],
        strength: float,
        motion_threshold: float,
        net: Optional[DLAANet] = None,
        features: Optional[ImageFeatures] = None,
        previous_features: Optional[ImageFeatures] = None,
    ) -> torch.Tensor:
        if previous is None:
            return current

        if previous.shape != current.shape:
            return current

        strength = clamp01(strength)

        if strength == 0.0:
            return current

        current_base = image_features(current, net, features).blur(3)
        previous_base = image_features(previous, net, previous_features).blur(3)

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
            (detail_energy - self.settings["detail_shimmer_threshold"])
            * self.settings["detail_shimmer_slope"]
        )

        blend_mask = (line_mask * stable_area * strength).clamp(
            0.0,
            self.settings["detail_shimmer_max_blend"],
        )

        stabilized_detail = torch.lerp(
            current_detail,
            previous_detail,
            blend_mask,
        )

        return torch.clamp(current_base + stabilized_detail, 0.0, 1.0)

    def _resolve_frame_config(
        self, preset: str, still: bool, motion: Optional[float]
    ) -> FrameParams:
        if preset == "Auto" and not still:
            amount = (
                self.settings["auto_default_scene_motion"] if motion is None else motion
            )
            if amount < self.settings["auto_static_motion_threshold"]:
                preset, config = "Detail", AUTO_STATIC
            elif amount < self.settings["auto_balanced_motion_threshold"]:
                preset, config = "Balanced", AUTO_BALANCED
            else:
                preset, config = "Performance", AUTO_MOTION
        else:
            if preset == "Auto":
                preset = "Photo"
            config = PRESETS[preset]
        values = dict(config)
        # Presets from earlier releases still carry the render-time jitter setting.
        values.pop("jitter_scale", None)
        if still:
            values["taa_strength"] = 0.0
        return FrameParams(preset=preset, **values)

    def _apply_texture_pass(
        self,
        texture_net: Optional[DLAANet],
        dlaa_out: torch.Tensor,
        tile_size: int,
        motion_gate: float,
        dark_mask: torch.Tensor,
        highlight_mask: torch.Tensor,
        preset: str,
        texture_intensity: float = 1.00,
        motion_stability: float = 1.00,
        frame_index: Optional[int] = None,
    ) -> torch.Tensor:
        if texture_net is None:
            return dlaa_out

        if texture_intensity <= 0.0:
            return dlaa_out

        texture_cfg = self.texture_presets.get(preset, self.texture_presets["Balanced"])
        if not texture_cfg.get("enabled", True):
            return dlaa_out

        try:
            gen_out = self._tiled_forward(
                texture_net,
                dlaa_out,
                tile_size=tile_size,
                overlap=self.settings["texture_tile_overlap"],
            )

        except RuntimeError as error:
            if not is_oom(error):
                raise
            gen_out = None

        if gen_out is None:
            clear_device_cache()
            if self._first_time("texture_oom"):
                log.warning(
                    "[DLAA] Texture pass ran out of memory, skipping texture pass."
                )
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
        thin_edge_mask = F.avg_pool2d(
            thin_edge_mask, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )

        blur_kernel = int(texture_cfg.get("blur_kernel", 3))
        blur_kernel = max(3, blur_kernel)

        if blur_kernel % 2 == 0:
            blur_kernel += 1

        gen_blur = F.avg_pool2d(
            gen_out,
            kernel_size=blur_kernel,
            stride=1,
            padding=blur_kernel // 2,
            count_include_pad=False,
        )
        texture_delta = gen_out - gen_blur

        interval = self.settings["texture_log_interval"]
        log_frame = (
            log.isEnabledFor(logging.DEBUG)
            and frame_index is not None
            and interval > 0
            and frame_index % interval == 0
        )
        raw_gen_delta = 0.0
        if log_frame:
            raw_gen_delta = texture_delta.abs().mean().item()

        dark_base = texture_cfg["dark_base"]
        texture_strength = texture_cfg["strength"] * texture_intensity
        texture_limit = texture_cfg["limit"] * texture_intensity
        motion_suppression = min(
            texture_cfg["motion_suppression"] * motion_stability, 0.98
        )
        line_suppression = min(texture_cfg["line_suppression"] * motion_stability, 0.95)

        texture_mask = dark_base + (1.0 - dark_base) * dark_mask
        texture_mask = texture_mask * (
            1.0 - highlight_mask * texture_cfg["highlight_suppression"]
        )
        texture_mask = texture_mask * (1.0 - motion_gate * motion_suppression)
        texture_mask = texture_mask * (1.0 - thin_edge_mask * line_suppression)

        texture_delta = texture_delta * texture_mask
        texture_delta = texture_delta.clamp(-texture_limit, texture_limit)

        out = dlaa_out + texture_delta * texture_strength

        if log_frame:
            final_texture_delta = (out - dlaa_out).abs().mean().item()
            log.debug(
                "[DLAA] texture frame=%d raw_delta=%.6f final_delta=%.6f",
                frame_index,
                raw_gen_delta,
                final_texture_delta,
            )

        return torch.clamp(out, 0.0, 1.0)

    def _normalize_run_inputs(
        self,
        preset: str,
        dlaa_intensity: float,
        texture_intensity: float,
        motion_stability: float,
    ) -> tuple[str, float, float, float]:
        preset_aliases = {
            "High Detail": "Detail",
            "Sharp": "Detail",
            "Cinematic": "Smooth",
        }

        preset = preset_aliases.get(preset, preset)
        if preset != "Auto" and preset not in PRESETS:
            preset = "Balanced"

        dlaa_intensity = max(0.0, min(float(dlaa_intensity), 2.0))
        texture_intensity = max(0.0, min(float(texture_intensity), 2.0))
        motion_stability = max(0.5, min(float(motion_stability), 2.0))

        return preset, dlaa_intensity, texture_intensity, motion_stability

    def _get_device(self) -> torch.device:
        if mm is not None:
            return mm.get_torch_device()

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _frame_blur_radius(self, preset: str, is_single_image: bool) -> int:
        if is_single_image:
            if preset == "Photo":
                return 1
            return 0

        return 1

    def _frame_edge_aa_strength(self, preset: str, is_single_image: bool) -> float:
        if is_single_image:
            if preset == "Photo":
                return self.settings["photo_edge_aa_strength"]
            return 0.0

        if preset == "Detail":
            return self.settings["detail_edge_aa_strength"]

        if preset == "Photo":
            return self.settings["photo_edge_aa_strength"]

        return 1.0

    def _vram_mb(self, device: torch.device) -> int:
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

            return torch.cuda.get_device_properties(device_index).total_memory // (
                1024 * 1024
            )
        except Exception:
            return 0

    def _tile_size_for_vram(self, vram_mb: int) -> int:
        if vram_mb <= 0:
            return 512

        if vram_mb <= 8192:
            return 512

        if vram_mb <= 16384:
            return 1024

        return 1536

    def _log_tiling(
        self, height: int, width: int, tile_size: int, overlap: int = 32
    ) -> None:
        rows = tile_starts(height, tile_size, overlap)
        columns = tile_starts(width, tile_size, overlap)
        if len(rows) * len(columns) > 1:
            log.debug(
                "[DLAA] Tiled inference: %d tiles (size=%d, overlap=%d)",
                len(rows) * len(columns),
                tile_size,
                overlap,
            )

    def _load_frame(
        self, images: torch.Tensor, frame_index: int, device: torch.device
    ) -> torch.Tensor:
        img = (
            images[frame_index : frame_index + 1, :, :, :3]
            .to(device=device, dtype=torch.float32)
            .permute(0, 3, 1, 2)
        )
        if img.shape[1] == 1:
            img = img.expand(-1, 3, -1, -1)
        return img.contiguous()

    def _texture_net_for_run(
        self, device: torch.device, preset: str, texture_intensity: float
    ) -> Optional[ManagedModel]:
        texture_cfg = self.texture_presets.get(preset, self.texture_presets["Balanced"])

        texture_enabled = self.settings["texture_pass_enabled"]

        if not texture_enabled:
            return None

        if not texture_cfg.get("enabled", True):
            return None

        if texture_intensity <= 0.0:
            return None

        return self._texture_net(device)

    def _apply_taa(
        self,
        source: torch.Tensor,
        taa: TAAState,
        aligned: Optional[AlignedHistory],
        params: FrameParams,
        net: DLAANet,
        still: bool,
        run: RunContext,
    ) -> torch.Tensor:
        spatial = self._edge_aa(
            source,
            self.settings["edge_threshold"],
            self._frame_blur_radius(params.preset, still),
            net,
            self._frame_edge_aa_strength(params.preset, still),
            features=ImageFeatures(source, net, run.sobel_kernel),
        )
        if aligned is None or params.taa_strength == 0:
            taa.history = spatial.detach()
            return spatial
        taa.history = aligned.taa
        accumulated = taa.update(
            spatial, self.settings["taa_alpha"], params.motion_sensitivity
        )
        accepted = torch.lerp(spatial, accumulated, aligned.confidence)
        taa.history = accepted.detach()
        return torch.lerp(spatial, accepted, params.taa_strength)

    def _run_dlaa_with_retry(
        self,
        net: Optional[DLAANet],
        rgb: torch.Tensor,
        tile_size: int,
        debug_stats: bool,
        frame_index: Optional[int],
        overlap: int = 32,
    ) -> tuple[torch.Tensor, int]:
        tile = int(tile_size)
        min_tile_size = max(128, overlap * 2 + 1)
        last_error = None
        while tile >= min_tile_size:
            try:
                out = self._tiled_forward(net, rgb, tile_size=tile, overlap=overlap)
                if debug_stats and frame_index == 0:
                    log.debug(
                        "[DLAA] raw_model_delta=%.6f", (out - rgb).abs().mean().item()
                    )
                return out, tile
            except RuntimeError as error:
                if not is_oom(error):
                    raise
                # Keep the original exception type, without retaining failed tiles.
                last_error = error.with_traceback(None)
                last_error.__context__ = None
                last_error.__cause__ = None
            clear_device_cache()
            if tile == min_tile_size:
                log.error("[DLAA] Inference failed at minimum tile size %d.", tile)
                raise last_error
            next_tile = max(min_tile_size, tile // 2)
            log.warning(
                "[DLAA] Out of memory at tile size %d, retrying with %d.",
                tile,
                next_tile,
            )
            tile = next_tile
        raise ValueError(f"tile_size must be at least {min_tile_size}, got {tile_size}")

    def _apply_model_residual(
        self,
        rgb: torch.Tensor,
        dlaa_out: torch.Tensor,
        preset: str,
        debug_stats: bool,
        frame_index: Optional[int],
    ) -> tuple[torch.Tensor, Optional[float]]:
        dlaa_mean = dlaa_out.mean(dim=(1, 2, 3), keepdim=True)
        rgb_mean = rgb.mean(dim=(1, 2, 3), keepdim=True)
        dlaa_out = dlaa_out - dlaa_mean + rgb_mean

        preset_model_weight = self.preset_model_weight.get(preset, 1.00)

        model_delta = dlaa_out - rgb

        model_delta_value = None
        if debug_stats:
            model_delta_value = model_delta.abs().mean().item()

        dlaa_out = torch.clamp(
            rgb + model_delta * self.settings["model_weight"] * preset_model_weight,
            0.0,
            1.0,
        )

        if debug_stats and frame_index == 0:
            log.debug("[DLAA] model_delta_first=%.6f", model_delta_value)

        return dlaa_out, model_delta_value

    def _apply_highlight_preblend(
        self, dlaa_out: torch.Tensor, rgb: torch.Tensor, preset: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        luma = rgb_luma(dlaa_out)
        highlight_mask = torch.sigmoid(
            (luma - self.settings["highlight_threshold"])
            * self.settings["highlight_slope"]
        )

        highlight_pre_blend = self.settings["highlight_pre_blend"] * (
            self.settings["detail_highlight_pre_scale"] if preset == "Detail" else 1.0
        )
        dlaa_out = torch.lerp(dlaa_out, rgb, highlight_mask * highlight_pre_blend)
        dlaa_out = torch.clamp(dlaa_out, 0.0, 1.0)

        return dlaa_out, highlight_mask

    def _apply_motion_suppression(
        self,
        preset: str,
        detail_boost: float,
        edge_boost: float,
        micro_limit: float,
        motion_gate: float,
        motion_stability: float,
    ) -> tuple[float, float, float]:
        motion_cfg = (
            MOTION_SUPPRESSION["Detail"]
            if preset == "Detail"
            else MOTION_SUPPRESSION["Default"]
        )

        detail_boost *= 1.0 - motion_gate * motion_cfg["detail"] * motion_stability
        edge_boost *= 1.0 - motion_gate * motion_cfg["edge"] * motion_stability
        micro_limit *= 1.0 - motion_gate * motion_cfg["micro"] * motion_stability

        return detail_boost, edge_boost, micro_limit

    def _apply_detail_pass(
        self,
        dlaa_out: torch.Tensor,
        net: Optional[DLAANet],
        preset: str,
        detail_boost: float,
        edge_boost: float,
        micro_limit: float,
        edge_sharp_strength: float,
        highlight_mask: torch.Tensor,
        features: Optional[ImageFeatures] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        features = image_features(dlaa_out, net, features)
        local_avg_rgb = features.blur(3)
        fine_detail_rgb = dlaa_out - local_avg_rgb

        luma, edge = features.luma_edge()
        fine_detail = rgb_luma(fine_detail_rgb)

        texture_dark_mask = torch.clamp(
            (luma - self.settings["detail_dark_luma_start"])
            / self.settings["detail_dark_luma_range"],
            0.0,
            1.0,
        )

        if preset == "Detail":
            dark_mask = texture_dark_mask
            micro_limit = micro_limit * (
                self.settings["detail_dark_mix_base"]
                + self.settings["detail_dark_mix_scale"] * dark_mask
            )
        else:
            dark_mask = 1.0

        detail_strength = fine_detail.abs().mean(dim=(1, 2, 3), keepdim=True)
        detail_scale = (
            self.settings["detail_base_scale"]
            + (self.settings["detail_ref_scale"] / (detail_strength + 1e-6))
        ).clamp(self.settings["detail_min_scale"], self.settings["detail_max_scale"])

        fine_detail = fine_detail * torch.sigmoid(fine_detail * detail_scale)
        fine_detail = fine_detail.clamp(
            -self.settings["fine_detail_limit"], self.settings["fine_detail_limit"]
        )

        local_detail = F.avg_pool2d(
            fine_detail.abs(), 7, stride=1, padding=3, count_include_pad=False
        )
        global_detail = fine_detail.abs().mean(dim=(1, 2, 3), keepdim=True)

        edge_detail_weight = torch.sigmoid(
            (edge - self.settings["edge_sharp_threshold"])
            * self.settings["edge_sharp_slope"]
        )

        detail_gain = (global_detail / (local_detail + 1e-6)).clamp(
            self.settings["detail_min_gain"],
            self.settings["detail_max_gain"],
        )
        detail_gain = detail_gain * (1.0 - local_detail.clamp(0.0, 0.5))
        detail_gain = detail_gain * (
            1.0 + edge_detail_weight * self.settings["detail_edge_boost"]
        )
        detail_gain = detail_gain * (
            1.0 - highlight_mask * self.settings["detail_highlight_suppression"]
        )

        micro_detail = fine_detail_rgb * detail_gain * detail_boost * dark_mask
        micro_detail = micro_detail.clamp(-micro_limit, micro_limit)
        dlaa_out = dlaa_out + micro_detail

        edge_mask = edge_detail_weight
        edge_detail = fine_detail_rgb * edge_mask * (1.0 - highlight_mask)

        edge_boosted = edge_detail * edge_sharp_strength * edge_boost * dark_mask
        edge_boosted = edge_boosted.clamp(
            -micro_limit * self.settings["edge_detail_limit_scale"],
            micro_limit * self.settings["edge_detail_limit_scale"],
        )
        dlaa_out = dlaa_out + edge_boosted

        return dlaa_out, texture_dark_mask

    def _apply_tone_and_color_pass(
        self,
        dlaa_out: torch.Tensor,
        rgb: torch.Tensor,
        preset: str,
        highlight_mask: torch.Tensor,
        tone_strength: float,
        luma_boost_mult: float,
        saturation_boost_mult: float,
    ) -> torch.Tensor:
        tone_mapped = dlaa_out / (dlaa_out + self.settings["tone_curve_bias"])
        dlaa_out = torch.lerp(dlaa_out, tone_mapped, highlight_mask * tone_strength)

        luma_boost = (
            self.settings["luma_boost_base"]
            * luma_boost_mult
            * (1.0 - highlight_mask * self.settings["luma_highlight_protect"])
        )
        dlaa_out = dlaa_out * (1.0 + luma_boost)

        mean_rgb = dlaa_out.mean(dim=1, keepdim=True)
        saturation_boost = (
            self.settings["saturation_boost_base"]
            * saturation_boost_mult
            * (1.0 - highlight_mask * self.settings["saturation_highlight_protect"])
        )
        dlaa_out = mean_rgb + (dlaa_out - mean_rgb) * (1.0 + saturation_boost)

        highlight_post_blend = self.settings["highlight_post_blend"] * (
            self.settings["detail_highlight_post_scale"] if preset == "Detail" else 1.0
        )
        dlaa_out = torch.lerp(dlaa_out, rgb, highlight_mask * highlight_post_blend)

        return torch.clamp(dlaa_out, 0.0, 1.0)

    def _temporal_detail(
        self,
        image: torch.Tensor,
        aligned: Optional[AlignedHistory],
        params: FrameParams,
        motion_gate: float,
        net: DLAANet,
        run: RunContext,
    ) -> torch.Tensor:
        if aligned is None:
            return image
        previous = aligned.detail
        previous_features = ImageFeatures(previous, net, run.sobel_kernel)
        filtered = image
        if params.preset == "Detail":
            filtered = self._temporal_specular_stabilizer(
                filtered,
                previous,
                motion_gate,
                self.settings["detail_specular_temporal_strength"],
                net=net,
                features=ImageFeatures(filtered, net, run.sobel_kernel),
                previous_features=previous_features,
            )
            filtered = self._fur_hair_stabilizer(
                filtered,
                previous,
                net,
                motion_gate,
                self.settings["detail_fur_stabilizer_strength"],
                features=ImageFeatures(filtered, net, run.sobel_kernel),
                previous_features=previous_features,
            )
            filtered = self._stabilize_fine_detail(
                filtered,
                previous,
                self.settings["detail_shimmer_strength"],
                params.motion_threshold,
                net=net,
                previous_features=previous_features,
            )
        filtered = self._temporal_refine(
            filtered,
            previous,
            params.temporal_strength,
            params.motion_threshold,
            net=net,
            previous_features=previous_features,
        )
        # Apply rejection after the neighborhood filters as well, including the fur pass.
        return torch.lerp(image, filtered, aligned.confidence)

    def _apply_dlaa_pipeline(
        self,
        rgb: torch.Tensor,
        net: DLAANet,
        texture_net: Optional[DLAANet],
        params: FrameParams,
        run: RunContext,
        motion_gate: float,
        aligned: Optional[AlignedHistory],
        frame_index: Optional[int],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[float]]:
        if params.dlaa_strength <= 0.0:
            return rgb, rgb.detach(), None

        preset = params.preset
        is_detail = preset == "Detail"

        dlaa_out, tile = self._run_dlaa_with_retry(
            net, rgb, run.tile_size, run.debug_stats, frame_index, run.tile_overlap
        )
        # Keep a successful fallback for all remaining frames in this run.
        run.tile_size = tile
        dlaa_out, model_delta_value = self._apply_model_residual(
            rgb, dlaa_out, preset, run.debug_stats, frame_index
        )
        dlaa_out, highlight_mask = self._apply_highlight_preblend(dlaa_out, rgb, preset)
        features = ImageFeatures(dlaa_out, net, run.sobel_kernel)
        texture_dark_mask = None
        if preset == "Photo":
            if texture_net is not None:
                texture_dark_mask = (
                    (features.luma() - self.settings["detail_dark_luma_start"])
                    / self.settings["detail_dark_luma_range"]
                ).clamp(0.0, 1.0)
        elif preset != "Performance":
            detail_boost, edge_boost, micro_limit = self._apply_motion_suppression(
                preset,
                params.detail_boost,
                params.edge_boost,
                params.micro_limit,
                motion_gate,
                run.motion_stability,
            )
            dlaa_out, texture_dark_mask = self._apply_detail_pass(
                dlaa_out,
                net,
                preset,
                detail_boost,
                edge_boost,
                micro_limit,
                params.edge_sharp_strength,
                highlight_mask,
                features=features,
            )
            features = image_features(dlaa_out, net, features)

        if is_detail:
            dlaa_out = self._fine_line_aa(
                dlaa_out,
                net,
                self.settings["detail_fine_line_aa_strength"],
                features=features,
            )
            features = image_features(dlaa_out, net, features)

        if texture_net is not None and texture_dark_mask is not None:
            dlaa_out = self._apply_texture_pass(
                texture_net=texture_net,
                dlaa_out=dlaa_out,
                tile_size=tile,
                motion_gate=motion_gate,
                dark_mask=texture_dark_mask,
                highlight_mask=highlight_mask,
                preset=preset,
                texture_intensity=run.texture_intensity,
                motion_stability=run.motion_stability,
                frame_index=frame_index,
            )
            features = image_features(dlaa_out, net, features)

        if is_detail:
            dlaa_out = self._specular_detail(
                dlaa_out,
                net,
                highlight_mask,
                self.settings["detail_specular_strength"],
                features=features,
            )
            features = image_features(dlaa_out, net, features)
            dlaa_out = self._micro_contrast(
                dlaa_out,
                highlight_mask,
                self.settings["detail_micro_contrast_strength"],
                net=net,
                features=features,
            )
            features = image_features(dlaa_out, net, features)
            dlaa_out = self._edge_dehalo(
                dlaa_out,
                net,
                self.settings["detail_dehalo_strength"],
                features=features,
            )
            features = image_features(dlaa_out, net, features)
            dlaa_out = self._chroma_edge_cleanup(
                dlaa_out,
                net,
                self.settings["detail_chroma_cleanup_strength"],
                features=features,
            )
            features = image_features(dlaa_out, net, features)
            dlaa_out = self._subpixel_edge_reconstruction(
                dlaa_out,
                net,
                motion_gate,
                self.settings["detail_subpixel_edge_strength"],
                features=features,
                grid_cache=run.grid_cache,
            )
            features = image_features(dlaa_out, net, features)
            dlaa_out = self._local_tone_mapping(
                dlaa_out,
                highlight_mask,
                motion_gate,
                self.settings["detail_local_tonemap_strength"],
                net=net,
                features=features,
            )

        dlaa_out = self._apply_tone_and_color_pass(
            dlaa_out,
            rgb,
            preset,
            highlight_mask,
            params.tone_strength,
            params.luma_boost_mult,
            params.saturation_boost_mult,
        )
        filtered = self._temporal_detail(
            dlaa_out, aligned, params, motion_gate, net, run
        )
        blend = params.dlaa_strength * self.settings["dlaa_blend_scale"]
        if is_detail:
            blend = min(blend * self.settings["detail_blend_boost"], 1.0)
        return (
            torch.lerp(rgb, filtered.clamp(0.0, 1.0), blend),
            filtered.detach(),
            model_delta_value,
        )

    def execute(
        self,
        images: torch.Tensor,
        preset: str,
        dlaa_intensity: float = 1.0,
        texture_intensity: float = 1.0,
        motion_stability: float = 1.0,
        detail_intensity: Optional[float] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor]:
        if detail_intensity is not None:
            dlaa_intensity = detail_intensity
        output, _ = self._process(
            images,
            preset,
            dlaa_intensity,
            texture_intensity,
            motion_stability,
            video_chunks=False,
            state=None,
        )
        return (output,)

    def _process(
        self,
        images: torch.Tensor,
        preset: str,
        dlaa_intensity: float,
        texture_intensity: float,
        motion_stability: float,
        *,
        video_chunks: bool,
        state: Optional[ClipHistory],
    ) -> tuple[torch.Tensor, Optional[ClipHistory]]:
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError(
                "[DLAA] Expected a [batch, height, width, channels] image tensor."
            )
        B, H, W, C = images.shape
        if H < 1 or W < 1 or (C != 1 and C < 3) or not images.is_floating_point():
            raise ValueError(
                "[DLAA] Expected floating-point images with positive dimensions and one or at least three channels."
            )
        preset, intensity, texture_intensity, motion_stability = (
            self._normalize_run_inputs(
                preset, dlaa_intensity, texture_intensity, motion_stability
            )
        )
        signature = (preset, intensity, texture_intensity, motion_stability)
        if intensity == 0.0:
            # Preserve every input value and channel, including a one-channel IMAGE.
            return images.detach().cpu(), None
        if state is not None and (
            state.source.shape != (1, 3, H, W) or state.settings != signature
        ):
            raise ValueError(
                "[DLAA] Chunk dimensions or settings changed. Disconnect previous_state to start a new clip."
            )
        out = torch.empty((B, H, W, max(C, 3)), dtype=images.dtype, device="cpu")
        if B == 0:
            return out, state

        still = B == 1 and not video_chunks
        device = self._get_device()
        main = self._net(device)
        load_preset = "Photo" if preset == "Auto" and still else preset
        texture = self._texture_net_for_run(device, load_preset, texture_intensity)
        models = (main, texture)
        tile_size = self._tile_size_for_vram(self._vram_mb(device))
        if state is not None:
            tile_size = min(tile_size, state.tile_size)
        progress = ProgressBar(B) if ProgressBar is not None else None
        debug_stats = log.isEnabledFor(logging.DEBUG)
        delta_sum, delta_count = 0.0, 0
        taa = TAAState()
        run = None

        try:
            self._activate_models(models, device, H, W, tile_size)
            with torch.inference_mode():
                run = RunContext(
                    tile_size=tile_size,
                    tile_overlap=int(self.settings["tile_overlap"]),
                    motion_stability=motion_stability,
                    texture_intensity=texture_intensity,
                    debug_stats=debug_stats,
                    sobel_kernel=torch.cat((main.net.sobel_x, main.net.sobel_y)),
                )
                self._log_tiling(H, W, run.tile_size, run.tile_overlap)
                history = state.on(device) if state is not None else None
                start = state.frames if state is not None else 0
                for i in range(B):
                    if mm is not None:
                        mm.throw_exception_if_processing_interrupted()
                    source = self._load_frame(images, i, device)
                    motion = (
                        measure_camera_motion(source, history.source, run.grid_cache)
                        if history is not None
                        else None
                    )
                    if (
                        motion
                        and not motion.matched
                        and motion.amount >= self.settings["scene_cut_min_change"]
                    ):
                        motion = None
                    params = self._resolve_frame_config(
                        preset, still, motion.amount if motion else None
                    )
                    aligned = (
                        align_history(
                            source,
                            history,
                            motion,
                            params.motion_sensitivity,
                            run.grid_cache,
                        )
                        if motion and history is not None
                        else None
                    )
                    motion_gate = (
                        min(1.0, motion.amount * self.settings["motion_gate_scale"])
                        if motion
                        else 0.0
                    )
                    rgb = self._apply_taa(
                        source, taa, aligned, params, main.net, still, run
                    )
                    result, detail, delta = self._apply_dlaa_pipeline(
                        rgb,
                        main.net,
                        texture.net if texture is not None else None,
                        params,
                        run,
                        motion_gate,
                        aligned,
                        start + i,
                    )
                    if delta is not None:
                        delta_sum += delta
                        delta_count += 1
                    # Scale the combined TAA, DLAA and texture residual exactly once.
                    if intensity != 1.0:
                        result = torch.lerp(source, result.clamp(0.0, 1.0), intensity)
                    out[i : i + 1, :, :, :3].copy_(
                        result.clamp(0.0, 1.0).permute(0, 2, 3, 1)
                    )
                    if C > 3:
                        out[i : i + 1, :, :, 3:].copy_(images[i : i + 1, :, :, 3:])
                    history = ClipHistory(
                        source,
                        taa.history,
                        detail,
                        start + i + 1,
                        run.tile_size,
                        signature,
                    )
                    if progress is not None:
                        progress.update(1)
                # Copy the last frame even on CPU; an IMAGE view can own the entire input batch.
                next_state = history.on("cpu", copy=True) if video_chunks else None
        finally:
            if run is not None:
                run.grid_cache.clear()
            taa.reset()
            self._release_unmanaged_models(models)

        if debug_stats and delta_count:
            log.debug("[DLAA] model_delta_avg=%.6f", delta_sum / delta_count)
        return out, next_state


class VideoTAADLAAChunks(VideoTAADLAA):
    DESCRIPTION = (
        "Process consecutive chunks of one video. Leave previous_state disconnected for the first "
        "chunk or a new clip. Connect state to the next chunk's previous_state and keep the same "
        "settings and resolution. A one-frame chunk uses video processing."
    )
    RETURN_TYPES = ("IMAGE", "SOLRICKS_DLAA_STATE")
    RETURN_NAMES = ("images", "state")
    FUNCTION = "process_chunk"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        inputs = super().INPUT_TYPES()
        inputs["optional"]["previous_state"] = ("SOLRICKS_DLAA_STATE",)
        return inputs

    def process_chunk(
        self,
        images: torch.Tensor,
        preset: str,
        dlaa_intensity: float = 1.0,
        texture_intensity: float = 1.0,
        motion_stability: float = 1.0,
        previous_state: Optional[ClipHistory] = None,
    ) -> tuple[torch.Tensor, Optional[ClipHistory]]:
        return self._process(
            images,
            preset,
            dlaa_intensity,
            texture_intensity,
            motion_stability,
            video_chunks=True,
            state=previous_state,
        )


NODE_CLASS_MAPPINGS = {
    "VideoTAADLAA": VideoTAADLAA,
    "VideoTAADLAAChunks": VideoTAADLAAChunks,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoTAADLAA": "🎮 Video Anti-Aliasing (TAA + DLAA)",
    "VideoTAADLAAChunks": "Video Anti-Aliasing (TAA + DLAA) – Video Chunks",
}
