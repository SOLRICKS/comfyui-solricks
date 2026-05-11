# Video Adaptive Anti-aliasing
# v0.1.0
# SOLRICKS
# AI-assisted


import torch
import torch.nn.functional as F


try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

try:
    import comfy.model_management as mm
except ImportError:
    mm = None


EDGE_EPSILON = 1e-6
EDGE_FADE_WIDTH = 0.10
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)

SOBEL_X = (
    (-1.0, 0.0, 1.0),
    (-2.0, 0.0, 2.0),
    (-1.0, 0.0, 1.0),
)

SOBEL_Y = (
    (-1.0, -2.0, -1.0),
    (0.0, 0.0, 0.0),
    (1.0, 2.0, 1.0),
)


class VideoAdaptiveAA:
    def __init__(self):
        self._sobel_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "edge_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blur_radius": ("INT", {"default": 1, "min": 0, "max": 3, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_aa"
    CATEGORY = "CustomPostProcess"

    def _get_sobel_kernels(self, dtype, device):
        cache_key = (dtype, str(device))

        if cache_key not in self._sobel_cache:
            sobel_x = torch.tensor(SOBEL_X, dtype=dtype, device=device).view(1, 1, 3, 3)
            sobel_y = torch.tensor(SOBEL_Y, dtype=dtype, device=device).view(1, 1, 3, 3)

            self._sobel_cache[cache_key] = (sobel_x, sobel_y)

        return self._sobel_cache[cache_key]

    def _apply_aa_frame(self, frame, strength, edge_threshold, blur_radius):
        img = frame.permute(0, 3, 1, 2)

        if img.shape[1] not in (3, 4):
            raise ValueError("VideoAdaptiveAA expects RGB or RGBA images.")

        has_alpha = img.shape[1] == 4

        if has_alpha:
            alpha = img[:, 3:4, :, :]
            img_rgb = img[:, :3, :, :]
        else:
            img_rgb = img

        img_gray = (
            LUMA_WEIGHTS[0] * img_rgb[:, 0:1, :, :]
            + LUMA_WEIGHTS[1] * img_rgb[:, 1:2, :, :]
            + LUMA_WEIGHTS[2] * img_rgb[:, 2:3, :, :]
        )

        sobel_x, sobel_y = self._get_sobel_kernels(img.dtype, img.device)

        edges_x = F.conv2d(img_gray, sobel_x, padding=1)
        edges_y = F.conv2d(img_gray, sobel_y, padding=1)

        edges = torch.sqrt(edges_x.square() + edges_y.square())

        edge_max = edges.amax(dim=(1, 2, 3), keepdim=True).clamp_min(EDGE_EPSILON)
        edges = edges / edge_max

        edge_mask = torch.clamp(
            (edges - edge_threshold) / EDGE_FADE_WIDTH,
            0.0,
            1.0,
        )

        edge_mask = torch.clamp(edge_mask * strength, 0.0, 1.0)

        kernel_size = blur_radius * 2 + 1

        img_padded = F.pad(
            img_rgb,
            (blur_radius, blur_radius, blur_radius, blur_radius),
            mode="replicate",
        )

        blurred = F.avg_pool2d(
            img_padded,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )

        result_rgb = img_rgb * (1.0 - edge_mask) + blurred * edge_mask
        result_rgb = torch.clamp(result_rgb, 0.0, 1.0)

        if has_alpha:
            result = torch.cat([result_rgb, alpha], dim=1)
        else:
            result = result_rgb

        return result.permute(0, 2, 3, 1).contiguous()

    @torch.inference_mode()
    def apply_aa(self, images, strength, edge_threshold, blur_radius):
        if strength <= 0.0 or blur_radius <= 0:
            return (images,)

        frame_count = images.shape[0]
        pbar = ProgressBar(frame_count) if ProgressBar is not None else None

        out_tensor = torch.empty_like(images)

        for i in range(frame_count):
            if mm is not None and hasattr(mm, "throw_exception_if_processing_interrupted"):
                mm.throw_exception_if_processing_interrupted()

            frame = images[i:i + 1]
            frame_out = self._apply_aa_frame(
                frame,
                strength,
                edge_threshold,
                blur_radius,
            )

            if frame_out.dtype != out_tensor.dtype or frame_out.device != out_tensor.device:
                frame_out = frame_out.to(device=out_tensor.device, dtype=out_tensor.dtype)

            out_tensor[i:i + 1].copy_(frame_out)

            if pbar is not None:
                pbar.update(1)

        return (out_tensor,)


NODE_CLASS_MAPPINGS = {
    "VideoAdaptiveAA": VideoAdaptiveAA
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoAdaptiveAA": "✨ Video Adaptive Anti-Aliasing"
}