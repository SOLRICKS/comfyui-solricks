"""Compare an installed VideoTAADLAA node with bypass and an optional older version."""

import argparse
import datetime
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import safetensors
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taadlaa_validation_support import load_standalone, select_device, sha256

PRESETS = ("Performance", "Balanced", "High Detail", "Auto")
LUMA = torch.tensor((0.2126, 0.7152, 0.0722), dtype=torch.float64)
SAMPLES_PER_AXIS = 8


def size_arg(value):
    match = re.fullmatch(r"([1-9]\d*)[xX]([1-9]\d*)", value)
    if not match:
        raise argparse.ArgumentTypeError("use WIDTHxHEIGHT, for example 128x128")
    return tuple(map(int, match.groups()))


def render_pattern(width, height, frames, samples):
    """Render the same continuous shapes at pixel centers or on a subpixel grid."""
    y, x = torch.meshgrid(
        (torch.arange(height * samples, dtype=torch.float64) + 0.5) / samples,
        (torch.arange(width * samples, dtype=torch.float64) + 0.5) / samples,
        indexing="ij",
    )
    angles = (0.0, 1.5, 2.3)
    sequence = []
    for t in range(frames):
        px = x - 0.375 * t
        py = y - 0.25 * t
        background = torch.stack(
            [
                0.30
                + 0.08 * torch.sin(0.16 * px + 0.07 * py + phase)
                + 0.06 * torch.cos(0.08 * py - phase)
                for phase in angles
            ],
            dim=-1,
        )
        # Wires and a pair of rings lie over a smooth, textured background.
        slats = ((px + 0.63 * py).remainder(5.7) < 1.15) & (px < 0.48 * width)
        radius = ((px - 0.73 * width).square() + (py - 0.43 * height).square()).sqrt()
        rings = ((radius - height * 0.22).abs() < 0.7) | (
            (radius - height * 0.30).abs() < 0.85
        )
        panel = (py > 0.61 * height + 0.31 * (px - 0.6 * width)) & (px >= 0.48 * width)
        image = torch.where(panel.unsqueeze(-1), background * 0.30 + 0.41, background)
        image = torch.where(
            slats.unsqueeze(-1), image.new_tensor((0.90, 0.84, 0.66)), image
        )
        image = torch.where(
            rings.unsqueeze(-1), image.new_tensor((0.87, 0.54, 0.35)), image
        )
        sequence.append(
            image.reshape(height, samples, width, samples, 3).mean((1, 3)).float()
        )
    return torch.stack(sequence)


def reference_scores(image, reference):
    # All components use the same display-referred [0,1] values. The temporal
    # metric measures errors in frame differences including true scene motion.
    candidate = image.double()
    target = reference.double()
    error = candidate - target
    mse = float(error.square().mean())
    luma_error = (error * LUMA).sum(-1)
    scores = {
        "rgb_mse": mse,
        "rgb_psnr_db": -10 * math.log10(mse) if mse else None,
        "luma_frame_diff_error_mae_255": float(
            luma_error.diff(dim=0).abs().mean() * 255
        ),
    }
    if min(image.shape[1:3]) >= 11:
        scores["luma_ssim"] = luma_ssim(
            (candidate * LUMA).sum(-1), (target * LUMA).sum(-1)
        )
    return scores


def luma_ssim(left, right):
    """Mean SSIM on valid pixels, 11x11 Gaussian window, sigma 1.5, range 1."""
    axis = torch.arange(-5, 6, dtype=left.dtype)
    weights = torch.exp(-axis.square() / (2 * 1.5**2))
    weights /= weights.sum()
    kernel = torch.outer(weights, weights).view(1, 1, 11, 11)

    def average(value):
        return F.conv2d(value.unsqueeze(1), kernel)

    mx, my = average(left), average(right)
    vx = (average(left.square()) - mx.square()).clamp_min(0)
    vy = (average(right.square()) - my.square()).clamp_min(0)
    cov = average(left * right) - mx * my
    score = ((2 * mx * my + 0.01**2) * (2 * cov + 0.03**2)) / (
        (mx.square() + my.square() + 0.01**2) * (vx + vy + 0.03**2)
    )
    return float(score.mean())


def source_change(image, source):
    error = image.double() - source.double()
    return {
        "rgb_change_mae_255": float(error.abs().mean() * 255),
        "signed_luma_change_255": float((error * LUMA).sum(-1).mean() * 255),
    }


def video_clip(path, start, count, crop):
    metadata = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ]
    )
    stream = json.loads(metadata)["streams"][0]
    native_width, native_height = stream["width"], stream["height"]
    width, height = crop or (native_width, native_height)
    if width > native_width or height > native_height:
        raise ValueError(
            f"Crop {width}x{height} exceeds source {native_width}x{native_height}"
        )
    x, y = (native_width - width) // 2, (native_height - height) // 2
    # Crop after RGB conversion to keep exact offsets on subsampled sources.
    filters = f"trim=start_frame={start}:end_frame={start + count},format=rgb24,crop={width}:{height}:{x}:{y}"
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-noautorotate",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vf",
            filters,
            "-frames:v",
            str(count),
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    if len(raw) != count * height * width * 3:
        raise ValueError(f"Need {count} consecutive frames starting at index {start}")
    images = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(
        count, height, width, 3
    )
    return images.float().div_(255), {
        "file": path.name,
        "sha256": sha256(path),
        "native_stream": stream,
        "first_frame_zero_based": start,
        "crop_xywh": [x, y, width, height],
        "decode": "FFmpeg RGB24; native pixel scale; consecutive frames; no audio",
    }


def make_video_pair(images):
    """A 3x3 area target and its phase-aligned point-sampled counterpart."""
    frames, height, width, channels = images.shape
    if height % 3 or width % 3:
        raise ValueError("Native reference crop dimensions must be multiples of three")
    blocks = images.reshape(frames, height // 3, 3, width // 3, 3, channels)
    return blocks[:, :, 1, :, 1, :].contiguous(), blocks.mean((2, 4))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(node, images, preset, intensity, device, warmups, repeats):
    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for _ in range(warmups):
            (warmup,) = node.execute(images, preset, dlaa_intensity=intensity)
            del warmup
        synchronize(device)
        elapsed = []
        peak_allocated = []
        peak_reserved = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            (output,) = node.execute(images, preset, dlaa_intensity=intensity)
            synchronize(device)
            elapsed.append(time.perf_counter() - start)
            if device.type == "cuda":
                peak_allocated.append(torch.cuda.max_memory_allocated(device))
                peak_reserved.append(torch.cuda.max_memory_reserved(device))
            if output.shape != images.shape or output.dtype != images.dtype:
                raise AssertionError(f"{preset}: RGB shape/dtype changed")
            if not torch.isfinite(output).all() or output.min() < 0 or output.max() > 1:
                raise AssertionError(
                    f"{preset}: invalid output range or non-finite value"
                )
    median = statistics.median(elapsed)
    return output, {
        "trial_seconds": elapsed,
        "median_seconds": median,
        "median_ms_per_frame": median * 1000 / len(images),
        "frames_per_second": len(images) / median,
        "peak_cuda_allocated_mib": max(peak_allocated) / 2**20
        if peak_allocated
        else None,
        "peak_cuda_reserved_mib": max(peak_reserved) / 2**20 if peak_reserved else None,
    }


def fingerprints(root):
    paths = [
        root / name
        for name in (
            "VideoTAADLAA.py",
            "config.py",
            "model.py",
            "presets.py",
            "taa.py",
            "utils.py",
        )
    ]
    for stem in ("DLAANet", "DLAATexture"):
        weight = next(
            (
                root / f"{stem}.{ext}"
                for ext in ("safetensors", "pth")
                if (root / f"{stem}.{ext}").is_file()
            ),
            None,
        )
        if weight:
            paths.append(weight)
    return {path.name: sha256(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--node-dir", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--baseline-node-dir", type=Path, help="older package, compared at Balanced"
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument(
        "--intensity", type=float, default=1.0, help="master DLAA intensity (0 to 2)"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="untimed whole-clip runs per row, 0 for cold execution",
    )
    parser.add_argument("--presets", nargs="+", choices=PRESETS, default=list(PRESETS))
    parser.add_argument(
        "--synthetic-size", type=size_arg, default=(96, 64), metavar="WIDTHxHEIGHT"
    )
    parser.add_argument(
        "--video", type=Path, help="optional local clip, requires FFmpeg and ffprobe"
    )
    parser.add_argument(
        "--start-frame", type=int, default=0, help="zero-based, used with --video"
    )
    size = parser.add_mutually_exclusive_group()
    size.add_argument(
        "--center-crop", type=size_arg, default=(128, 128), metavar="WIDTHxHEIGHT"
    )
    size.add_argument(
        "--full-frame", action="store_true", help="use all native pixels in --video"
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("edges_pan", "clean_control", "video", "video_pair"),
        help="default: both synthetic cases, plus video if supplied",
    )
    parser.add_argument("--output", type=Path, default=Path("taadlaa_benchmark.json"))
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error("--warmups must be nonnegative")
    if not 0 <= args.intensity <= 2:
        parser.error("--intensity must be between 0 and 2")
    if args.frames < 2 or args.repeats < 1 or args.start_frame < 0 or args.threads < 1:
        parser.error(
            "need at least 2 frames, 1 repeat, 1 thread, and a nonnegative start frame"
        )
    chosen = args.cases or (
        ["edges_pan", "clean_control", "video"]
        if args.video
        else ["edges_pan", "clean_control"]
    )
    if any(case in chosen for case in ("video", "video_pair")) and not args.video:
        parser.error("video and video_pair require --video")
    if args.full_frame and "video_pair" in chosen:
        parser.error(
            "video_pair uses --center-crop for its output size (3x native source area per axis)"
        )
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    torch.set_num_threads(args.threads)
    torch.manual_seed(913)
    jobs = [("current", args.node_dir.resolve(), args.presets)]
    if args.baseline_node_dir:
        jobs.append(("baseline", args.baseline_node_dir.resolve(), ["Balanced"]))
    report = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "safetensors": safetensors.__version__,
        "device": device_name,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "logical_cpu_count": os.cpu_count(),
        "torch_cpu_threads": torch.get_num_threads(),
        "frames_per_case": args.frames,
        "warmups_per_row": args.warmups,
        "timed_runs_per_row": args.repeats,
        "node_settings": {
            "dlaa_intensity": args.intensity,
            "texture_intensity": 1.0,
            "motion_stability": 1.0,
        },
        "mode": "standalone, public execute(), CPU input and output; timing excludes decode and metrics",
        "node_files_sha256": {tag: fingerprints(root) for tag, root, _ in jobs},
        "cases": {},
    }
    cases = []
    if any(case in chosen for case in ("edges_pan", "clean_control")):
        width, height = args.synthetic_size
        reference = render_pattern(width, height, args.frames, SAMPLES_PER_AXIS)
        if "edges_pan" in chosen:
            aliased = render_pattern(width, height, args.frames, 1)
            cases.append(("edges_pan", aliased, reference, {}))
        if "clean_control" in chosen:
            cases.append(("clean_control", reference, reference, {}))
        report["synthetic"] = {
            "reference_samples_per_axis": SAMPLES_PER_AXIS,
            "pan_xy_px_per_frame": [0.375, 0.25],
        }
    if "video" in chosen:
        images, metadata = video_clip(
            args.video,
            args.start_frame,
            args.frames,
            None if args.full_frame else args.center_crop,
        )
        cases.append(("video", images, None, metadata))
    if "video_pair" in chosen:
        native_crop = tuple(3 * side for side in args.center_crop)
        native, metadata = video_clip(
            args.video, args.start_frame, args.frames, native_crop
        )
        point, area = make_video_pair(native)
        metadata["comparison"] = (
            "Same RGB24 source, center pixel vs 3x3 mean of each native block"
        )
        cases.append(("video_pair", point, area, metadata))
    for case, images, truth, metadata in cases:
        entry = {"size_wh": [images.shape[2], images.shape[1]], **metadata, "rows": []}
        report["cases"][case] = entry
        bypass = {"node": "bypass", **source_change(images, images)}
        if truth is not None:
            bypass.update(reference_scores(images, truth))
        entry["rows"].append(bypass)
    modules = {tag: load_standalone(root, tag) for tag, root, _ in jobs}
    for case, images, truth, _ in cases:
        for tag, _, presets in jobs:
            module = modules[tag]
            # The clean control uses one preset so it does not dominate run time.
            selected = ["Balanced"] if case == "clean_control" else presets
            with select_device(module, device):
                for preset in selected:
                    node = module.NODE_CLASS_MAPPINGS["VideoTAADLAA"]()
                    output, timing = measure(
                        node,
                        images,
                        preset,
                        args.intensity,
                        device,
                        args.warmups,
                        args.repeats,
                    )
                    row = {
                        "node": tag,
                        "preset": preset,
                        **timing,
                        **source_change(output, images),
                    }
                    if truth is not None:
                        row.update(reference_scores(output, truth))
                    report["cases"][case]["rows"].append(row)
                    print(
                        f"{case} / {tag} / {preset}: {timing['median_ms_per_frame']:.1f} ms/frame",
                        flush=True,
                    )
                    del node, output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
