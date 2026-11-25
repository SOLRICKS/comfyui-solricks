"""CPU regression tests for VideoTAADLAA v0.2.3 and its Video Chunks node."""

import argparse
import logging
import math
import shutil
import sys
import tempfile
import types
import unittest
import weakref
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_taadlaa import make_video_pair, reference_scores, render_pattern
from taadlaa_validation_support import load_standalone, select_device

NODE_DIR = Path(__file__).resolve().parents[1]


def setUpModule():
    global node_module
    node_module = load_standalone(NODE_DIR, "regressions")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    unittest.addModuleCleanup(torch.set_num_threads, previous_threads)
    selected = select_device(node_module, "cpu")
    selected.__enter__()
    unittest.addModuleCleanup(selected.__exit__, None, None, None)


class FakePatcher:
    def __init__(self, model, load_device, offload_device):
        self.model = model
        self.load_device = load_device
        self.offload_device = offload_device


class FakeManager:
    def __init__(self):
        self.calls = []
        self.loaded = []
        self.cancel = False

    def get_torch_device(self):
        return torch.device("cpu")

    def load_models_gpu(self, models, **kwargs):
        self.calls.append((models, kwargs))
        for p in models:
            p.model.to(p.load_device)
            if p not in self.loaded:
                self.loaded.append(p)

    def unload_all_models(self):
        for p in self.loaded:
            p.model.to(p.offload_device)
        self.loaded.clear()

    def throw_exception_if_processing_interrupted(self):
        if self.cancel:
            raise KeyboardInterrupt("test cancellation")

    def soft_empty_cache(self):
        pass


class DLAARegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(913)
        cls.node = node_module.VideoTAADLAA()
        cls.net = cls.node._net(torch.device("cpu")).net
        cls.images = torch.rand(2, 12, 18, 3)
        cls.image = torch.rand(1, 3, 17, 25)

    def test_all_presets_and_run_state(self):
        for preset in [
            "Auto",
            "Performance",
            "Balanced",
            "High Detail",
            "Smooth",
            "Photo",
        ]:
            with self.subTest(preset=preset):
                out = self.node.execute(self.images, preset)[0]
                self.assertEqual(out.shape, self.images.shape)
                self.assertEqual(out.dtype, self.images.dtype)
                self.assertEqual(out.device.type, "cpu")
                self.assertTrue(torch.isfinite(out).all())
                self.assertGreaterEqual(out.min().item(), 0)
                self.assertLessEqual(out.max().item(), 1)
                torch.testing.assert_close(
                    out, self.node.execute(self.images, preset)[0], rtol=0, atol=0
                )

    def test_grayscale_and_extra_channels(self):
        for channels in [1, 3, 4, 6]:
            for dtype in [torch.float16, torch.float32, torch.float64]:
                with self.subTest(channels=channels, dtype=dtype):
                    image = torch.rand(1, 9, 11, channels).to(dtype)
                    out = self.node.execute(image, "Auto")[0]
                    self.assertEqual(out.shape, (1, 9, 11, max(channels, 3)))
                    self.assertEqual(out.dtype, dtype)
                    if channels > 3:
                        self.assertTrue(torch.equal(out[..., 3:], image[..., 3:]))

    def test_invalid_and_empty_inputs(self):
        for image in [
            torch.rand(2, 3),
            torch.rand(1, 5, 5, 2),
            torch.rand(1, 0, 5, 3),
            torch.rand(1, 5, 5, 0),
            torch.ones(1, 5, 5, 3, dtype=torch.uint8),
            None,
        ]:
            with (
                self.subTest(shape=getattr(image, "shape", None)),
                self.assertRaises(ValueError),
            ):
                self.node.execute(image, "Auto")
        empty = self.node.execute(torch.empty(0, 5, 6, 1), "Auto")[0]
        self.assertEqual(empty.shape, (0, 5, 6, 3))

    def test_small_axes(self):
        for h, w in [(1, 1), (1, 5), (5, 1), (2, 2), (3, 4)]:
            for preset in ["Auto", "High Detail", "Performance"]:
                with self.subTest(shape=(h, w), preset=preset):
                    out = self.node.execute(torch.rand(2, h, w, 3), preset)[0]
                    self.assertTrue(torch.isfinite(out).all())

    def test_subpixel_zero_offset_identity(self):
        node = node_module.VideoTAADLAA()
        node.settings = {**node.settings, "detail_subpixel_sample_scale": 0.0}
        for h, w in [(17, 25), (1, 9), (9, 1), (1, 1)]:
            image = torch.rand(1, 3, h, w)
            cache = {}
            out = node._subpixel_edge_reconstruction(
                image, self.net, 0, 1, grid_cache=cache
            )
            self.assertIs(out, image)
            self.assertEqual(len(cache), 0)
            node._subpixel_edge_reconstruction(
                image, self.net, 0, 1, grid_cache=cache
            )  # disabled pass stays allocation-free
        node.settings = {**node.settings, "detail_subpixel_sample_scale": 0.35}
        cache = {}
        node._subpixel_edge_reconstruction(self.image, self.net, 0, 1, grid_cache=cache)
        grid = next(iter(cache.values()))
        theta = torch.eye(2, 3).unsqueeze(0)
        expected_grid = F.affine_grid(theta, self.image.shape, align_corners=False)
        torch.testing.assert_close(grid, expected_grid, rtol=0, atol=2e-7)

    def test_tiling_and_logs_use_actual_overlap(self):
        image = torch.rand(2, 3, 139, 173)
        for overlap in [0, 8, 16, 24]:
            sizes = []

            def identity(tile, seen=sizes):
                seen.append(tile.shape)
                return tile

            out = self.node._tiled_forward(identity, image, 64, overlap)
            torch.testing.assert_close(out, image, rtol=2e-6, atol=2e-7)
            count = len(node_module.tile_starts(139, 64, overlap)) * len(
                node_module.tile_starts(173, 64, overlap)
            )
            self.assertEqual(len(sizes), count)
            with self.assertLogs("VideoTAADLAA", level="DEBUG") as logs:
                self.node._log_tiling(139, 173, 64, overlap)
            self.assertIn(f"{count} tiles", logs.output[0])
        for tile, overlap in [(0, 0), (64, -1), (64, 32)]:
            with self.assertRaises(ValueError):
                self.node._tiled_forward(lambda x: x, image, tile, overlap)

    def test_oom_retry_both_exception_forms(self):
        for error_type in [RuntimeError, torch.OutOfMemoryError]:
            calls = []

            def forward(net, image, tile_size, overlap, seen=calls, error=error_type):
                seen.append(tile_size)
                if tile_size > 128:
                    raise error("CUDA out of memory")
                return image

            with patch.object(self.node, "_tiled_forward", side_effect=forward):
                out, tile = self.node._run_dlaa_with_retry(
                    self.net, self.image, 512, False, 0, 16
                )
            self.assertEqual(calls, [512, 256, 128])
            self.assertEqual(tile, 128)
            self.assertIs(out, self.image)
        calls.clear()
        with patch.object(self.node, "_tiled_forward", side_effect=forward):
            self.node._run_dlaa_with_retry(self.net, self.image, 1536, False, 0)
        self.assertEqual(calls, [1536, 768, 384, 192, 128])
        with (
            patch.object(
                self.node, "_tiled_forward", side_effect=RuntimeError("invalid shape")
            ),
            self.assertRaisesRegex(RuntimeError, "invalid shape"),
        ):
            self.node._run_dlaa_with_retry(self.net, self.image, 512, False, 0)
        with (
            patch.object(
                self.node,
                "_tiled_forward",
                side_effect=torch.OutOfMemoryError("terminal"),
            ),
            self.assertLogs(node_module.log, level="ERROR") as logs,
            self.assertRaises(torch.OutOfMemoryError),
        ):
            self.node._run_dlaa_with_retry(self.net, self.image, 128, False, 0)
        self.assertIn("minimum tile size 128", logs.output[0])

    def test_oom_tracebacks_release_failed_tensors(self):
        refs = []

        def forward(net, image, tile_size, overlap):
            if tile_size > 128:
                activation = torch.empty(4)
                refs.append(weakref.ref(activation))
                raise RuntimeError("out of memory")
            return image

        def clear():
            self.assertTrue(all(ref() is None for ref in refs))

        with (
            patch.object(self.node, "_tiled_forward", side_effect=forward),
            patch.object(node_module, "clear_device_cache", side_effect=clear),
        ):
            self.node._run_dlaa_with_retry(self.net, self.image, 512, False, 0)

    def test_texture_oom_and_non_oom_errors(self):
        kwargs = {
            "texture_net": self.net,
            "dlaa_out": self.image,
            "tile_size": 512,
            "motion_gate": 0,
            "dark_mask": torch.ones(1, 1, 17, 25),
            "highlight_mask": torch.zeros(1, 1, 17, 25),
            "preset": "Balanced",
        }
        for error in [
            RuntimeError("out of memory"),
            torch.OutOfMemoryError("allocation failed"),
        ]:
            with patch.object(self.node, "_tiled_forward", side_effect=error):
                self.assertIs(self.node._apply_texture_pass(**kwargs), self.image)
        with (
            patch.object(
                self.node, "_tiled_forward", side_effect=RuntimeError("size mismatch")
            ),
            self.assertRaisesRegex(RuntimeError, "size mismatch"),
        ):
            self.node._apply_texture_pass(**kwargs)

    def test_no_stale_spatial_cache(self):
        cached = node_module.ImageFeatures(self.image, self.net)
        self.assertIs(cached.luma(), cached.luma())
        self.assertIs(cached.blur(3), cached.blur(3))
        self.assertIs(node_module.image_features(self.image, self.net, cached), cached)
        changed = self.image * 0.8
        new = node_module.image_features(changed, self.net, cached)
        self.assertIsNot(new, cached)
        torch.testing.assert_close(new.luma(), node_module.rgb_luma(changed))
        expected = F.avg_pool2d(F.pad(changed, [1] * 4, mode="reflect"), 3, stride=1)
        torch.testing.assert_close(new.blur(3), expected)

    def test_cache_optimization_preserves_computation(self):
        image = self.images
        expected = self.node.execute(image, "High Detail")[0]
        blur = node_module.ImageFeatures.blur
        luma = node_module.ImageFeatures.luma
        edge = node_module.ImageFeatures.luma_edge

        def uncached_blur(obj, *args, **kwargs):
            obj._blurs.clear()
            return blur(obj, *args, **kwargs)

        def uncached_luma(obj, *args, **kwargs):
            obj._luma = None
            return luma(obj, *args, **kwargs)

        def uncached_edge(obj, *args, **kwargs):
            obj._luma_edge = None
            return edge(obj, *args, **kwargs)

        with (
            patch.object(node_module.ImageFeatures, "blur", uncached_blur),
            patch.object(node_module.ImageFeatures, "luma", uncached_luma),
            patch.object(node_module.ImageFeatures, "luma_edge", uncached_edge),
        ):
            actual = self.node.execute(image, "High Detail")[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_managed_registration_unload_reload(self):
        manager = FakeManager()
        with (
            patch.object(node_module, "mm", manager),
            patch.object(
                node_module,
                "model_patcher",
                types.SimpleNamespace(ModelPatcher=FakePatcher),
            ),
        ):
            node = node_module.VideoTAADLAA()
            first = node.execute(self.images[:1], "Balanced")[0]
            self.assertEqual(len(manager.loaded), 2)
            entries = list(node.net_cache.values()) + list(
                node.texture_net_cache.values()
            )
            for entry in entries:
                self.assertIsInstance(entry.patcher, FakePatcher)
                self.assertEqual(entry.patcher.offload_device.type, "cpu")
            self.assertTrue(manager.calls[-1][1]["force_full_load"])
            self.assertGreater(manager.calls[-1][1]["memory_required"], 0)
            manager.unload_all_models()
            second = node.execute(self.images[:1], "Balanced")[0]
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            self.assertEqual(
                entries,
                list(node.net_cache.values()) + list(node.texture_net_cache.values()),
            )
            manager.cancel = True
            with self.assertRaises(KeyboardInterrupt):
                node.execute(self.images, "High Detail")

    def test_standalone_error_cleanup(self):
        node = node_module.VideoTAADLAA()
        runs = []

        def fail(*args):
            run = args[4]
            run.grid_cache["temporary"] = torch.zeros(1)
            runs.append(run)
            raise RuntimeError("intentional failure")

        with (
            patch.object(node, "_apply_dlaa_pipeline", side_effect=fail),
            patch.object(
                node, "_release_unmanaged_models", wraps=node._release_unmanaged_models
            ) as release,
        ):
            with self.assertRaisesRegex(RuntimeError, "intentional failure"):
                node.execute(self.images, "High Detail")
            release.assert_called_once()
        self.assertEqual(runs[0].grid_cache, {})
        self.assertTrue(
            all(
                p.device.type == "cpu"
                for entry in node.net_cache.values()
                for p in entry.net.parameters()
            )
        )

    def test_successful_tile_persists(self):
        calls = []

        def forward(net, image, tile_size, overlap):
            if net is self.net:
                calls.append(tile_size)
                if tile_size > 128:
                    raise RuntimeError("out of memory")
            return image

        with patch.object(self.node, "_tiled_forward", side_effect=forward):
            self.node.execute(self.images, "High Detail", texture_intensity=0)
        self.assertEqual(calls, [512, 256, 128, 128])

    def test_legacy_inputs_and_texture_off(self):
        image = self.images[:1]
        expected = self.node.execute(image, "High Detail", dlaa_intensity=0.6)[0]
        for alias in ["Detail", "Sharp", "High Detail"]:
            actual = self.node.execute(image, alias, detail_intensity=0.6)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        node = node_module.VideoTAADLAA()
        node.execute(image, "Balanced", texture_intensity=0)
        self.assertEqual(len(node.texture_net_cache), 0)


def rgb_view(hwc):
    return hwc.permute(0, 3, 1, 2).contiguous()


def chunk_history(frame):
    return node_module.ClipHistory(
        frame, frame.clone(), frame.clone(), 1, 512, ("Auto", 1.0, 1.0, 1.0)
    )


def warped(frame, dx, dy):
    h, w = frame.shape[-2:]
    grid = node_module.pixel_grid(frame, {}) - frame.new_tensor(
        (2 * dx / w, 2 * dy / h)
    )
    return F.grid_sample(frame, grid, align_corners=False, padding_mode="border")


class RevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = torch.Generator().manual_seed(78123)
        cls.node = node_module.VideoTAADLAA()
        cls.chunks = node_module.VideoTAADLAAChunks()
        cls.weights = load_file(str(NODE_DIR / "DLAANet.safetensors"))
        h, w = 29, 39
        canvas = 0.12 + 0.65 * torch.rand(1, h + 8, w + 11, 3, generator=generator)
        a = canvas[:, 3 : 3 + h, 4 : 4 + w]
        pan = canvas[:, 4 : 4 + h, 2 : 2 + w]
        cut = torch.rand(a.shape, generator=generator)
        cls.clip = torch.cat(
            (a, a, a + 0.032, pan + 0.032, cut, cut, cut + 0.007)
        ).clamp(0, 1)

    def test_zero_is_bit_exact_and_allocation_free_before_models(self):
        with (
            patch.object(
                self.node, "_get_device", side_effect=AssertionError("device requested")
            ),
            patch.object(
                self.node, "_net", side_effect=AssertionError("model requested")
            ),
        ):
            for channels in (1, 3, 4, 6):
                x = torch.linspace(
                    -0.12, 1.2, 2 * 5 * 7 * channels, dtype=torch.float64
                ).view(2, 5, 7, channels)
                for kwargs in ({"dlaa_intensity": 0}, {"detail_intensity": 0}):
                    (out,) = self.node.execute(x, "Auto", **kwargs)
                    self.assertTrue(torch.equal(x, out))
                    self.assertEqual(out.dtype, x.dtype)
                    self.assertEqual(out.data_ptr(), x.data_ptr())
        bypass, state = self.chunks.process_chunk(self.clip[:1], "Auto", 0)
        self.assertTrue(torch.equal(bypass, self.clip[:1]))
        self.assertIsNone(state)

    def test_master_blend_covers_taa_texture_and_whole_output(self):
        for name in ("Auto", "Performance", "High Detail", "Balanced"):
            with self.subTest(preset=name):
                x = self.clip[:4]
                (full,) = self.node.execute(x, name)
                self.assertGreater((full - x).abs().mean().item(), 0.001)
                for intensity in (0.25, 1.5):
                    (actual,) = self.node.execute(x, name, dlaa_intensity=intensity)
                    expected = torch.lerp(x, full, intensity).clamp(0, 1)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-7)

    def test_strict_checkpoint_loads_official_convolutions_and_norms(self):
        net = node_module.load_checkpoint("DLAANet")
        for name, parameter in net.named_parameters():
            self.assertTrue(torch.equal(parameter, self.weights[name].float()), name)
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(node_module, "__file__", str(Path(d) / "VideoTAADLAA.py")),
        ):
            p = Path(d) / "DLAANet.safetensors"
            for k, op in (
                ("enc1.0.weight", "remove"),
                ("dec.1.bias", "reshape"),
                ("foreign.weight", "extra"),
            ):
                weights = dict(self.weights)
                if op == "remove":
                    del weights[k]
                elif op == "reshape":
                    weights[k] = weights[k][:2].contiguous()
                else:
                    weights[k] = torch.zeros(3)
                save_file(weights, str(p))
                node = node_module.VideoTAADLAA()
                with self.assertRaisesRegex(ValueError, "DLAANet.safetensors") as error:
                    node.execute(self.clip[:1], "Balanced")
                self.assertIn(k, str(error.exception))
                self.assertFalse(node.net_cache)
            weights = dict(self.weights)
            del weights["jitter_offsets"]
            save_file(weights, str(p))
            self.assertTrue(
                torch.equal(
                    node_module.load_checkpoint("DLAANet").enc1[0].weight,
                    net.enc1[0].weight,
                )
            )

    def test_pth_wrapper_uses_weights_only_and_surfaces_loader_errors(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(node_module, "__file__", str(Path(d) / "VideoTAADLAA.py")),
        ):
            p = Path(d) / "DLAANet.pth"
            torch.save({"state_dict": dict(self.weights)}, p)
            with patch.object(torch, "load", wraps=torch.load) as loader:
                net = node_module.load_checkpoint("DLAANet")
            self.assertEqual(loader.call_count, 1)
            self.assertIs(loader.call_args.kwargs["weights_only"], True)
            self.assertTrue(
                torch.equal(
                    net.reconstructor.weight,
                    self.weights["reconstructor.weight"].float(),
                )
            )
            with patch.object(
                torch, "load", side_effect=TypeError("bad value")
            ) as loader:
                with self.assertRaisesRegex(TypeError, "bad value"):
                    node_module.load_checkpoint("DLAANet")
                self.assertEqual(loader.call_count, 1)

    def test_optional_texture_fails_closed(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(node_module, "__file__", str(Path(d) / "VideoTAADLAA.py")),
        ):
            d = Path(d)
            shutil.copyfile(NODE_DIR / "DLAANet.safetensors", d / "DLAANet.safetensors")
            save_file(
                {"reconstructor.weight": self.weights["reconstructor.weight"]},
                str(d / "DLAATexture.safetensors"),
            )
            node = node_module.VideoTAADLAA()
            with self.assertLogs(node_module.log, "ERROR") as logs:
                (actual,) = node.execute(self.clip[:1], "Balanced")
                (again,) = node.execute(self.clip[:1], "Balanced")
            self.assertEqual(len(logs.records), 1)
            self.assertFalse(node.texture_net_cache)
            (expected,) = node.execute(self.clip[:1], "Balanced", texture_intensity=0)
            self.assertTrue(
                torch.equal(actual, expected) and torch.equal(again, expected)
            )

    def test_constant_edges_blur_sobel_detail_and_texture(self):
        net = self.node._net(torch.device("cpu")).net
        for h, w in ((1, 1), (1, 7), (2, 3), (19, 26)):
            frame = (
                torch.tensor([0.3125, 0.625, 0.8125])
                .view(1, 3, 1, 1)
                .expand(1, 3, h, w)
                .contiguous()
            )
            features = node_module.ImageFeatures(frame, net)
            for k in (3, 5, 7):
                torch.testing.assert_close(features.blur(k), frame, rtol=0, atol=1e-7)
            _, edge, dx, dy = features.luma_edge(return_grad=True)
            self.assertLess(float(torch.amax(dx.abs() + dy.abs())), 5e-7)
            self.assertLess(float((edge - edge[..., 0, 0]).abs().max()), 1e-7)
            mask = frame[:, :1] * 0
            detail, _ = self.node._apply_detail_pass(
                frame, net, "Detail", 1, 0.9, 0.03, 0.1, mask, features
            )
            self.assertTrue(torch.equal(detail, frame))
            with patch.object(self.node, "_tiled_forward", return_value=frame):
                textured = self.node._apply_texture_pass(
                    net, frame, 512, 0, mask + 1, mask, "Balanced"
                )
            torch.testing.assert_close(textured, frame, rtol=0, atol=1e-7)

    def test_both_axes_pan_with_disocclusion_and_low_contrast(self):
        h, w = 88, 127
        g = torch.Generator().manual_seed(33)
        canvas = 0.43 + 0.08 * torch.rand(1, 3, h + 32, w + 32, generator=g)
        old = canvas[:, :, 16 : 16 + h, 16 : 16 + w].contiguous()
        for dx, dy in ((6, -4), (-7, 5), (0, 0)):
            new = canvas[
                :, :, 16 - dy : 16 - dy + h, 16 - dx : 16 - dx + w
            ].contiguous()
            cache = {}
            motion = node_module.measure_camera_motion(new, old, cache)
            self.assertTrue(motion.matched)
            self.assertEqual((motion.dx, motion.dy), (dx, dy))
            aligned = node_module.align_history(
                new, chunk_history(old), motion, 0.08, cache
            )
            interior = (..., slice(10, -10), slice(10, -10))
            self.assertGreater(float(aligned.confidence[interior].mean()), 0.999)
            self.assertLess(
                float((aligned.taa[interior] - new[interior]).abs().mean()), 2e-6
            )
            if dx:
                side = (
                    aligned.confidence[..., :dx]
                    if dx > 0
                    else aligned.confidence[..., dx:]
                )
                self.assertEqual(torch.count_nonzero(side).item(), 0)
            if dy:
                side = (
                    aligned.confidence[..., :dy, :]
                    if dy > 0
                    else aligned.confidence[..., dy:, :]
                )
                self.assertEqual(torch.count_nonzero(side).item(), 0)

    def test_subpixel_and_fullhd_pans(self):
        for h, w, dx, dy in (
            (91, 127, 0.375, -0.625),
            (768, 1366, -5.375, 3.125),
            (1080, 1920, 13, -9),
        ):
            frame = torch.rand(1, 3, h, w, generator=torch.Generator().manual_seed(12))
            moved = warped(frame, dx, dy)
            motion = node_module.measure_camera_motion(moved, frame, {})
            with self.subTest(size=(h, w), offset=(dx, dy)):
                self.assertTrue(motion.matched)
                self.assertEqual((motion.dx, motion.dy), (dx, dy))

    def test_disagreeing_foreground_rejected_at_static_background(self):
        old = 0.15 + 0.5 * torch.rand(
            1, 3, 110, 133, generator=torch.Generator().manual_seed(444)
        )
        new = old.clone()
        new[:, :, 19:41, 47:75] += 0.3
        motion = node_module.measure_camera_motion(new, old, {})
        self.assertTrue(motion.matched)
        self.assertEqual((motion.dx, motion.dy), (0, 0))
        a = node_module.align_history(new, chunk_history(old), motion, 0.08, {})
        self.assertEqual(a.confidence[..., 20:40, 48:74].max().item(), 0)
        self.assertEqual(a.confidence[..., 55:70, 40:62].min().item(), 1)

    def test_cut_discards_history_across_all_temporal_paths(self):
        a, b = self.clip[:1], self.clip[4:5]
        motion = node_module.measure_camera_motion(rgb_view(b), rgb_view(a), {})
        self.assertFalse(motion.matched)
        self.assertGreater(motion.amount, self.node.settings["scene_cut_min_change"])
        for p in ("Auto", "High Detail", "Balanced", "Performance"):
            _, prior = self.chunks.process_chunk(a, p)
            fresh, state = self.chunks.process_chunk(b, p)
            after, cut_state = self.chunks.process_chunk(b, p, previous_state=prior)
            self.assertTrue(torch.equal(after, fresh), p)
            self.assertTrue(torch.equal(state.detail, cut_state.detail), p)
            self.assertTrue(torch.equal(state.taa, cut_state.taa), p)

    def test_reprojected_history_contributes_only_where_accepted_in_taa(self):
        h, w = 57, 73
        old = 0.20 + 0.40 * torch.rand(
            1, 3, h, w, generator=torch.Generator().manual_seed(727)
        )
        current = torch.roll(old, (2, 5), (-2, -1))
        current[:, :, 21:36, 31:52] += 0.27
        net = self.node._net(torch.device("cpu")).net
        cache = {}
        motion = node_module.measure_camera_motion(current, old, cache)
        self.assertEqual((motion.dx, motion.dy), (5, 2))
        aligned = node_module.align_history(
            current, chunk_history(old), motion, 0.08, cache
        )
        params = self.node._resolve_frame_config("Balanced", False, motion.amount)
        run = node_module.RunContext(
            512, 32, 1, 1, False, torch.cat((net.sobel_x, net.sobel_y)), cache
        )
        spatial = self.node._apply_taa(
            current, node_module.TAAState(), None, params, net, False, run
        )
        modified = replace(aligned, taa=aligned.taa + 0.009)
        base = self.node._apply_taa(
            current, node_module.TAAState(), aligned, params, net, False, run
        )
        out = self.node._apply_taa(
            current, node_module.TAAState(), modified, params, net, False, run
        )
        reject = aligned.confidence.expand_as(out) == 0
        self.assertTrue(torch.equal(out[reject], spatial[reject]))
        trusted = (aligned.confidence > 0.99).expand_as(out)
        raw_taa = node_module.TAAState()
        raw_taa.history = modified.taa
        accumulated = raw_taa.update(
            spatial, self.node.settings["taa_alpha"], params.motion_sensitivity
        )
        expected = torch.lerp(
            spatial,
            torch.lerp(spatial, accumulated, aligned.confidence),
            params.taa_strength,
        )
        torch.testing.assert_close(out[trusted], expected[trusted], rtol=0, atol=0)
        self.assertTrue(((out - base).abs()[trusted] > 1e-6).any().item())
        _, state = self.chunks.process_chunk(old.permute(0, 2, 3, 1), "Balanced")
        source = current.permute(0, 2, 3, 1)
        a, _ = self.chunks.process_chunk(source, "Balanced", previous_state=state)
        b, _ = self.chunks.process_chunk(
            source, "Balanced", previous_state=replace(state, taa=state.taa + 0.009)
        )
        self.assertGreater((a - b).abs().mean().item(), 1e-6)

    def test_tile_retry_survives_chunk_handoff(self):
        sizes = []

        def forward(net, image, tile_size, overlap):
            sizes.append(tile_size)
            if tile_size > 128:
                raise RuntimeError("out of memory")
            return image

        with patch.object(self.chunks, "_tiled_forward", side_effect=forward):
            _, s = self.chunks.process_chunk(
                self.clip[:1], "Balanced", texture_intensity=0
            )
            _, t = self.chunks.process_chunk(
                self.clip[1:3], "Balanced", texture_intensity=0, previous_state=s
            )
        self.assertEqual(sizes, [512, 256, 128, 128, 128])
        self.assertEqual((s.tile_size, t.tile_size), (128, 128))

    def test_chunk_partitions_match_full_execution_pixel_for_pixel(self):
        for p in ("Auto", "Balanced", "High Detail", "Performance", "Smooth", "Photo"):
            (expected,) = self.node.execute(self.clip, p)
            for sizes in ([1, 3, 1, 2], [2, 2, 3], [1] * len(self.clip)):
                with self.subTest(preset=p, partition=sizes):
                    prior = None
                    parts = []
                    pos = 0
                    for size in sizes:
                        part, prior = self.chunks.process_chunk(
                            self.clip[pos : pos + size], p, previous_state=prior
                        )
                        pos += size
                        parts.append(part)
                    self.assertEqual(prior.frames, len(self.clip))
                    self.assertTrue(torch.equal(torch.cat(parts), expected))

    def test_standalone_photo_and_single_frame_video_are_distinct(self):
        x = self.clip[:1]
        (photo,) = self.node.execute(x, "Auto")
        (expected_photo,) = self.node.execute(x, "Photo")
        self.assertTrue(torch.equal(photo, expected_photo))
        part, s = self.chunks.process_chunk(x, "Auto")
        (video,) = self.node.execute(self.clip[:2], "Auto")
        self.assertTrue(torch.equal(part, video[:1]))
        self.assertFalse(torch.equal(photo, part))
        self.assertEqual(s.frames, 1)
        self.assertEqual(self.node.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(
            self.chunks.RETURN_TYPES[1],
            self.chunks.INPUT_TYPES()["optional"]["previous_state"][0],
        )
        self.assertEqual(self.chunks.FUNCTION, "process_chunk")
        self.assertIn("VideoTAADLAAChunks", node_module.NODE_CLASS_MAPPINGS)

    def test_chunk_snapshot_independent_retry_and_reset(self):
        src = self.clip[:3].clone()
        _, state = self.chunks.process_chunk(src, "Auto")
        stored = (state.source.clone(), state.taa.clone(), state.detail.clone())
        src.fill_(0.37)
        for field, expected in zip((state.source, state.taa, state.detail), stored):
            self.assertTrue(torch.equal(field, expected))
            self.assertEqual(field.device.type, "cpu")
            self.assertEqual(
                field.untyped_storage().nbytes(), field.numel() * field.element_size()
            )
        empty, unchanged = self.chunks.process_chunk(
            src[:0], "Auto", previous_state=state
        )
        self.assertIs(unchanged, state)
        self.assertEqual(len(empty), 0)
        next_input = self.clip[3:4]
        a, _ = self.chunks.process_chunk(next_input, "Auto", previous_state=state)
        b, _ = node_module.VideoTAADLAAChunks().process_chunk(
            next_input, "Auto", previous_state=state
        )
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(state.frames, 3)
        with self.assertRaisesRegex(ValueError, "settings"):
            self.chunks.process_chunk(next_input, "Balanced", previous_state=state)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.chunks.process_chunk(
                next_input[:, :, :-1], "Auto", previous_state=state
            )
        x, cleared = self.chunks.process_chunk(
            next_input, "Auto", 0, previous_state=state
        )
        self.assertIsNone(cleared)
        self.assertTrue(torch.equal(x, next_input))
        self.assertTrue(
            torch.equal(
                self.chunks.process_chunk(next_input, "Auto")[0],
                node_module.VideoTAADLAAChunks().process_chunk(next_input, "Auto")[0],
            )
        )

    def test_preset_dispatch_and_single_motion_measurement(self):
        passes = (
            "_apply_detail_pass",
            "_fine_line_aa",
            "_specular_detail",
            "_apply_texture_pass",
        )
        for preset, expected in (
            ("Performance", (0, 0, 0, 0)),
            ("Balanced", (2, 0, 0, 2)),
            ("High Detail", (2, 2, 2, 2)),
        ):
            with ExitStack() as stack:
                calls = [
                    stack.enter_context(
                        patch.object(self.node, fn, wraps=getattr(self.node, fn))
                    )
                    for fn in passes
                ]
                self.node.execute(self.clip[:2], preset)
                self.assertEqual(tuple(c.call_count for c in calls), expected, preset)
        actual = []
        apply = self.node._apply_dlaa_pipeline

        def record(*args):
            actual.append(args[3].preset)
            return apply(*args)

        with (
            patch.object(self.node, "_apply_dlaa_pipeline", side_effect=record),
            patch.object(
                node_module,
                "measure_camera_motion",
                wraps=node_module.measure_camera_motion,
            ) as measure,
        ):
            self.node.execute(self.clip, "Auto")
        self.assertEqual(measure.call_count, len(self.clip) - 1)
        self.assertEqual(
            actual,
            [
                "Balanced",
                "Detail",
                "Balanced",
                "Performance",
                "Balanced",
                "Detail",
                "Detail",
            ],
        )


class MeasurementTests(unittest.TestCase):
    def test_reference_metrics_on_known_signal(self):
        target = torch.full((3, 16, 18, 3), 0.25)
        same = reference_scores(target, target)
        self.assertEqual(same["rgb_mse"], 0.0)
        self.assertIsNone(same["rgb_psnr_db"])  # infinite for an exact match
        self.assertAlmostEqual(same["luma_ssim"], 1.0, places=9)
        self.assertEqual(same["luma_frame_diff_error_mae_255"], 0.0)
        actual = target.clone()
        actual[1] += 0.125
        scores = reference_scores(actual, target)
        self.assertAlmostEqual(
            scores["rgb_psnr_db"], -10 * math.log10(0.125**2 / 3), places=7
        )
        self.assertAlmostEqual(
            scores["luma_frame_diff_error_mae_255"], 255 * 0.125, places=7
        )
        self.assertLess(scores["luma_ssim"], 1.0)

    def test_area_reference_is_aligned_with_point_sample(self):
        y, x = torch.meshgrid(torch.arange(36), torch.arange(39), indexing="ij")
        ramp = ((x + 2 * y).float() / 128).view(1, 36, 39, 1).expand(2, 36, 39, 3)
        source, target = make_video_pair(ramp)
        self.assertEqual(source.shape, (2, 12, 13, 3))
        torch.testing.assert_close(source, target, atol=0, rtol=0)


class QualityAcceptanceTests(unittest.TestCase):
    def test_balanced_improves_aliased_moving_edges_on_fixture(self):
        source = render_pattern(96, 64, 8, 1)
        reference = render_pattern(96, 64, 8, 8)
        (output,) = node_module.VideoTAADLAA().execute(source, "Balanced")
        before = reference_scores(source, reference)
        after = reference_scores(output, reference)
        improved = {
            "rgb_psnr_db": after["rgb_psnr_db"] > before["rgb_psnr_db"],
            "luma_ssim": after["luma_ssim"] > before["luma_ssim"],
            "luma_frame_diff_error_mae_255": after["luma_frame_diff_error_mae_255"]
            < before["luma_frame_diff_error_mae_255"],
        }
        failures = [
            f"{k}: {before[k]:.5f} -> {after[k]:.5f}"
            for k, passes in improved.items()
            if not passes
        ]
        self.assertFalse(failures, "Bypass -> Balanced 1.0: " + ", ".join(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--node-dir", type=Path, default=NODE_DIR)
    args, rest = parser.parse_known_args()
    NODE_DIR = args.node_dir.resolve()
    logging.basicConfig(level=logging.ERROR)
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)
