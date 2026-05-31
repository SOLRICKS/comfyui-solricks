# ComfyUI Video Anti-Aliasing Pack
<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI--Manager-Verified-green?style=flat-square&logo=github" alt="Manager">
  <img src="https://img.shields.io/github/v/release/SOLRICKS/comfyui-solricks?style=flat-square&color=orange" alt="Release">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/github/stars/SOLRICKS/comfyui-solricks?style=flat-square&color=gold" alt="Stars">
</p>

> Available in **ComfyUI Manager**. Search for **SOLRICKS**.

Anti-aliasing and detail refinement nodes for ComfyUI image and video workflows. VideoTAADLAA combines temporal anti-aliasing, jittered sampling, and DLAA-inspired refinement for cleaner, more stable edges. VideoDetailRefiner adds a lightweight detail and texture refinement pass for image and video outputs. VideoAdaptiveAA provides a lightweight edge-focused cleanup pass for aliasing-prone regions.

---

> [!IMPORTANT]
> **Disclaimer**
> This node pack does not use NVIDIA's closed-source SDKs or native DLAA/DLSS binaries. Instead, it is a custom adaptation of Temporal and Spatial Anti-Aliasing techniques commonly found in modern game engines, rebuilt entirely from scratch using PyTorch tensor architecture specifically for ComfyUI video post-processing.

---

## Preview
| TAA + DLAA Anti-Aliasing | Adaptive Anti-Aliasing |
| :---: | :---: |
| <img src="https://github.com/SOLRICKS/comfyui-solricks/blob/main/assets/video_taadlaa.png" width="100%"> | <img src="https://github.com/SOLRICKS/comfyui-solricks/blob/main/assets/video_adaptive_aa.png" width="100%"> |

---

### Models
The pack includes two 1x refinement models:

- **DLAANet.safetensors** — main anti-aliasing refinement model for cleaner edges.
- **DLAATexture.safetensors** — optional texture refinement model for fine detail and micro-texture.

Both models keep the original image resolution and are not ESRGAN models.

> **Note:** DLAATexture.safetensors is a 1x detail refinement model and does not upscale image resolution.

---

## Features
- **Temporal stability:** Reduces flicker, shimmer, and sub-pixel jitter across frames.
- **DLAA-inspired refinement:** Improves edge quality while keeping the original resolution.
- **Motion-aware cleanup:** Helps reduce ghosting and trailing artifacts in moving scenes.
- **Edge-preserving detail:** Uses edge detection to clean aliasing without over-blurring fine texture.
- **Lightweight inference:** Designed for ComfyUI post-processing with modest VRAM usage.
- **Detail refinement:** Enhances fine texture, hair, fabric, small edges, and micro-detail.

---

## Comparison
<p align="center">
  <strong>AdaptiveAA Comparison — Before / After</strong>
</p>
<table align="center">
  <tr>
	<td align="center"><strong></strong></td>
	<td align="center"><strong></strong></td>
  </tr>
  <tr>
	<td width="33%"><img src="assets/compare_1.png" width="100%"></td>
	<td width="33%"><img src="assets/compare_2.png" width="100%"></td>
  </tr>
</table>

<br>

<p align="center">
  <strong>TAA + DLAA Comparison — Before / After</strong>
</p>
<table align="center">
  <tr>
	<td align="center"><strong></strong></td>
	<td align="center"><strong></strong></td>
	<td align="center"><strong></strong></td>
  </tr>
  <tr>
    <td width="33%"><video src="https://github.com/user-attachments/assets/4109726e-c5db-404b-9c0d-23d49b9641cf" width="100%" controls autoplay muted loop>
  </video></td>
    <td width="33%"><video src="https://github.com/user-attachments/assets/05f8cbb1-9388-44e5-9274-0ee80d6aa37b" width="100%" controls autoplay muted loop>
  </video></td>
    <td width="33%"><video src="https://github.com/user-attachments/assets/ddf7c74b-289d-48a7-ab43-47558aa9deab" width="100%" controls autoplay muted loop>
  </video></td>
  </tr>
</table>

---

## 🚀 How to use
Recommended starting point:

- Use **Auto** for general footage.
- Use **Performance** for longer videos or faster previews.
- Use **Balanced** for stable video cleanup with a good quality/speed balance.
- Use **High Detail** for hair, fur, wires, fine lines, and high-detail edges.

The node uses a simple game-style preset system:

| Preset | Description |
|---|---|
| **Auto** | Automatically adjusts settings based on scene motion. Good default for most videos. |
| **Performance** | Faster processing for longer videos. Uses lighter detail cleanup while keeping the image stable. |
| **Balanced** | General-purpose preset with a balance between quality and speed. |
| **High Detail** | Best quality for fine detail, edge cleanup, and temporal refinement. Slower. |

Start with **Auto**, then adjust based on your content.

For LTX or Wan workflows, run this node separately as the final post-process.

---

### Video Detail Refiner

Use **Video Detail Refiner** as an optional final pass after image or video generation, or after the main **Video Anti-Aliasing (TAA + DLAA)** node.

Recommended starting point:

- Use **Photo** for single images.
- Use **Video Balanced** for most videos.
- Use **Video High Detail** for hair, fur, fabric, wires, and fine texture.
- Use **Soft** for safer, smoother refinement.
- Use **Performance** for faster previews or lower VRAM usage.

| Preset | Description |
|---|---|
| **Photo** | Detail refinement tuned for single images. |
| **Video Balanced** | Good default for most videos with stable detail enhancement. |
| **Video High Detail** | Stronger detail and texture refinement for fine edges, hair, fur, fabric, and micro-texture. |
| **Soft** | Safer refinement with smoother output and stronger protection. |
| **Performance** | Faster processing with model pass disabled. Useful for previews and lower VRAM setups. |

---

## Installation
1. Install via **ComfyUI Manager** by searching for **SOLRICKS** or **Anti-Aliasing Pack**, or clone this repo into your `custom_nodes` folder.
2. The required model files are included with this custom node.
3. No additional Python dependencies are required.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
