# ComfyUI Video Anti-Aliasing & MiniMax H3 Upscaler
<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI--Manager-Verified-green?style=flat-square&logo=github" alt="ComfyUI Manager Verified">
  <img src="https://flat.badgen.net/github/release/SOLRICKS/comfyui-solricks?icon=github&label=release&color=orange&cache=3600" alt="Release">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python" alt="Python 3.10+">
  <a href="https://github.com/SOLRICKS/comfyui-solricks/stargazers">
    <img src="https://img.shields.io/github/stars/SOLRICKS/comfyui-solricks?style=flat-square&logo=github&color=24292f&label=stars" alt="GitHub stars">
  </a>
</p>

> Available in **ComfyUI Manager**. Search for **SOLRICKS**.

Anti-aliasing, detail refinement, cinematic effects, and MiniMax H3 upscaling for ComfyUI image and video workflows. VideoTAADLAA combines temporal anti-aliasing, jittered sampling, and DLAA-inspired refinement for cleaner, more stable edges. VideoDetailRefiner adds a lightweight detail and texture refinement pass for image and video outputs. VideoAdaptiveAA provides a lightweight edge-focused cleanup pass for aliasing-prone regions.

---

> [!IMPORTANT]
> **Disclaimer**
> The anti-aliasing nodes do not use NVIDIA's closed-source SDKs or native DLAA/DLSS binaries. Instead, they are custom adaptations of Temporal and Spatial Anti-Aliasing techniques commonly found in modern game engines, rebuilt entirely from scratch using PyTorch tensor architecture specifically for ComfyUI video post-processing.

---

## MiniMax H3 Upscaler

Upscale an existing H3 video with a learned latent upscaler, then refine detail with an H3 LoRA. The node uses ComfyUI's built-in model implementations and returns **images** and **audio**.

### Comparison

<table>
  <thead>
    <tr><th>Before — 1376×768</th><th>After — MiniMax H3 Upscaler · 1952×1088</th></tr>
  </thead>
  <tbody>
    <tr>
      <td width="50%"><video controls playsinline preload="metadata" width="100%" src="https://github.com/user-attachments/assets/5177d194-77af-4bbd-99d4-5f116a9997e7"></video></td>
      <td width="50%"><video controls playsinline preload="metadata" width="100%" src="https://github.com/user-attachments/assets/2ddd1845-8f32-461d-9363-934b31840b66"></video></td>
    </tr>
    <tr>
      <td align="center"><a href="https://huggingface.co/SOLRICKS/MiniMax-H3-Upscaler/resolve/main/Minimax_H3_I2V.mp4">Open original video</a></td>
      <td align="center"><a href="https://huggingface.co/SOLRICKS/MiniMax-H3-Upscaler/resolve/main/Minimax-H3-Upscaler.mp4">Open upscaled video</a></td>
    </tr>
  </tbody>
</table>

*AI-generated source and upscaled output. Both clips: 243 frames, 24 FPS, 10.125 seconds. Web previews are compressed for GitHub playback. The links above open the full-quality MP4 files on Hugging Face.*

### Models on Hugging Face

Download the models and example workflows from the [Visual AI Models & Tools](https://huggingface.co/collections/SOLRICKS/visual-ai-models-and-tools) collection.

| Model | Download | Place in |
|---|---|---|
| Experimental INT8 refiner LoRA | [h3_upscale_refiner_int8.safetensors](https://huggingface.co/SOLRICKS/MiniMax-H3-Upscaler/resolve/main/h3_upscale_refiner_int8.safetensors?download=true) | `ComfyUI/models/loras/` |
| FP16 latent upscaler | [minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors](https://huggingface.co/SOLRICKS/MiniMax-H3-Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors?download=true) | `ComfyUI/models/latent_upscale_models/` |

You also need a compatible **MiniMax H3 FL2VA diffusion model** and **H3 video VAE**, such as the files in [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental). Put these in `models/diffusion_models/` and `models/vae/` respectively. Connect a model without generation/Turbo/style LoRAs: this node applies the refiner LoRA itself.

### Workflow downloads

- [MiniMax H3 Upscaler — ComfyUI workflow](workflows/MiniMax_H3_Upscaler.json)
- [MiniMax H3 Upscaler — API prompt](workflows/MiniMax_H3_Upscaler_API.json)
- [Hugging Face models and workflows](https://huggingface.co/SOLRICKS/MiniMax-H3-Upscaler)

The GUI workflow uses core **Load Video / Get Video Components** and **VideoHelperSuite Video Combine**. Install [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) for the example's MP4 export. Select your input video and the installed model filenames after loading the workflow.

### Connections and controls

Required connections: **model**, **vae**, **images**. Optional: **conditioning** (H3 text prompt), **audio**, **fps** and **audio_vae**. Connect source FPS when it is not 24. Audio is passed through unchanged; an audio VAE is only needed for soundtrack conditioning during refinement.

| Control | Purpose |
|---|---|
| `upscale_model` | Select the installed H3 latent upscaler. |
| `megapixels` | Set target area; default 2.0. H3 dimensions are aligned to 32 pixels. Choose a target at least as large as the input. |
| `lora` | Select the refiner LoRA, including the INT8 repack or upstream BF16 file. |
| `enable temporal chunking` | Allow overlapping time windows to manage longer clips and VRAM use. |
| `enable spatial tiling` | Allow overlapping spatial tiles for larger frames. |

Refinement defaults are fixed in the code: two Euler steps, seed 42, LoRA strength 0.8 and redraw start 0.6. Sparse attention is preferred, with dense fallback for unsupported sparse backends. Recognized OOMs retry smaller windows.

The node provides a green progress bar and cancellation checks between processing batches, model calls and sampler steps. An active GPU/VAE call must return before cancellation can be handled. Routine console messages are deduplicated; warnings and errors remain visible.

Temporal chunking and spatial tiling limit per-pass model work. The full source and output IMAGE batches still occupy system RAM. The INT8 refiner has measured per-matrix relative error of 8.24% median and 28.41% maximum; these are numerical errors rather than visual-quality percentages. INT8 storage is approximately half the BF16 file size and is lossy; identical visual quality, faster inference and lower runtime VRAM are not guaranteed.

---

### Anti-Aliasing Models
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
  <strong>TAA + DLAA Comparison — Before / After</strong>
</p>

https://github.com/user-attachments/assets/9772e56f-6c9f-4dc0-a8c9-03701fb25102

<br>

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

## How to use
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
2. The anti-aliasing/refinement model files are included. MiniMax H3 model files are downloaded separately using the links above.
3. MiniMax H3 Upscaler requires a current ComfyUI with H3 latent upscaling support. For the new node, use the current GitHub/main version of this repository and restart ComfyUI.
4. No additional upscaler node pack is required. The example video workflow uses VideoHelperSuite for MP4 export.

---

## License
SOLRICKS node code is licensed under the MIT License; see [LICENSE](LICENSE). Separately downloaded model weights keep their own licenses. The H3 refiner follows the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/6818f6c32d12b210915e44ad56a4228c2608f160/LICENSE); the [LBH-123-AI latent upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) is Apache-2.0. The SOLRICKS INT8 file is a conversion of an upstream refiner, without additional training.
