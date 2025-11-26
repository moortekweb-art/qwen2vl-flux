# Quick Start Guide: Qwen2VL-FLUX with ComfyUI

Get your Qwen2VL-FLUX pipeline running with ComfyUI in minutes.

## 5-Minute Setup

### Prerequisites
- Two NVIDIA GPUs with CUDA 12.4 support
- ComfyUI installed
- Qwen2VL-FLUX repository cloned
- Model checkpoints downloaded

### Step 1: Copy Custom Nodes (1 minute)

```bash
# From your ComfyUI installation directory
cp -r /path/to/qwen2vl-flux/comfyui_nodes custom_nodes/qwen2vl_flux
```

### Step 2: Set Environment Variables (1 minute)

Create `.env` in your ComfyUI root:
```bash
export QWEN2VL_FLUX_PATH=/path/to/qwen2vl-flux
export CHECKPOINT_DIR=/path/to/qwen2vl-flux/checkpoints
export NVIDIA_VISIBLE_DEVICES=0,1
```

### Step 3: Start ComfyUI (1 minute)

```bash
cd /path/to/ComfyUI
python main.py --listen 0.0.0.0 --port 8188
```

Open browser: `http://localhost:8188`

### Step 4: Load a Workflow (1 minute)

In ComfyUI:
1. Click **Load** button
2. Select workflow: `/path/to/qwen2vl-flux/workflows/qwen2vl_flux_variation.json`
3. Click **Queue Prompt**

### Step 5: Wait for Results (1-2 minutes)

Watch the console for generation progress. Output saved to `ComfyUI/output/`

---

## What Just Happened?

You've just created an image variation using:
1. **Qwen2VL Image Encoder** - Extracted visual embeddings from your image
2. **FLUX Generator** - Generated new variations using dual-GPU optimization
3. **Saved** - Stored results automatically

---

## Try Other Modes

### Image-to-Image (Style Transfer)

```bash
# In ComfyUI, load:
workflows/qwen2vl_flux_img2img.json
```

**What it does**: Transform one image based on another's style while preserving structure.

### Inpainting (Fill Missing Regions)

```bash
# Load:
workflows/qwen2vl_flux_inpaint.json
```

**What it does**: Fill masked areas (white) with generated content while preserving masked areas (black).

### ControlNet (Guided Generation)

```bash
# Load:
workflows/qwen2vl_flux_controlnet.json
```

**What it does**: Generate images guided by depth maps and edge detection for precise control.

---

## Common Tasks

### Generate Multiple Variations

In FLUX Generator node:
- Set `batch_size` to 4
- Adjust `seed` for variety
- Click Queue

### Change Generation Style

In FLUX Generator node:
- Edit `prompt` field
- Examples:
  - "oil painting style"
  - "cyberpunk aesthetic"
  - "ultra realistic photography"

### Control Generation Strength

- **img2img**: Reduce `denoise_strength` (0.4 = more structure)
- **ControlNet**: Adjust `depth_strength` and `line_strength`
- **guidance_scale**: Higher = follow prompt more strictly (3-5 recommended)

### Reproduce Results

In FLUX Generator node:
- Set `seed` to specific number
- Use identical parameters
- Same results every time

---

## GPU Memory Tips

If you get "CUDA out of memory":

1. **Reduce batch size** to 1
2. **Reduce inference steps** to 20
3. **Disable controls** (if using ControlNet)
4. **Check other processes**: `nvidia-smi`

```bash
# Monitor GPU usage in real-time
nvidia-smi -l 1
```

---

## Troubleshooting

### Nodes Not Appearing

```bash
# 1. Verify folder structure
ls -la /path/to/ComfyUI/custom_nodes/qwen2vl_flux/

# 2. Check console for errors
# (Watch ComfyUI terminal output)

# 3. Restart ComfyUI
# Ctrl+C in terminal, then restart
```

### Model Not Found

```bash
# Verify checkpoints directory
ls -la /path/to/checkpoints/

# Set correct environment variable
export CHECKPOINT_DIR=/path/to/checkpoints
export QWEN2VL_FLUX_PATH=/path/to/qwen2vl-flux
```

### GPU Not Detected

```bash
# Verify nvidia-docker is working
docker run --rm --runtime=nvidia nvidia/cuda:12.4.1-base nvidia-smi

# Check NVIDIA_VISIBLE_DEVICES
echo $NVIDIA_VISIBLE_DEVICES
```

### Slow Generation

**Expected timings** (RTX A6000, per image):
- 28 steps: 45-60 seconds
- 20 steps: 32-40 seconds
- Turbo mode: 15-20 seconds

To speed up:
- Reduce `num_inference_steps` to 20
- Reduce `batch_size` to 1
- Check GPU utilization: `nvidia-smi`

---

## Docker Quick Start (Alternative)

```bash
# Build production image
docker build -t qwen2vl-flux:prod .

# Run with docker-compose
docker-compose -f docker-compose.prod.yml up -d

# ComfyUI available at: http://localhost:8188
```

---

## Next Steps

### Learn More
- **[COMFYUI_INTEGRATION.md](COMFYUI_INTEGRATION.md)** - Detailed setup guide
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Production deployment
- **[comfyui_nodes/README.md](comfyui_nodes/README.md)** - Node documentation

### Customize
1. Create your own workflows
2. Modify node parameters
3. Combine with other ComfyUI nodes
4. Export workflows as JSON

### Scale
- Deploy to Kubernetes
- Set up API endpoint
- Integrate with frontends (OpenWebUI, Gradio)
- Batch process images

---

## Key Concepts

### Dual-GPU Architecture
- **GPU 0** (cuda:0): Qwen2VL encoder, text encoders (≈20GB)
- **GPU 1** (cuda:1): FLUX transformer, VAE (≈40GB)
- Prevents OOM on single GPU

### Generation Modes

| Mode | Use Case | Speed |
|------|----------|-------|
| **variation** | Creative variations | Fast |
| **img2img** | Style transfer | Medium |
| **inpaint** | Fill masked regions | Medium |
| **controlnet** | Depth/line guidance | Slow |

### Key Parameters

| Parameter | Effect | Recommended |
|-----------|--------|-------------|
| **guidance_scale** | Follow prompt strength | 3-4 |
| **num_inference_steps** | Quality vs speed | 20-28 |
| **denoise_strength** | img2img structure | 0.6-0.8 |
| **seed** | Reproducibility | Fixed for consistency |

---

## Example Prompts

### Natural Photography
"Professional landscape photography, golden hour, ultra high quality, 8k resolution"

### Digital Art
"digital illustration, character art, anime style, vibrant colors, detailed"

### 3D Rendering
"3d model, isometric view, unreal engine 5, octane render, studio lighting"

### Architecture
"modern architecture, minimalist design, concrete and glass, detailed"

### Abstract
"abstract expressionism, oil painting, geometric shapes, modern art"

---

## Performance Checklist

- [ ] Both GPUs detected: `nvidia-smi`
- [ ] Environment variables set: `echo $QWEN2VL_FLUX_PATH`
- [ ] Checkpoints accessible: `ls $CHECKPOINT_DIR`
- [ ] Custom nodes visible in ComfyUI
- [ ] First generation completes successfully
- [ ] Output images saved to `ComfyUI/output/`

---

## Common Workflows

### Workflow 1: Batch Variation
```
Load Image → Qwen2VL Encode → FLUX Generate (batch=4) → Save
```

### Workflow 2: Style Transfer
```
Load Ref + Target → Encode Both → FLUX img2img → Save
```

### Workflow 3: Controlled Generation
```
Load Image → Encode → Extract Controls → FLUX ControlNet → Save
```

---

## Hardware Requirements

**Minimum:**
- 2x GPUs with 24GB+ VRAM each
- 128GB system RAM
- NVIDIA Driver 555+

**Recommended:**
- 2x RTX A6000 / L40S / RTX 6000
- 256GB system RAM
- NVMe storage for models

**Tested:**
- RTX A6000 (48GB) - Excellent
- RTX L40S (48GB) - Excellent
- RTX 6000 (48GB) - Excellent

---

## Support Resources

- Check `comfyui_nodes/README.md` for node documentation
- See `COMFYUI_INTEGRATION.md` for detailed setup
- Review `DEPLOYMENT_GUIDE.md` for production setup
- Check ComfyUI console for error messages

---

## What's Next?

1. ✅ Generate your first images
2. Experiment with different prompts
3. Try different generation modes
4. Create custom workflows
5. Deploy to production (see DEPLOYMENT_GUIDE.md)

---

## Success! 🎉

You now have a fully functional Qwen2VL-FLUX pipeline with ComfyUI!

Need help? Check the troubleshooting section above or review the comprehensive guides included in the project.
