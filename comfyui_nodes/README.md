# Qwen2VL-FLUX Custom Nodes for ComfyUI

This directory contains custom ComfyUI nodes for integrating your optimized Qwen2VL-FLUX pipeline with ComfyUI's visual node editor.

## Quick Start

### 1. Installation

```bash
# Navigate to your ComfyUI installation
cd /path/to/ComfyUI

# Copy these custom nodes
cp -r /path/to/qwen2vl-flux/comfyui_nodes custom_nodes/qwen2vl_flux

# Restart ComfyUI server
```

### 2. Verify Installation

After restarting, check the ComfyUI console for:
```
Loading Qwen2VL-FLUX Custom Nodes for ComfyUI
Successfully imported all custom node modules
Registered 4 custom nodes
  - Qwen2VL Image Encoder (Qwen2VLImageEncoder)
  - FLUX Generator (Dual-GPU) (FLUXGeneratorDualGPU)
  - ControlNet Processor (Depth/Line) (ControlNetProcessor)
  - ControlNet Denoiser (ControlNetDenoiser)
```

### 3. Load a Workflow

In ComfyUI's web interface:
1. Click "Load" → select a workflow JSON from `../workflows/`
2. Example: `qwen2vl_flux_variation.json`
3. Click the "Queue Prompt" button

## Available Nodes

### Qwen2VL Image Encoder

Encodes images using the Qwen2VL multimodal model to extract visual embeddings compatible with FLUX.

**Inputs:**
- `image` (IMAGE) - Input image tensor (B, H, W, C) in [0, 1]
- `apply_attention` (BOOLEAN) - Apply attention masking to focus on regions
- `center_x` (FLOAT, 0-1) - X coordinate of attention center
- `center_y` (FLOAT, 0-1) - Y coordinate of attention center
- `radius` (FLOAT, 0-1) - Attention radius

**Outputs:**
- `qwen_embedding` (QWEN_EMBEDDING) - Image embeddings for FLUX input
- `grid_thw` (QWEN_GRID_THW) - Grid dimensions for reference

**Use Cases:**
- Extract visual features from reference images
- Prepare images for variation generation
- Focus attention on specific image regions

### FLUX Generator (Dual-GPU)

Generates images using the optimized FLUX pipeline with dual-GPU memory management.

**Inputs:**
- `qwen_embedding` (QWEN_EMBEDDING) - Embeddings from Qwen2VL encoder
- `qwen_grid_thw` (QWEN_GRID_THW) - Grid information
- `prompt` (STRING) - Text guidance for image generation
- `mode` (CHOICE) - Generation mode:
  - `variation` - Generate variations of the input
  - `img2img` - Transform one image to another
  - `inpaint` - Fill masked regions
  - `controlnet` - Use control signals (depth/line)
  - `controlnet-inpaint` - ControlNet + inpainting
- `guidance_scale` (FLOAT, 1-15) - Classifier-free guidance strength (default: 3.5)
- `num_inference_steps` (INT, 1-100) - Diffusion steps (default: 28)
- `aspect_ratio` (CHOICE) - Output dimensions:
  - `1:1` (1024×1024)
  - `16:9` (1344×768)
  - `9:16` (768×1344)
  - `2.4:1` (1536×640)
  - `3:4` (896×1152)
  - `4:3` (1152×896)
- `batch_size` (INT, 1-8) - Number of images to generate
- `seed` (INT) - Random seed for reproducibility
- `input_image` (IMAGE, optional) - Input image for img2img/inpaint/controlnet
- `input_image_b` (IMAGE, optional) - Secondary input for some modes
- `mask_image` (IMAGE, optional) - Mask for inpainting (white = fill, black = preserve)
- `denoise_strength` (FLOAT, 0-1) - Denoising strength for img2img (default: 0.8)
- `enable_depth_control` (BOOLEAN) - Enable depth guidance (default: True)
- `enable_line_control` (BOOLEAN) - Enable line guidance (default: True)
- `depth_strength` (FLOAT, 0-1) - Depth control strength (default: 0.2)
- `line_strength` (FLOAT, 0-1) - Line control strength (default: 0.4)

**Outputs:**
- `images` (IMAGE) - Generated image batch

**Use Cases:**
- Generate variations of images
- Style transfer (img2img)
- Content filling (inpaint)
- Guided generation with control signals

### ControlNet Processor

Extracts depth maps and edge/line maps from images for ControlNet guidance.

**Inputs:**
- `image` (IMAGE) - Input image
- `control_type` (CHOICE) - Type of control to extract:
  - `depth` - Depth estimation only
  - `line` - Edge detection only
  - `both` - Both depth and line (combined)
- `canny_low_threshold` (INT, 0-255) - Low threshold for Canny edge detection
- `canny_high_threshold` (INT, 0-255) - High threshold for Canny edge detection

**Outputs:**
- `control_image` (IMAGE) - Extracted control signal
- `processed_image` (IMAGE) - Original image for reference

**Use Cases:**
- Extract depth information from photos
- Create edge/line maps for precise control
- Prepare control signals for ControlNet mode

### ControlNet Denoiser

Optional post-processing node to clean and smooth ControlNet output.

**Inputs:**
- `control_image` (IMAGE) - Control image from processor
- `blur_kernel` (INT, 1-21, odd) - Gaussian blur kernel size

**Outputs:**
- `denoised_control` (IMAGE) - Smoothed control image

**Use Cases:**
- Reduce noise in depth maps
- Smooth edge detection results
- Improve ControlNet guidance quality

## Example Workflows

### 1. Image Variation

Generate creative variations of an input image.

**Workflow:** `../workflows/qwen2vl_flux_variation.json`

**Steps:**
1. Load Image → Load your reference image
2. Qwen2VL Encoder → Extract visual embeddings
3. FLUX Generator → Generate variations with optional text prompt
4. Save Image → Save results

**Parameters:**
- Mode: `variation`
- Text prompt: Describe desired style/content
- Batch size: Number of variations
- Guidance scale: 3-4 for balanced results

### 2. Image-to-Image (Style Transfer)

Transform one image based on characteristics of another.

**Workflow:** `../workflows/qwen2vl_flux_img2img.json`

**Steps:**
1. Load reference and target images
2. Encode both images
3. FLUX Generator in `img2img` mode
4. Save results

**Parameters:**
- Mode: `img2img`
- Denoise strength: 0.6-0.8 (lower = more structure preservation)
- Guidance scale: 3.5-5 for style transfer

### 3. Inpainting

Fill masked regions in an image.

**Workflow:** `../workflows/qwen2vl_flux_inpaint.json`

**Steps:**
1. Load image and mask (white = fill, black = preserve)
2. Encode image
3. FLUX Generator in `inpaint` mode with mask
4. Save results

**Parameters:**
- Mode: `inpaint`
- Prompt: Describe what to fill
- Denoise strength: 0.7-0.9

### 4. ControlNet Depth Guidance

Generate images guided by depth maps.

**Workflow:** `../workflows/qwen2vl_flux_controlnet.json`

**Steps:**
1. Load image
2. Encode with Qwen2VL
3. ControlNet Processor → Extract depth and lines
4. ControlNet Denoiser → Smooth results
5. FLUX Generator in `controlnet` mode
6. Save results

**Parameters:**
- Mode: `controlnet`
- Enable depth/line control: Yes
- Depth strength: 0.2-0.4
- Line strength: 0.3-0.6

## Advanced Configuration

### Environment Variables

Create `.env` file in ComfyUI root:

```bash
# Path to qwen2vl-flux project
QWEN2VL_FLUX_PATH=/path/to/qwen2vl-flux

# Checkpoint directory
CHECKPOINT_DIR=/path/to/qwen2vl-flux/checkpoints

# GPU settings
CUDA_VISIBLE_DEVICES=0,1

# Hugging Face token (if models require it)
HF_TOKEN=your_token_here

# Model cache
HF_HOME=/path/to/cache
```

### GPU Memory Optimization

For systems with limited VRAM:

1. **Reduce batch size**: Set `batch_size=1`
2. **Reduce inference steps**: Use 20-24 instead of 28
3. **Enable sequential processing**: Process images one at a time
4. **Use turbo mode** (future enhancement): Faster inference with slightly lower quality

### Logging and Debugging

Enable verbose logging by modifying `__init__.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

Check ComfyUI console for detailed error messages and model loading information.

## Troubleshooting

### Issue: Nodes not appearing in ComfyUI

**Solution:**
1. Verify folder structure: `ComfyUI/custom_nodes/qwen2vl_flux/`
2. Check console for import errors
3. Restart ComfyUI completely
4. Clear browser cache and refresh page

### Issue: "CUDA out of memory"

**Solution:**
- Reduce `batch_size` to 1
- Reduce `num_inference_steps` to 20
- Ensure no other GPU processes running
- Check GPU memory: `nvidia-smi`

### Issue: Model not found

**Solution:**
- Verify `CHECKPOINT_DIR` environment variable
- Check model files exist in checkpoints directory
- Set `QWEN2VL_FLUX_PATH` to your project directory

### Issue: Slow generation

**Causes and solutions:**
- **Insufficient VRAM**: Reduce batch size or disable controls
- **CPU bottleneck**: Check system CPU usage
- **Disk I/O**: Ensure models on fast storage (NVMe)
- **High inference steps**: Start with 20 steps

### Issue: Inconsistent outputs

**Solution:**
- Always set a `seed` value for reproducibility
- Use same parameters for comparison
- Check GPU isn't being used by other processes

## Performance Benchmarks

Typical generation times (on RTX A6000, per image):

| Mode | Steps | Time | Memory |
|------|-------|------|--------|
| Variation | 28 | 45-60s | 48GB |
| Variation | 20 | 32-40s | 48GB |
| Image-to-Image | 28 | 50-65s | 50GB |
| Inpaint | 28 | 48-62s | 50GB |
| ControlNet | 28 | 55-70s | 60GB |

**Optimization tips:**
- Use 20-24 steps for faster iterations
- Process images sequentially for lower VRAM usage
- Cache models in RAM if possible

## API Integration

Use ComfyUI's API to queue prompts programmatically:

```python
import requests
import json

SERVER_URL = "http://localhost:8188"

workflow = {
    "1": {"inputs": {"image": "path/to/image.png"}, "class_type": "LoadImage"},
    "2": {"inputs": {"image": ["1", 0], "apply_attention": False}, "class_type": "Qwen2VLImageEncoder"},
    "3": {
        "inputs": {
            "qwen_embedding": ["2", 0],
            "qwen_grid_thw": ["2", 1],
            "prompt": "A beautiful landscape",
            "mode": "variation",
        },
        "class_type": "FLUXGeneratorDualGPU",
    },
    "4": {"inputs": {"images": ["3", 0]}, "class_type": "SaveImage"}
}

response = requests.post(f"{SERVER_URL}/prompt", json={"prompt": workflow})
print(response.json())
```

## Development

### Adding Custom Functionality

To extend these nodes:

1. Modify the relevant `.py` file
2. Update `INPUT_TYPES()` for new parameters
3. Update `FUNCTION` logic
4. Restart ComfyUI to reload

### File Structure

```
comfyui_nodes/
├── __init__.py                      # Node registration
├── qwen2vl_image_encoder.py        # Qwen2VL encoder node
├── flux_generator_dual_gpu.py      # FLUX generator node
├── controlnet_processor.py          # ControlNet processor nodes
└── README.md                        # This file
```

## Support and Resources

- **Issues**: Check ComfyUI console for error messages
- **Documentation**: See `COMFYUI_INTEGRATION.md` for detailed setup
- **Workflows**: Example workflows in `../workflows/`
- **Configuration**: See `DEPLOYMENT_GUIDE.md` for environment setup

## License

These custom nodes are part of the qwen2vl-flux project. See LICENSE file in project root.
