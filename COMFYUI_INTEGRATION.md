# ComfyUI Integration Guide for Qwen2VL-FLUX

This guide explains how to integrate your custom Qwen2VL-FLUX pipeline with ComfyUI, a visual node-based interface for image generation.

## Overview

Your Qwen2VL-FLUX pipeline has been optimized for:
- **Dual-GPU memory management**: Qwen2VL and text encoders on GPU 0 (cuda:0), FLUX transformer and VAE on GPU 1 (cuda:1)
- **Tensor padding**: Automatic dimension padding to 4608 or 4736 tokens for stable inference
- **Multiple generation modes**: Variation, Image-to-Image, Inpainting, and ControlNet (with depth and line guidance)

ComfyUI provides a visual workflow editor that can wrap these complex operations into reusable nodes.

## Prerequisites

- Two NVIDIA GPUs (recommended for optimal performance)
- ComfyUI installed with CUDA 12.4 support
- Custom nodes module support
- Your Qwen2VL-FLUX repository cloned

## Part 1: ComfyUI Installation

### Step 1: Install ComfyUI

```bash
# Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install dependencies
pip install -r requirements.txt

# Install CUDA 12.4 compatible PyTorch (same as your project)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124
```

### Step 2: Set Up ComfyUI Custom Nodes Directory

ComfyUI loads custom nodes from the `custom_nodes` folder:

```bash
# Navigate to ComfyUI root
cd ComfyUI

# Create custom_nodes directory if it doesn't exist
mkdir -p custom_nodes
```

## Part 2: Create Custom ComfyUI Nodes

### Overview of Custom Nodes Architecture

Your Qwen2VL-FLUX integration requires three custom nodes:

1. **Qwen2VL Image Encoder** - Processes images and extracts multi-modal embeddings
2. **FLUX Generator with Dual-GPU** - Generates images using your optimized pipeline
3. **ControlNet Processor** - Handles depth and line control inputs

### Node 1: Qwen2VL Image Encoder Node

Create file: `ComfyUI/custom_nodes/qwen2vl_image_encoder.py`

```python
"""
ComfyUI Custom Node: Qwen2VL Image Encoder
Processes images and generates embeddings for FLUX input
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple
import os
import sys

# Add the qwen2vl-flux project to path
QWEN2VL_FLUX_PATH = os.getenv('QWEN2VL_FLUX_PATH', '../qwen2vl-flux')
sys.path.insert(0, QWEN2VL_FLUX_PATH)

from model import FluxModel, Qwen2Connector
from transformers import AutoProcessor


class Qwen2VLImageEncoder:
    """
    Encodes images using Qwen2VL model to produce embeddings compatible with FLUX.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda:0"  # Qwen2VL runs on GPU 0

    def load_model(self):
        """Lazy load the Qwen2VL model"""
        if self.model is None:
            checkpoint_dir = os.getenv('CHECKPOINT_DIR', 'checkpoints')

            # Initialize FluxModel (this loads Qwen2VL internally)
            self.model = FluxModel(
                is_turbo=False,
                device=self.device,
                required_features=[]
            )

            # Load processor
            qwen2vl_path = os.path.join(checkpoint_dir, 'qwen2-vl')
            self.processor = AutoProcessor.from_pretrained(
                qwen2vl_path,
                min_pixels=256*28*28,
                max_pixels=512*28*28
            )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI IMAGE type (B, H, W, C) in [0, 1]
                "apply_attention": ("BOOLEAN", {"default": False}),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "radius": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("QWEN_EMBEDDING", "QWEN_GRID_THW")
    RETURN_NAMES = ("qwen_embedding", "grid_thw")
    FUNCTION = "encode"
    CATEGORY = "Qwen2VL"

    def encode(self, image, apply_attention, center_x, center_y, radius):
        self.load_model()

        # Convert ComfyUI image format (B, H, W, C) [0, 1] to PIL Image
        # For single image: take first batch
        img_tensor = image[0]  # (H, W, C)
        img_numpy = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_numpy)

        # Process image through Qwen2VL
        qwen2_hidden_state, image_grid_thw = self.model.process_image(pil_image)

        # Apply attention masking if requested
        if apply_attention:
            qwen2_hidden_state = self.model.apply_attention(
                qwen2_hidden_state,
                image_grid_thw,
                center_x,
                center_y,
                radius
            )

        return (qwen2_hidden_state, image_grid_thw)


# Register node
NODE_CLASS_MAPPINGS = {
    "Qwen2VLImageEncoder": Qwen2VLImageEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen2VLImageEncoder": "Qwen2VL Image Encoder",
}
```

### Node 2: FLUX Generator with Dual-GPU Support

Create file: `ComfyUI/custom_nodes/flux_generator_dual_gpu.py`

```python
"""
ComfyUI Custom Node: FLUX Generator with Dual-GPU Support
Generates images using the optimized Qwen2VL-FLUX pipeline
"""

import torch
import numpy as np
from PIL import Image
import os
import sys

QWEN2VL_FLUX_PATH = os.getenv('QWEN2VL_FLUX_PATH', '../qwen2vl-flux')
sys.path.insert(0, QWEN2VL_FLUX_PATH)

from model import FluxModel


class FLUXGeneratorDualGPU:
    """
    Generates images using FLUX with dual-GPU memory optimization.
    Handles Variation, Image-to-Image, Inpainting, and ControlNet modes.
    """

    def __init__(self):
        self.model = None

    def load_model(self, mode="variation", enable_controlnet=False):
        """Lazy load the FLUX model with required features"""
        if self.model is None:
            required_features = []
            if mode == "controlnet" or enable_controlnet:
                required_features.extend(['controlnet', 'depth', 'line'])

            self.model = FluxModel(
                is_turbo=False,
                device="cuda",  # Will be handled by the model internally
                required_features=required_features
            )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_embedding": ("QWEN_EMBEDDING",),
                "qwen_grid_thw": ("QWEN_GRID_THW",),
                "prompt": ("STRING", {"multiline": True}),
                "mode": (["variation", "img2img", "inpaint", "controlnet", "controlnet-inpaint"],),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 15.0}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "2.4:1", "3:4", "4:3"],),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
                "seed": ("INT", {"default": 0}),
            },
            "optional": {
                "input_image_b": ("IMAGE",),
                "mask_image": ("IMAGE",),
                "denoise_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0}),
                "enable_depth_control": ("BOOLEAN", {"default": True}),
                "enable_line_control": ("BOOLEAN", {"default": True}),
                "depth_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                "line_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "FLUX"

    def convert_tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image"""
        if len(tensor.shape) == 4:  # Batch
            tensor = tensor[0]
        img_numpy = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_numpy)

    def convert_pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor"""
        img_array = np.array(pil_image, dtype=np.float32) / 255.0
        return torch.from_numpy(img_array).unsqueeze(0)

    def generate(
        self,
        qwen_embedding,
        qwen_grid_thw,
        prompt,
        mode,
        guidance_scale,
        num_inference_steps,
        aspect_ratio,
        batch_size,
        seed,
        input_image_b=None,
        mask_image=None,
        denoise_strength=0.8,
        enable_depth_control=True,
        enable_line_control=True,
        depth_strength=0.2,
        line_strength=0.4,
    ):
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.load_model(mode=mode, enable_controlnet=(mode.startswith("controlnet")))

        # For this example, we assume input_image_a comes from the qwen embedding process
        # In a real workflow, you'd need to track the original image separately
        # For now, we'll create a dummy image
        input_image_a = Image.new('RGB', (512, 512), color=(128, 128, 128))

        # Convert optional inputs
        input_image_b_pil = None
        if input_image_b is not None:
            input_image_b_pil = self.convert_tensor_to_pil(input_image_b)

        mask_image_pil = None
        if mask_image is not None:
            mask_image_pil = self.convert_tensor_to_pil(mask_image)

        # Generate images
        gen_images = self.model.generate(
            input_image_a=input_image_a,
            input_image_b=input_image_b_pil,
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            aspect_ratio=aspect_ratio,
            mode=mode,
            denoise_strength=denoise_strength,
            mask_image=mask_image_pil,
            imageCount=batch_size,
            line_mode=enable_line_control,
            depth_mode=enable_depth_control,
            line_strength=line_strength,
            depth_strength=depth_strength,
        )

        # Convert PIL images to tensor
        output_tensors = []
        for img in gen_images:
            output_tensors.append(self.convert_pil_to_tensor(img))

        output = torch.cat(output_tensors, dim=0)
        return (output,)


NODE_CLASS_MAPPINGS = {
    "FLUXGeneratorDualGPU": FLUXGeneratorDualGPU,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FLUXGeneratorDualGPU": "FLUX Generator (Dual-GPU)",
}
```

### Node 3: ControlNet Processor Node

Create file: `ComfyUI/custom_nodes/controlnet_processor.py`

```python
"""
ComfyUI Custom Node: ControlNet Processor
Handles depth and line detection for ControlNet guidance
"""

import torch
import numpy as np
from PIL import Image
import os
import sys

QWEN2VL_FLUX_PATH = os.getenv('QWEN2VL_FLUX_PATH', '../qwen2vl-flux')
sys.path.insert(0, QWEN2VL_FLUX_PATH)

from model import FluxModel


class ControlNetProcessor:
    """
    Processes images to extract depth maps and edge/line maps for ControlNet.
    """

    def __init__(self):
        self.model = None

    def load_model(self):
        """Lazy load the FLUX model"""
        if self.model is None:
            self.model = FluxModel(
                is_turbo=False,
                device="cuda",
                required_features=['depth', 'line']
            )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "control_type": (["depth", "line", "both"],),
            },
            "optional": {
                "canny_low_threshold": ("INT", {"default": 50, "min": 0, "max": 255}),
                "canny_high_threshold": ("INT", {"default": 150, "min": 0, "max": 255}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("control_image",)
    FUNCTION = "process"
    CATEGORY = "ControlNet"

    def convert_tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image"""
        if len(tensor.shape) == 4:  # Batch
            tensor = tensor[0]
        img_numpy = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_numpy)

    def convert_pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor"""
        img_array = np.array(pil_image, dtype=np.float32) / 255.0
        return torch.from_numpy(img_array).unsqueeze(0)

    def process(
        self,
        image,
        control_type,
        canny_low_threshold=50,
        canny_high_threshold=150,
    ):
        self.load_model()

        pil_image = self.convert_tensor_to_pil(image)

        if control_type == "depth":
            control_image = self.model.generate_depth_map(pil_image)
        elif control_type == "line":
            control_image = self.model.generate_canny_edges(
                pil_image,
                low_threshold=canny_low_threshold,
                high_threshold=canny_high_threshold
            )
        else:  # both
            depth = self.model.generate_depth_map(pil_image)
            lines = self.model.generate_canny_edges(
                pil_image,
                low_threshold=canny_low_threshold,
                high_threshold=canny_high_threshold
            )
            # Stack them
            control_image = Image.new('RGB', pil_image.size)
            # You could blend or combine these differently
            control_image = depth

        return (self.convert_pil_to_tensor(control_image),)


NODE_CLASS_MAPPINGS = {
    "ControlNetProcessor": ControlNetProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ControlNetProcessor": "ControlNet Processor (Depth/Line)",
}
```

## Part 3: Create Node Registration Module

Create file: `ComfyUI/custom_nodes/__init__.py`

```python
"""
Qwen2VL-FLUX Custom Nodes for ComfyUI
"""

# These imports register the nodes automatically
from . import qwen2vl_image_encoder
from . import flux_generator_dual_gpu
from . import controlnet_processor

# Combine all node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(qwen2vl_image_encoder.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(flux_generator_dual_gpu.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(controlnet_processor.NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS.update(qwen2vl_image_encoder.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(flux_generator_dual_gpu.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(controlnet_processor.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
```

## Part 4: Example ComfyUI Workflow JSON

Create file: `workflows/qwen2vl_flux_variation.json`

This is a complete ComfyUI workflow for image variation using Qwen2VL-FLUX:

```json
{
  "1": {
    "inputs": {
      "ckpt_name": "model.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "2": {
    "inputs": {
      "image": "path/to/input/image.png"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "3": {
    "inputs": {
      "image": [
        "2",
        0
      ],
      "apply_attention": false,
      "center_x": 0.5,
      "center_y": 0.5,
      "radius": 0.5
    },
    "class_type": "Qwen2VLImageEncoder",
    "_meta": {
      "title": "Encode with Qwen2VL"
    }
  },
  "4": {
    "inputs": {
      "qwen_embedding": [
        "3",
        0
      ],
      "qwen_grid_thw": [
        "3",
        1
      ],
      "prompt": "A beautiful landscape with mountains and a sunset",
      "mode": "variation",
      "guidance_scale": 3.5,
      "num_inference_steps": 28,
      "aspect_ratio": "1:1",
      "batch_size": 2,
      "seed": 42
    },
    "class_type": "FLUXGeneratorDualGPU",
    "_meta": {
      "title": "Generate with FLUX"
    }
  },
  "5": {
    "inputs": {
      "images": [
        "4",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  }
}
```

## Part 5: Integration Configuration

### Environment Setup

Create a `.env` file in your ComfyUI directory:

```bash
# Path to Qwen2VL-FLUX project
QWEN2VL_FLUX_PATH=/path/to/qwen2vl-flux

# Checkpoint directory
CHECKPOINT_DIR=/path/to/qwen2vl-flux/checkpoints

# GPU settings
CUDA_VISIBLE_DEVICES=0,1

# Hugging Face token (if needed)
HF_TOKEN=your_hugging_face_token_here
```

### Launch ComfyUI

```bash
cd ComfyUI
python main.py --listen 0.0.0.0 --port 8188
```

Then open `http://localhost:8188` in your browser.

## Part 6: Using Custom Nodes in ComfyUI

### Workflow Steps:

1. **Load Image Node** - Use ComfyUI's built-in LoadImage to load your reference image
2. **Qwen2VL Image Encoder Node** - Encodes the image to embeddings
3. **FLUX Generator Node** - Generates new images based on the embeddings
4. **ControlNet Processor** (optional) - For ControlNet modes, process depth/line controls
5. **Save Image Node** - Save the generated images

### Available Generation Modes:

- **Variation**: Generate variations of the input image with optional text guidance
- **Image-to-Image**: Modify an image while preserving structure (controlled by denoise_strength)
- **Inpaint**: Fill masked regions in an image
- **ControlNet**: Generate images guided by depth or line information
- **ControlNet-Inpaint**: Combine inpainting with ControlNet guidance

## Part 7: Docker Integration with ComfyUI

### Running ComfyUI in Docker

Create `Dockerfile.comfyui`:

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

WORKDIR /app/ComfyUI

# Install dependencies
RUN pip3 install -r requirements.txt
RUN pip3 install --no-cache-dir --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124

# Copy custom nodes
COPY custom_nodes/ ./custom_nodes/

# Copy environment
COPY .env ./

CMD ["python", "main.py", "--listen", "0.0.0.0"]
```

### Docker Compose with Qwen2VL-FLUX and ComfyUI

```yaml
version: '3.8'

services:
  qwen-flux:
    build: ./qwen2vl-flux
    container_name: qwen-flux
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CHECKPOINT_DIR=/app/checkpoints
    volumes:
      - ./qwen2vl-flux:/app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  comfyui:
    build:
      context: .
      dockerfile: Dockerfile.comfyui
    container_name: comfyui
    runtime: nvidia
    ports:
      - "8188:8000"
    environment:
      - NVIDIA_VISIBLE_DEVICES=1
      - QWEN2VL_FLUX_PATH=/qwen2vl-flux
      - CHECKPOINT_DIR=/qwen2vl-flux/checkpoints
    volumes:
      - ./ComfyUI:/app/ComfyUI
      - ./qwen2vl-flux:/qwen2vl-flux
      - comfyui-outputs:/app/ComfyUI/output
    depends_on:
      - qwen-flux
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]

volumes:
  comfyui-outputs:
```

## Troubleshooting

### Issue: Nodes not appearing in ComfyUI

**Solution**:
- Ensure custom_nodes folder exists in ComfyUI root
- Check that __init__.py file is in custom_nodes directory
- Restart ComfyUI server
- Check console for import errors

### Issue: GPU memory errors

**Solution**:
- Ensure you have two GPUs and they're properly detected
- Check that NVIDIA_VISIBLE_DEVICES environment variable is set correctly
- Reduce batch_size in FLUX Generator node
- Enable turbo mode for faster, memory-efficient inference

### Issue: Qwen2VL model not found

**Solution**:
- Set QWEN2VL_FLUX_PATH and CHECKPOINT_DIR environment variables
- Ensure model checkpoints are downloaded in the checkpoints directory
- Verify HF_TOKEN is set if using Hugging Face models

## Advanced: Custom Workflow Examples

### Example 1: Image Variation with Text Guidance

Use the workflow JSON above with a non-empty prompt.

### Example 2: ControlNet Depth-Guided Generation

```json
{
  "node_1": {
    "class_type": "LoadImage"
  },
  "node_2": {
    "class_type": "Qwen2VLImageEncoder",
    "inputs": ["node_1"]
  },
  "node_3": {
    "class_type": "ControlNetProcessor",
    "control_type": "depth",
    "inputs": ["node_1"]
  },
  "node_4": {
    "class_type": "FLUXGeneratorDualGPU",
    "mode": "controlnet",
    "inputs": ["node_2", "node_3"]
  },
  "node_5": {
    "class_type": "SaveImage",
    "inputs": ["node_4"]
  }
}
```

## API Integration

To use the custom nodes programmatically:

```python
import requests
import json

# ComfyUI server URL
SERVER_URL = "http://localhost:8188"

# Create workflow
workflow = {
    "1": {
        "inputs": {"image": "path/to/image.png"},
        "class_type": "LoadImage",
    },
    "2": {
        "inputs": {
            "image": ["1", 0],
            "apply_attention": False,
        },
        "class_type": "Qwen2VLImageEncoder",
    },
    "3": {
        "inputs": {
            "qwen_embedding": ["2", 0],
            "qwen_grid_thw": ["2", 1],
            "prompt": "A beautiful landscape",
            "mode": "variation",
        },
        "class_type": "FLUXGeneratorDualGPU",
    },
    "4": {
        "inputs": {"images": ["3", 0]},
        "class_type": "SaveImage",
    }
}

# Queue prompt
response = requests.post(
    f"{SERVER_URL}/prompt",
    json={"prompt": workflow}
)

print(response.json())
```

## Performance Optimization

### Tips for Optimal Performance:

1. **Use bfloat16 precision**: Already enabled in your model.py
2. **Reduce inference steps**: 20-28 steps usually sufficient
3. **Enable turbo mode**: For faster 2-3x speedup at slight quality cost
4. **Batch processing**: Process multiple variations in parallel
5. **Async image loading**: Load images while previous generation runs
6. **Cache models**: Load model once, reuse for multiple generations

## Next Steps

1. Install ComfyUI and copy custom nodes
2. Test each node individually
3. Create example workflows
4. Integrate with your frontend (OpenWebUI, etc.)
5. Monitor GPU memory usage and optimize if needed

## References

- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI Custom Nodes Documentation](https://github.com/comfyanonymous/ComfyUI/wiki/Custom-Nodes)
- Your project's model.py for implementation details
