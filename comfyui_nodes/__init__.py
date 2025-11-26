"""
Qwen2VL-FLUX Custom Nodes for ComfyUI

This module provides custom nodes for integrating the Qwen2VL-FLUX pipeline with ComfyUI.

Available Nodes:
- Qwen2VLImageEncoder: Encodes images using Qwen2VL to produce embeddings
- FLUXGeneratorDualGPU: Generates images using optimized FLUX pipeline
- ControlNetProcessor: Extracts depth and line controls for ControlNet
- ControlNetDenoiser: Optional post-processing for control images

Usage:
1. Copy this folder to ComfyUI/custom_nodes/
2. Install dependencies: pip install -r requirements.txt
3. Restart ComfyUI server
4. Nodes will appear in the node menu under Qwen2VL and FLUX categories
"""

import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Loading Qwen2VL-FLUX Custom Nodes for ComfyUI")

# Import custom nodes
try:
    from . import qwen2vl_image_encoder
    from . import flux_generator_dual_gpu
    from . import controlnet_processor
    logger.info("Successfully imported all custom node modules")
except ImportError as e:
    logger.error(f"Failed to import custom node modules: {e}")
    raise

# Combine all node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Register Qwen2VL nodes
NODE_CLASS_MAPPINGS.update(qwen2vl_image_encoder.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(qwen2vl_image_encoder.NODE_DISPLAY_NAME_MAPPINGS)

# Register FLUX nodes
NODE_CLASS_MAPPINGS.update(flux_generator_dual_gpu.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(flux_generator_dual_gpu.NODE_DISPLAY_NAME_MAPPINGS)

# Register ControlNet nodes
NODE_CLASS_MAPPINGS.update(controlnet_processor.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(controlnet_processor.NODE_DISPLAY_NAME_MAPPINGS)

logger.info(f"Registered {len(NODE_CLASS_MAPPINGS)} custom nodes")

# Log available nodes
for node_name in NODE_CLASS_MAPPINGS.keys():
    display_name = NODE_DISPLAY_NAME_MAPPINGS.get(node_name, node_name)
    logger.info(f"  - {display_name} ({node_name})")

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
]
