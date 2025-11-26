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
import logging

logger = logging.getLogger(__name__)

# Add parent directory to path to import from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from model import FluxModel
    from transformers import AutoProcessor
    HAS_QWEN2VL = True
except ImportError:
    HAS_QWEN2VL = False
    logger.warning("Qwen2VL modules not available. Install qwen2vl-flux project.")


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
            if not HAS_QWEN2VL:
                raise RuntimeError(
                    "Qwen2VL not available. Ensure qwen2vl-flux project is in parent directory."
                )

            checkpoint_dir = os.getenv('CHECKPOINT_DIR', os.path.join(parent_dir, 'checkpoints'))

            try:
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
                logger.info("Qwen2VL model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Qwen2VL model: {e}")
                raise

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI IMAGE type (B, H, W, C) in [0, 1]
                "apply_attention": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "radius": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("QWEN_EMBEDDING", "QWEN_GRID_THW")
    RETURN_NAMES = ("qwen_embedding", "grid_thw")
    FUNCTION = "encode"
    CATEGORY = "Qwen2VL"

    def encode(self, image, apply_attention, center_x=0.5, center_y=0.5, radius=0.5):
        """
        Encode image to Qwen2VL embeddings.

        Args:
            image: ComfyUI IMAGE tensor (B, H, W, C) in [0, 1]
            apply_attention: Whether to apply attention masking
            center_x: Attention center X coordinate (0-1)
            center_y: Attention center Y coordinate (0-1)
            radius: Attention radius (0-1)

        Returns:
            Tuple of (qwen_embedding, grid_thw)
        """
        self.load_model()

        try:
            # Convert ComfyUI image format (B, H, W, C) [0, 1] to PIL Image
            # For single image: take first batch
            if len(image.shape) == 4:
                img_tensor = image[0]  # (H, W, C)
            else:
                img_tensor = image  # Already (H, W, C)

            img_numpy = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_numpy)

            logger.info(f"Processing image: {pil_image.size}")

            # Process image through Qwen2VL
            qwen2_hidden_state, image_grid_thw = self.model.process_image(pil_image)

            # Apply attention masking if requested
            if apply_attention:
                logger.info(f"Applying attention mask at ({center_x}, {center_y}) with radius {radius}")
                qwen2_hidden_state = self.model.apply_attention(
                    qwen2_hidden_state,
                    image_grid_thw,
                    center_x,
                    center_y,
                    radius
                )

            logger.info(f"Embedding shape: {qwen2_hidden_state.shape}, Grid THW: {image_grid_thw}")

            return (qwen2_hidden_state, image_grid_thw)

        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise


# Register node
NODE_CLASS_MAPPINGS = {
    "Qwen2VLImageEncoder": Qwen2VLImageEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen2VLImageEncoder": "Qwen2VL Image Encoder",
}
