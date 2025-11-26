"""
ComfyUI Custom Node: FLUX Generator with Dual-GPU Support
Generates images using the optimized Qwen2VL-FLUX pipeline
"""

import torch
import numpy as np
from PIL import Image
import os
import sys
import logging

logger = logging.getLogger(__name__)

# Use realpath to resolve symlinks first
real_file = os.path.realpath(__file__)
current_dir = os.path.dirname(real_file)
# Go up one level: comfyui_nodes/ -> qwen2vl-flux/ (project root)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from model import FluxModel
    HAS_FLUX = True
except ImportError:
    HAS_FLUX = False
    logger.warning("FLUX modules not available.")


class FLUXGeneratorDualGPU:
    """
    Generates images using FLUX with dual-GPU memory optimization.
    Handles Variation, Image-to-Image, Inpainting, and ControlNet modes.
    """

    def __init__(self):
        self.model = None
        self.last_mode = None
        self.last_controlnet_enabled = False

    def load_model(self, mode="variation", enable_controlnet=False):
        """Lazy load the FLUX model with required features"""
        # Reload model if mode or feature requirements changed
        needs_reload = (
            self.model is None or
            self.last_mode != mode or
            self.last_controlnet_enabled != enable_controlnet
        )

        if needs_reload:
            if not HAS_FLUX:
                raise RuntimeError("FLUX modules not available.")

            required_features = []
            if mode == "controlnet" or mode == "controlnet-inpaint" or enable_controlnet:
                required_features.extend(['controlnet', 'depth', 'line'])

            logger.info(f"Loading FLUX model for mode: {mode}, features: {required_features}")

            try:
                self.model = FluxModel(
                    is_turbo=False,
                    device="cuda",
                    required_features=required_features
                )
                self.last_mode = mode
                self.last_controlnet_enabled = enable_controlnet
                logger.info("FLUX model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load FLUX model: {e}")
                raise

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_embedding": ("QWEN_EMBEDDING",),
                "qwen_grid_thw": ("QWEN_GRID_THW",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "mode": (["variation", "img2img", "inpaint", "controlnet", "controlnet-inpaint"],),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 15.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "2.4:1", "3:4", "4:3"],),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            },
            "optional": {
                "input_image": ("IMAGE",),
                "input_image_b": ("IMAGE",),
                "mask_image": ("IMAGE",),
                "denoise_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "enable_depth_control": ("BOOLEAN", {"default": True}),
                "enable_line_control": ("BOOLEAN", {"default": True}),
                "depth_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "line_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "FLUX"

    def convert_tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image"""
        if tensor is None:
            return None

        if len(tensor.shape) == 4:  # Batch
            tensor = tensor[0]
        elif len(tensor.shape) != 3:
            raise ValueError(f"Invalid tensor shape: {tensor.shape}")

        img_numpy = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_numpy)

    def convert_pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor"""
        if pil_image is None:
            return None

        img_array = np.array(pil_image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(img_array)

        # Ensure proper shape (H, W, C)
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
        elif tensor.shape[2] == 4:  # RGBA
            tensor = tensor[:, :, :3]

        return tensor.unsqueeze(0)  # Add batch dimension

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
        input_image=None,
        input_image_b=None,
        mask_image=None,
        denoise_strength=0.8,
        enable_depth_control=True,
        enable_line_control=True,
        depth_strength=0.2,
        line_strength=0.4,
    ):
        """
        Generate images using FLUX pipeline.

        Args:
            qwen_embedding: Qwen2VL embeddings from encoder node
            qwen_grid_thw: Grid dimensions from encoder
            prompt: Text prompt for guidance
            mode: Generation mode (variation, img2img, etc.)
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of diffusion steps
            aspect_ratio: Output image aspect ratio
            batch_size: Number of images to generate
            seed: Random seed for reproducibility
            input_image: Input image (optional)
            input_image_b: Second input image (optional)
            mask_image: Inpainting mask (optional)
            denoise_strength: Strength for img2img/inpaint
            enable_depth_control: Enable depth guidance for ControlNet
            enable_line_control: Enable line guidance for ControlNet
            depth_strength: Strength of depth control
            line_strength: Strength of line control

        Returns:
            Tuple of output IMAGE tensor
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.load_model(
            mode=mode,
            enable_controlnet=(mode.startswith("controlnet"))
        )

        try:
            # Convert optional inputs
            input_image_a_pil = self.convert_tensor_to_pil(input_image)
            input_image_b_pil = self.convert_tensor_to_pil(input_image_b)
            mask_image_pil = self.convert_tensor_to_pil(mask_image)

            # Use dummy image if none provided
            if input_image_a_pil is None:
                logger.warning("No input image provided, using gray placeholder")
                input_image_a_pil = Image.new('RGB', (512, 512), color=(128, 128, 128))

            logger.info(f"Generating {batch_size} image(s) in '{mode}' mode")
            logger.info(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}, Aspect: {aspect_ratio}")

            # Generate images
            gen_images = self.model.generate(
                input_image_a=input_image_a_pil,
                input_image_b=input_image_b_pil,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                aspect_ratio=aspect_ratio,
                center_x=None,
                center_y=None,
                radius=None,
                mode=mode,
                denoise_strength=denoise_strength,
                mask_image=mask_image_pil,
                imageCount=batch_size,
                line_mode=enable_line_control,
                depth_mode=enable_depth_control,
                line_strength=line_strength,
                depth_strength=depth_strength,
            )

            logger.info(f"Generated {len(gen_images)} image(s)")

            # Convert PIL images to tensor
            output_tensors = []
            for i, img in enumerate(gen_images):
                tensor = self.convert_pil_to_tensor(img)
                output_tensors.append(tensor)
                logger.info(f"Image {i+1}: {img.size}")

            if not output_tensors:
                raise RuntimeError("No images were generated")

            output = torch.cat(output_tensors, dim=0)
            logger.info(f"Output shape: {output.shape}")

            return (output,)

        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            raise


NODE_CLASS_MAPPINGS = {
    "FLUXGeneratorDualGPU": FLUXGeneratorDualGPU,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FLUXGeneratorDualGPU": "FLUX Generator (Dual-GPU)",
}
