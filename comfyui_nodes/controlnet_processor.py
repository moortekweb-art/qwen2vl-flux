"""
ComfyUI Custom Node: ControlNet Processor
Handles depth and line detection for ControlNet guidance
"""

import torch
import numpy as np
from PIL import Image
import os
import sys
import logging

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from model import FluxModel
    HAS_CONTROLNET = True
except ImportError:
    HAS_CONTROLNET = False
    logger.warning("ControlNet modules not available.")


class ControlNetProcessor:
    """
    Processes images to extract depth maps and edge/line maps for ControlNet.
    Supports depth estimation and edge detection.
    """

    def __init__(self):
        self.model = None

    def load_model(self):
        """Lazy load the FLUX model with ControlNet features"""
        if self.model is None:
            if not HAS_CONTROLNET:
                raise RuntimeError("ControlNet modules not available.")

            try:
                self.model = FluxModel(
                    is_turbo=False,
                    device="cuda",
                    required_features=['depth', 'line']
                )
                logger.info("ControlNet processor model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load ControlNet processor: {e}")
                raise

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

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("control_image", "processed_image")
    FUNCTION = "process"
    CATEGORY = "ControlNet"

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

    def process(
        self,
        image,
        control_type,
        canny_low_threshold=50,
        canny_high_threshold=150,
    ):
        """
        Process image to extract control signals.

        Args:
            image: Input IMAGE tensor (B, H, W, C)
            control_type: Type of control to extract ('depth', 'line', or 'both')
            canny_low_threshold: Low threshold for Canny edge detection
            canny_high_threshold: High threshold for Canny edge detection

        Returns:
            Tuple of (control_image, processed_image) tensors
        """
        self.load_model()

        try:
            pil_image = self.convert_tensor_to_pil(image)

            if pil_image is None:
                raise ValueError("Input image is None")

            logger.info(f"Processing {control_type} control for image: {pil_image.size}")

            control_images = []

            if control_type in ["depth", "both"]:
                try:
                    depth_image = self.model.generate_depth_map(pil_image)
                    control_images.append(depth_image)
                    logger.info(f"Depth map generated: {depth_image.size}")
                except Exception as e:
                    logger.warning(f"Failed to generate depth map: {e}")

            if control_type in ["line", "both"]:
                try:
                    line_image = self.model.generate_canny_edges(
                        pil_image,
                        low_threshold=canny_low_threshold,
                        high_threshold=canny_high_threshold
                    )
                    control_images.append(line_image)
                    logger.info(f"Line map generated: {line_image.size}")
                except Exception as e:
                    logger.warning(f"Failed to generate line map: {e}")

            if not control_images:
                # Fallback: return original image
                logger.warning("No control images generated, returning original")
                control_image = pil_image
            elif control_type == "both" and len(control_images) == 2:
                # Combine depth and line
                depth, lines = control_images
                # Simple combination: average the two
                depth_arr = np.array(depth, dtype=np.float32) / 255.0
                lines_arr = np.array(lines, dtype=np.float32) / 255.0
                combined = ((depth_arr + lines_arr) / 2 * 255).astype(np.uint8)
                control_image = Image.fromarray(combined)
                logger.info("Combined depth and line controls")
            else:
                control_image = control_images[0]

            # Convert output tensors
            control_tensor = self.convert_pil_to_tensor(control_image)
            original_tensor = self.convert_pil_to_tensor(pil_image)

            logger.info(f"Control tensor shape: {control_tensor.shape}")

            return (control_tensor, original_tensor)

        except Exception as e:
            logger.error(f"Error processing control: {e}", exc_info=True)
            raise


class ControlNetDenoiser:
    """
    Optional denoising node to clean up ControlNet outputs
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_image": ("IMAGE",),
                "blur_kernel": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("denoised_control",)
    FUNCTION = "denoise"
    CATEGORY = "ControlNet"

    def denoise(self, control_image, blur_kernel=3):
        """
        Denoise and smooth control image.

        Args:
            control_image: Input control IMAGE tensor
            blur_kernel: Gaussian blur kernel size

        Returns:
            Denoised control image tensor
        """
        try:
            import cv2

            if len(control_image.shape) == 4:
                control_image = control_image[0]

            img_numpy = (control_image.cpu().numpy() * 255).astype(np.uint8)

            # Apply Gaussian blur
            if len(img_numpy.shape) == 3:
                denoised = cv2.GaussianBlur(img_numpy, (blur_kernel, blur_kernel), 0)
            else:
                denoised = cv2.GaussianBlur(img_numpy, (blur_kernel, blur_kernel), 0)

            # Normalize and convert back to tensor
            denoised_float = denoised.astype(np.float32) / 255.0
            output_tensor = torch.from_numpy(denoised_float).unsqueeze(0)

            return (output_tensor,)

        except Exception as e:
            logger.error(f"Error denoising: {e}")
            raise


NODE_CLASS_MAPPINGS = {
    "ControlNetProcessor": ControlNetProcessor,
    "ControlNetDenoiser": ControlNetDenoiser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ControlNetProcessor": "ControlNet Processor (Depth/Line)",
    "ControlNetDenoiser": "ControlNet Denoiser",
}
