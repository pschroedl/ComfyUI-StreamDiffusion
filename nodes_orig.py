import torch
import logging
import os, sys

from typing import Dict, Optional, List
# Add the directory containing 'streamdiffusionwrapper' to sys.path
current_directory = os.path.dirname(os.path.abspath(__file__))
streamdiffusion_path = os.path.join(current_directory)  # Adjust the relative path
sys.path.append(streamdiffusion_path)

from streamdiffusionwrapper import StreamDiffusionWrapper


class DownloadAndLoadStreamDiffusionModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": (["stabilityai/sd-turbo"],),  # Changed from KBlueLeaf/kohaku-v2.1
                "device": (["cuda", "cpu", "mps"],),
                "lora_dict": ("DICT",),
                "use_lcm_lora": (["true", "false"],),
                "lcm_lora_id": ("TEXT",),
                "acceleration": (["tensorrt", "xformers", "none"],),  # Changed default to prioritize tensorrt
            },
        }

    RETURN_TYPES = ("STREAMDIFFUSIONMODEL",)
    RETURN_NAMES = ("streamdiffusion_model",)
    FUNCTION = "_load_model"
    CATEGORY = "StreamDiffusion"

    def _load_model(
        self,
        model_id: str,
        device: str,
        lora_dict: Optional[Dict[str, float]],
        use_lcm_lora: str,
        lcm_lora_id: str,
        acceleration: str,
    ):
        # Initialize the StreamDiffusionWrapper
        pipe = StreamDiffusionWrapper(
            model_id_or_path=model_id,
            lora_dict=lora_dict,
            use_lcm_lora=(use_lcm_lora.lower() == "true"),
            lcm_lora_id=lcm_lora_id,
            acceleration=acceleration,
            model_dir="/workspace/models"  #NOTE: this is temporary
        )
        
        # Configure for real-time performance
        pipe.prepare(
            prompt="Default prompt",
            num_inference_steps=1,  # Reduced for speed
            guidance_scale=1.0,     # Reduced for speed
        )

        return ({"pipe": pipe, "device": device},)


class StreamDiffusion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "streamdiffusion_model": ("STREAMDIFFUSIONMODEL",),
                "prompt": ("STRING",),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0}),
            },
        }

    RETURN_NAMES = ("PROCESSED_IMAGES",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "_process_images"
    CATEGORY = "StreamDiffusion"

    def _process_images(
        self,
        images,
        streamdiffusion_model,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
    ):
        pipe = streamdiffusion_model["pipe"]
        device = streamdiffusion_model["device"]

        # Update pipeline parameters
        pipe.stream.update_prompt(prompt)
        pipe.stream.update_steps(num_inference_steps)
        pipe.stream.update_guidance_scale(guidance_scale)

        processed_frames = []
        for img in images:
            img_tensor = pipe.preprocess_image(img).to(device)
            with torch.no_grad():
                processed_img = pipe(image=img_tensor)
            processed_frames.append(processed_img)

        # Stack processed frames into a single tensor
        stacked_frames = torch.stack(processed_frames)
        return (stacked_frames,)


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadStreamDiffusionModel": DownloadAndLoadStreamDiffusionModel,
    "StreamDiffusion": StreamDiffusion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadStreamDiffusionModel": "(Down)Load StreamDiffusion Model",
    "StreamDiffusion": "StreamDiffusion",
}
