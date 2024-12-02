import torch
import logging
import os, sys
from pathlib import Path
import folder_paths
import numpy as np
from .streamdiffusionwrapper import StreamDiffusionWrapper

#NOTE: these commands could possibly be handled by a model loader
# huggingface-cli download KBlueLeaf/kohaku-v2.1 --include "*.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
# huggingface-cli download stabilityai/sd-turbo --include "*.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models


#NOTE: additional LCM https://huggingface.co/latent-consistency/lcm-lora-sdxl

#NOTE: image2image could be implicit when image is passed in

#NOTE: should we infer width and height when input image is provided? Enforce width and height settings specified in the node?

#NOTE: for acceleration options, should we expose other parameters like warmup, do_add_noise, and use_denoising_batch?

#NOTE: expose seed?

#NOTE: 

# Define constants for model paths
MODELS_ROOT = "/workspace/models"
ENGINE_DIR = os.path.join(MODELS_ROOT, "StreamDiffusion--engines")

def get_model_path(model_id: str) -> str:
    """Helper function to get the correct model path including snapshots"""
    base_path = os.path.join(MODELS_ROOT, f"models--{model_id.replace('/', '--')}")
    if not os.path.exists(base_path):
        return model_id
        
    # Get the snapshot directory
    snapshots_dir = os.path.join(base_path, "snapshots")
    if not os.path.exists(snapshots_dir):
        return model_id
        
    # Get the first (and usually only) snapshot directory
    snapshot_dirs = os.listdir(snapshots_dir)
    if not snapshot_dirs:
        return model_id
        
    return os.path.join(snapshots_dir, snapshot_dirs[0])

def get_lcm_loras():
    """Get list of available LCM LoRAs from StreamDiffusion's model directory"""
    lcm_path = os.path.join(MODELS_ROOT, "models--latent-consistency--lcm-lora-sdv1-5")
    if not os.path.exists(lcm_path):
        return ["latent-consistency/lcm-lora-sdv1-5"]  # Default option
    
    snapshots_dir = os.path.join(lcm_path, "snapshots")
    if not os.path.exists(snapshots_dir):
        return ["latent-consistency/lcm-lora-sdv1-5"]
        
    snapshot_dirs = os.listdir(snapshots_dir)
    if not snapshot_dirs:
        return ["latent-consistency/lcm-lora-sdv1-5"]
    
    return [os.path.join(snapshots_dir, snapshot_dirs[0])]

def get_available_vaes():
    """Get list of available VAEs from StreamDiffusion's model directory"""
    # Default tiny VAE
    vaes = ["madebyollin/taesd"]
    
    # Check for local VAEs
    vae_path = os.path.join(MODELS_ROOT, "models--madebyollin--taesd")
    if os.path.exists(vae_path):
        snapshots_dir = os.path.join(vae_path, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshot_dirs = os.listdir(snapshots_dir)
            if snapshot_dirs:
                vaes.append(os.path.join(snapshots_dir, snapshot_dirs[0]))
    
    return vaes

class StreamDiffusionLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "previous_loras": ("LORA_DICT",),  # Allow chaining
            }
        }

    RETURN_TYPES = ("LORA_DICT",)
    FUNCTION = "load_lora"
    CATEGORY = "StreamDiffusion"

    def load_lora(self, lora_name, strength, previous_loras=None):
        # Initialize with previous loras if provided
        lora_dict = {} if previous_loras is None else previous_loras.copy()
        
        # Add new lora to dictionary
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_dict[lora_path] = strength
        
        return (lora_dict,)

class StreamDiffusionLcmLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lcm_lora_path": (get_lcm_loras(), {"default": "latent-consistency/lcm-lora-sdv1-5"}),
            }
        }

    RETURN_TYPES = ("LCM_LORA_PATH",)
    FUNCTION = "load_lcm_lora"
    CATEGORY = "StreamDiffusion"

    def load_lcm_lora(self, lcm_lora_path):
        return (lcm_lora_path,)

class StreamDiffusionVaeLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_path": (get_available_vaes(), {"default": "madebyollin/taesd"}),
            }
        }

    RETURN_TYPES = ("VAE_PATH",)
    FUNCTION = "load_vae"
    CATEGORY = "StreamDiffusion"

    def load_vae(self, vae_path):
        return (vae_path,)

class StreamDiffusionBaseModelLoader:
    """Loads just the base model without additional components"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_id_or_path": (["stabilityai/sd-turbo", "KBlueLeaf/kohaku-v2.1"], {"default": "stabilityai/sd-turbo"}),
                "acceleration": (["tensorrt", "xformers", "none"], {"default": "tensorrt"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
            }
        }

    RETURN_TYPES = ("SD_BASE_MODEL",)
    FUNCTION = "load_base_model"
    CATEGORY = "StreamDiffusion"

    def load_base_model(self, model_id_or_path, acceleration, width, height):
        # Get local path if available
        model_path = get_model_path(model_id_or_path)
        if model_path != model_id_or_path:
            print(f"Using local model path: {model_path}")
            model_id_or_path = model_path

#TODO remove this and create wrapper object only once in configure node
        # Create basic StreamDiffusion instance without additional components
        wrapper = StreamDiffusionWrapper(
            model_id_or_path=model_id_or_path,
            t_index_list=[],  # Will be set in configure node
            mode="img2img",   # Will be set in configure node
            width=width,
            height=height,
            acceleration=acceleration,
            engine_dir=ENGINE_DIR
        )
        
        return (wrapper,)

class StreamDiffusionConfigureNode:
    """Configures loaded model with specific parameters"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_model": ("SD_BASE_MODEL",),
                "t_index_list": ("STRING", {"default": "39,35"}),
                "mode": (["img2img", "txt2img"], {"default": "img2img"}),
                "frame_buffer_size": ("INT", {"default": 1, "min": 1, "max": 16}),
                "use_lcm_lora": ("BOOLEAN", {"default": True}),
                "use_tiny_vae": ("BOOLEAN", {"default": True}),
                "cfg_type": (["none", "full", "self", "initialize"], {"default": "self"}),
            },
            "optional": {
                "lora_dict": ("LORA_DICT",),
                "lcm_lora_path": ("LCM_LORA_PATH",),
                "vae_path": ("VAE_PATH",),
            }
        }

    RETURN_TYPES = ("STREAMDIFFUSION",)
    FUNCTION = "configure_model"
    CATEGORY = "StreamDiffusion"

    def configure_model(self, base_model, t_index_list, mode, frame_buffer_size, 
                       use_lcm_lora, use_tiny_vae, cfg_type,
                       lora_dict=None, lcm_lora_path=None, vae_path=None):
        
        # Parse t_index_list from string
        t_indices = [int(x.strip()) for x in t_index_list.split(",")]
        
        # Update base model configuration
        base_model.t_index_list = t_indices
        base_model.mode = mode
        base_model.frame_buffer_size = frame_buffer_size
        base_model.use_lcm_lora = use_lcm_lora
        base_model.use_tiny_vae = use_tiny_vae
        base_model.cfg_type = cfg_type
        
        # Apply optional components
        if lora_dict:
            print(f"Using LoRAs: {lora_dict}")
            base_model.lora_dict = lora_dict
            
        if lcm_lora_path:
            print(f"Using custom LCM LoRA: {lcm_lora_path}")
            base_model.lcm_lora_id = lcm_lora_path
            
        if vae_path:
            base_model.vae_id = vae_path.strip()
        
        return (base_model,)

class StreamDiffusionGenerateNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stream_model": ("STREAMDIFFUSION",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 20.0}),
                "delta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "StreamDiffusion"

    def generate(self, stream_model, prompt, negative_prompt, num_inference_steps, 
                guidance_scale, delta, image=None):
        
        # Prepare the model with parameters
        stream_model.prepare(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta
        )

        # Generate based on mode
        if stream_model.mode == "img2img" and image is not None:
            # VAEs eexpect CHW
            image_tensor = image[0].permute(2, 0, 1)  # HWC -> CHW
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension: CHW -> BCHW
            # Ensure values are in [-1, 1] range as expected by the VAE
            image_tensor = (image_tensor * 2.0) - 1.0
            output = stream_model(image=image_tensor, prompt=prompt)
        else:
            output = stream_model(prompt=prompt)

        # Convert PIL Image or list of PIL Images to ComfyUI tensor format (BHWC)
        if isinstance(output, list):
            output = output[0]  # Take first image if list
        
        # Convert PIL Image to tensor in BHWC format
        output_tensor = torch.from_numpy(np.array(output)).float() / 255.0
        output_tensor = output_tensor.unsqueeze(0)  # Add batch dimension: HWC -> BHWC
        
        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "StreamDiffusionBaseLoader": StreamDiffusionBaseModelLoader,
    "StreamDiffusionConfigure": StreamDiffusionConfigureNode,
    "StreamDiffusionGenerate": StreamDiffusionGenerateNode,
    "StreamDiffusionLoraLoader": StreamDiffusionLoraLoader,
    "StreamDiffusionLcmLoraLoader": StreamDiffusionLcmLoraLoader,
    "StreamDiffusionVaeLoader": StreamDiffusionVaeLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StreamDiffusionBaseLoader": "SD Base Model Loader",
    "StreamDiffusionConfigure": "SD Model Configure",
    "StreamDiffusionGenerate": "StreamDiffusion Generator",
    "StreamDiffusionLoraLoader": "StreamDiffusion LoRA Loader",
    "StreamDiffusionLcmLoraLoader": "StreamDiffusion LCM LoRA Loader",
    "StreamDiffusionVaeLoader": "StreamDiffusion VAE Loader",
}
