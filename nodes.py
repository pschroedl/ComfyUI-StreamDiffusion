import torch
import logging
import os, sys
from pathlib import Path
import folder_paths
import numpy as np
from .streamdiffusionwrapper import StreamDiffusionWrapper
import inspect
from PIL import Image
import folder_paths

# Define constants for model paths
MODELS_ROOT = folder_paths.models_dir

def get_engine_dir(model_id: str) -> str:
    """Get the engine directory for a specific model"""
    # Handle full snapshot paths
    if "snapshots" in model_id:
        if "KBlueLeaf--kohaku-v2.1" in model_id:
            model_name = "KBlueLeaf"
        elif "stabilityai--sd-turbo" in model_id:
            model_name = "stabilityai"
        else:
            model_name = model_id.split('/')[-1]  # fallback behavior
    # Handle huggingface-style IDs
    elif model_id == "stabilityai/sd-turbo":
        model_name = "stabilityai"
    elif model_id == "KBlueLeaf/kohaku-v2.1":
        model_name = "KBlueLeaf"
    else:
        model_name = model_id.split('/')[-1]  # fallback behavior
        
    engine_dir = os.path.join(MODELS_ROOT, "StreamDiffusion--engines", model_name)
    print(f"Using engine directory: {engine_dir}")  # Debug print
    return engine_dir

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
    #NOTE: this should ultimately point to the regular COMFYUI Lora directory. 
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

def get_wrapper_defaults(param_names):
    """Helper function to get default values from StreamDiffusionWrapper parameters
    Args:
        param_names (list): List of parameter names to extract
    Returns:
        dict: Dictionary of parameter names and their default values
    """
    wrapper_params = inspect.signature(StreamDiffusionWrapper).parameters
    return {name: wrapper_params[name].default for name in param_names if name in wrapper_params}

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

class StreamDiffusionAccelerationConfig:
    @classmethod
    def INPUT_TYPES(s):
        defaults = get_wrapper_defaults(["warmup", "do_add_noise", "use_denoising_batch"])
        return {
            "required": {
                "warmup": ("INT", {"default": defaults["warmup"], "min": 0, "max": 100}),
                "do_add_noise": ("BOOLEAN", {"default": defaults["do_add_noise"]}),
                "use_denoising_batch": ("BOOLEAN", {"default": defaults["use_denoising_batch"]}),
            }
        }

    RETURN_TYPES = ("ACCELERATION_CONFIG",)
    FUNCTION = "get_acceleration_config"
    CATEGORY = "StreamDiffusion"

    def get_acceleration_config(self, warmup, do_add_noise, use_denoising_batch):
        return ({
            "warmup": warmup,
            "do_add_noise": do_add_noise,
            "use_denoising_batch": use_denoising_batch
        },)

class StreamDiffusionSimilarityFilterConfig:
    @classmethod
    def INPUT_TYPES(s):
        defaults = get_wrapper_defaults([
            "enable_similar_image_filter",
            "similar_image_filter_threshold",
            "similar_image_filter_max_skip_frame"
        ])
        return {
            "required": {
                "enable_similar_image_filter": ("BOOLEAN", {"default": defaults["enable_similar_image_filter"]}),
                "similar_image_filter_threshold": ("FLOAT", {"default": defaults["similar_image_filter_threshold"], "min": 0.0, "max": 1.0}),
                "similar_image_filter_max_skip_frame": ("INT", {"default": defaults["similar_image_filter_max_skip_frame"], "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("SIMILARITY_FILTER_CONFIG",)
    FUNCTION = "get_similarity_filter_config"
    CATEGORY = "StreamDiffusion"

    def get_similarity_filter_config(self, enable_similar_image_filter, similar_image_filter_threshold, similar_image_filter_max_skip_frame):
        return ({
            "enable_similar_image_filter": enable_similar_image_filter,
            "similar_image_filter_threshold": similar_image_filter_threshold,
            "similar_image_filter_max_skip_frame": similar_image_filter_max_skip_frame
        },)

class StreamDiffusionModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_id_or_path": (["stabilityai/sd-turbo", "KBlueLeaf/kohaku-v2.1"], {"default": "stabilityai/sd-turbo"}),
            }
        }   

    RETURN_TYPES = ("STREAMDIFFUSION_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "StreamDiffusion"

    def load_model(self, model_id_or_path):
        if model_id_or_path in ["stabilityai/sd-turbo", "KBlueLeaf/kohaku-v2.1"]:
            model_path = get_model_path(model_id_or_path)
            if model_path != model_id_or_path:  # Only use local path if it exists
                model_id_or_path = model_path
                print(f"Using local model path: {model_path}")
        return (model_id_or_path,)

class StreamDiffusionConfig:
    @classmethod
    def INPUT_TYPES(s):
        defaults = get_wrapper_defaults([
            "mode", "width", "height", "acceleration", "frame_buffer_size",
            "use_tiny_vae", "cfg_type"
        ])
        
        return {
            "required": {
                "model": ("STREAMDIFFUSION_MODEL",),
                "t_index_list": ("STRING", {"default": "39,35,30"}),
                "mode": (["img2img", "txt2img"], {"default": defaults["mode"]}),
                "width": ("INT", {"default": defaults["width"], "min": 64, "max": 2048}),
                "height": ("INT", {"default": defaults["height"], "min": 64, "max": 2048}),
                "acceleration": (["none", "xformers", "tensorrt"], {"default": defaults["acceleration"]}),
                "frame_buffer_size": ("INT", {"default": defaults["frame_buffer_size"], "min": 1, "max": 16}),
                "use_tiny_vae": ("BOOLEAN", {"default": defaults["use_tiny_vae"]}),
                "cfg_type": (["none", "full", "self", "initialize"], {"default": defaults["cfg_type"]}),
                "use_lcm_lora": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "opt_lora_dict": ("LORA_DICT",),
                "opt_acceleration_config": ("ACCELERATION_CONFIG",),
                "opt_similarity_filter_config": ("SIMILARITY_FILTER_CONFIG",),
                # "opt_lcm_lora_path": ("LCM_LORA_PATH",),
                # "opt_vae_path": ("VAE_PATH",),
            }
        }

    RETURN_TYPES = ("STREAM_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "StreamDiffusion"

    def load_model(self, model, t_index_list, mode, width, height, acceleration, 
                  frame_buffer_size, use_tiny_vae, cfg_type, use_lcm_lora,
                  opt_lora_dict=None, opt_acceleration_config=None,
                  opt_similarity_filter_config=None,
                  #opt_lcm_lora_path=None, opt_vae_path=None,
                  ):
        
        t_index_list = [int(x.strip()) for x in t_index_list.split(",")]
        
        # Build base configuration with all current parameters
        config = {
            "model_id_or_path": model,
            "t_index_list": t_index_list,
            "mode": mode,
            "width": width,
            "height": height,
            "acceleration": acceleration,
            "frame_buffer_size": frame_buffer_size,
            "use_tiny_vae": use_tiny_vae,
            "cfg_type": cfg_type,
            "use_lcm_lora": use_lcm_lora,
            "device": "cuda",
            "dtype": torch.float16,
            "output_type": "pil",
            "do_add_noise": True,
            "use_denoising_batch": True,
        }

        if opt_lora_dict:
            config["lora_dict"] = opt_lora_dict

        # if opt_lcm_lora_path:
        #     config["lcm_lora_id"] = opt_lcm_lora_path

        # if opt_vae_path is not None:
        #     config["vae_id"] = opt_vae_path.strip() if opt_vae_path.strip() else None

        # Add acceleration config if provided
        if opt_acceleration_config:
            config.update(opt_acceleration_config)

        # Add similarity filter config if provided
        if opt_similarity_filter_config:
            config.update(opt_similarity_filter_config)

        engine_dir = get_engine_dir(model)
        wrapper = StreamDiffusionWrapper(**config)

        return (wrapper,)

class StreamDiffusionAccelerationSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stream_model": ("STREAM_MODEL",),
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
        
        stream_model.prepare(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta
        )

        # Warmup loop for img2img mode
        if stream_model.mode == "img2img" and image is not None:
            image_tensor = stream_model.preprocess_image(
                Image.fromarray((image[0].numpy() * 255).astype(np.uint8))
            )
            # Perform warmup iterations
            for _ in range(stream_model.batch_size - 1):
                stream_model(image=image_tensor)
            # Final generation
            output = stream_model(image=image_tensor)
        else:
            output = stream_model.txt2img()
        
        output_array = np.array(output)
        
        # Convert to tensor and normalize to 0-1 range
        output_tensor = torch.from_numpy(output_array).float() / 255.0
        
        # Ensure BHWC format
        if len(output_tensor.shape) == 3:  # If HWC
            output_tensor = output_tensor.unsqueeze(0)  # Add batch dimension -> BHWC
        
        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "StreamDiffusionConfig": StreamDiffusionConfig,
    "StreamDiffusionAccelerationSampler": StreamDiffusionAccelerationSampler,
    "StreamDiffusionLoraLoader": StreamDiffusionLoraLoader,
    # "StreamDiffusionLcmLoraLoader": StreamDiffusionLcmLoraLoader,
    # "StreamDiffusionVaeLoader": StreamDiffusionVaeLoader,
    "StreamDiffusionAccelerationConfig": StreamDiffusionAccelerationConfig,
    "StreamDiffusionSimilarityFilterConfig": StreamDiffusionSimilarityFilterConfig,
    "StreamDiffusionModelLoader": StreamDiffusionModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StreamDiffusionConfig": "StreamDiffusionConfig",
    "StreamDiffusionAccelerationSampler": "StreamDiffusionAccelerationSampler",
    "StreamDiffusionLoraLoader": "StreamDiffusionLoraLoader",
    # "StreamDiffusionLcmLoraLoader": "StreamDiffusionLcmLoraLoader",
    # "StreamDiffusionVaeLoader": "StreamDiffusionVaeLoader",
    "StreamDiffusionAccelerationConfig": "StreamDiffusionAccelerationConfig",
    "StreamDiffusionSimilarityFilterConfig": "StreamDiffusionSimilarityFilterConfig", 
    "StreamDiffusionModelLoader": "StreamDiffusionModelLoader",
}
