import torch
import os
import folder_paths
import numpy as np
from .streamdiffusionwrapper import StreamDiffusionWrapper
from .utils import Engine
import inspect
from PIL import Image

# Define constants for model paths
ENGINE_DIR = os.path.join(folder_paths.models_dir, "StreamDiffusion--engines", "KBlueLeaf", "kohaku-v2.1--lcm_lora-True--tiny_vae-True--max_batch-1--min_batch-1--mode-img2img")

def get_wrapper_defaults(param_names):
    """Helper function to get default values from StreamDiffusionWrapper parameters
    Args:
        param_names (list): List of parameter names to extract
    Returns:
        dict: Dictionary of parameter names and their default values
    """
    wrapper_params = inspect.signature(StreamDiffusionWrapper).parameters
    return {name: wrapper_params[name].default for name in param_names if name in wrapper_params}

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

class StreamDiffusionConfig:
    @classmethod
    def INPUT_TYPES(s):
        defaults = get_wrapper_defaults([
            "mode", "width", "height", "acceleration", "frame_buffer_size",
            "use_tiny_vae", "cfg_type"
        ])
        
        return {
            "required": {
                "engine": (os.listdir(ENGINE_DIR),),
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
            }
        }

    RETURN_TYPES = ("STREAM_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "StreamDiffusion"

    def load_model(self, engine, t_index_list, mode, width, height, acceleration, 
                  frame_buffer_size, use_tiny_vae, cfg_type, use_lcm_lora,
                  opt_lora_dict=None, opt_acceleration_config=None,
                  opt_similarity_filter_config=None,
                  ):
        

        # setup tensorrt engine
        if (not hasattr(self, 'engine') or self.engine_label != engine):
            self.engine = Engine(os.path.join(ENGINE_DIR,engine))
            self.engine.load()
            self.engine.activate()
            self.engine.allocate_buffers()
            self.engine_label = engine


        t_index_list = [int(x.strip()) for x in t_index_list.split(",")]
        
        # Build base configuration with all current parameters
        config = {
            "model_id_or_path": engine,
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

        # Add acceleration config if provided
        if opt_acceleration_config:
            config.update(opt_acceleration_config)

        # Add similarity filter config if provided
        if opt_similarity_filter_config:
            config.update(opt_similarity_filter_config)

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
    # "StreamDiffusionLoraLoader": StreamDiffusionLoraLoader,
    # "StreamDiffusionLcmLoraLoader": StreamDiffusionLcmLoraLoader,
    # "StreamDiffusionVaeLoader": StreamDiffusionVaeLoader,
    "StreamDiffusionAccelerationConfig": StreamDiffusionAccelerationConfig,
    "StreamDiffusionSimilarityFilterConfig": StreamDiffusionSimilarityFilterConfig,
    # "StreamDiffusionModelLoader": StreamDiffusionModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StreamDiffusionConfig": "StreamDiffusionConfig",
    "StreamDiffusionAccelerationSampler": "StreamDiffusionAccelerationSampler",
    # "StreamDiffusionLoraLoader": "StreamDiffusionLoraLoader",
    "StreamDiffusionAccelerationConfig": "StreamDiffusionAccelerationConfig",
    "StreamDiffusionSimilarityFilterConfig": "StreamDiffusionSimilarityFilterConfig", 
    # "StreamDiffusionModelLoader": "StreamDiffusionModelLoader",
}
