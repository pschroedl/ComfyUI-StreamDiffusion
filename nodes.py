import torch
import os
import folder_paths
from .streamdiffusionwrapper import StreamDiffusionWrapper
import inspect

ENGINE_DIR = os.path.join(folder_paths.models_dir, "StreamDiffusion--engines")
LIVE_PEER_CHECKPOINT_DIR = os.path.join(folder_paths.models_dir,"models/ComfyUI--models/checkpoints")

_wrapper_params = None
def _get_wrapper_params():
    """Private function to get and cache StreamDiffusionWrapper parameters so consequential defaults for nodes are centrally managed"""
    global _wrapper_params
    if _wrapper_params is None:
        print("Getting wrapper params")
        params = inspect.signature(StreamDiffusionWrapper).parameters
        
        # Store parameters with overridden defaults
        _wrapper_params = {
            name: param.default for name, param in params.items()
        }
        # Override specific defaults
        _wrapper_params['engine_dir'] = os.path.join(folder_paths.models_dir, 'StreamDiffusion--engines')
        _wrapper_params['output_type'] = 'pt'
        
    return _wrapper_params

def get_engine_configs():
    """Scan StreamDiffusion--engines directory for available engine configurations"""
    if not os.path.exists(ENGINE_DIR):
        return []
    
    configs = []
    # Look for parent directories that contain complete engine sets
    for parent_dir in os.listdir(ENGINE_DIR):
        parent_path = os.path.join(ENGINE_DIR, parent_dir)
        if not os.path.isdir(parent_path):
            continue
            
        # Look for subdirectories containing the engines
        has_unet = False
        has_vae = False
        
        for subdir in os.listdir(parent_path):
            subdir_path = os.path.join(parent_path, subdir)
            if not os.path.isdir(subdir_path):
                continue
                
            # Check for engines
            if os.path.exists(os.path.join(subdir_path, "unet.engine")):
                has_unet = True
            if os.path.exists(os.path.join(subdir_path, "vae_encoder.engine")) and \
               os.path.exists(os.path.join(subdir_path, "vae_decoder.engine")):
                has_vae = True
        
        # Only add if we found both UNet and VAE engines
        if has_unet and has_vae:
            configs.append(parent_dir)
    
    return configs

def get_live_peer_checkpoints():
    """Get list of .safetensors files from LivePeer checkpoint directory"""

    if not os.path.exists(LIVE_PEER_CHECKPOINT_DIR):
        return []
    
    checkpoints = []
    for file in os.listdir(LIVE_PEER_CHECKPOINT_DIR):
        if file.endswith(".safetensors"):
            checkpoints.append(file)
            
    return checkpoints

#TODO: remove?
class StreamDiffusionTensorRTEngineLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "engine_name": (get_engine_configs(), ),
            }
        }

    RETURN_TYPES = ("SDMODEL",)
    FUNCTION = "load_engine"
    CATEGORY = "StreamDiffusion"

    def load_engine(self, engine_name):
        return (os.path.join(ENGINE_DIR, engine_name),)

#TODO: remove?
class StreamDiffusionLPCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_live_peer_checkpoints(), ),
            }
        }
    
    RETURN_TYPES = ("SDMODEL",)
    FUNCTION = "load_model"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Loads a model from LivePeer checkpoint directory. Interchangeable with StreamDiffusionCheckpointLoader."

    def load_model(self, model_name):
        mod = os.path.join(LIVE_PEER_CHECKPOINT_DIR, model_name)
        return (mod,)

class StreamDiffusionLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA model to load."}),
                "strength": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "The strength scale for the LoRA model. Higher values result in greater influence of the LoRA on the output."}),
            },
            "optional": {
                "previous_loras": ("LORA_DICT", {"tooltip": "Optional dictionary of previously loaded LoRAs to which the new LoRA will be added. Use this to combine multiple LoRAs."}),
            }
        }

    RETURN_TYPES = ("LORA_DICT",)
    OUTPUT_TOOLTIPS = ("Dictionary of loaded LoRA models.",)
    FUNCTION = "load_lora"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Loads a LoRA (Low-Rank Adaptation) model and adds it to the existing LoRA dictionary for application to the pipeline."

    def load_lora(self, lora_name, strength, previous_loras=None):
        # Initialize with previous loras if provided
        lora_dict = {} if previous_loras is None else previous_loras.copy()
        
        # Add new lora to dictionary
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_dict[lora_path] = strength
        
        return (lora_dict,)

class StreamDiffusionVaeLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"), ),
            }
        }

    RETURN_TYPES = ("VAE_PATH",)
    FUNCTION = "load_vae"
    CATEGORY = "StreamDiffusion"

    def load_vae(self, vae_name):
        vae_path = folder_paths.get_full_path("vae", vae_name)
        return (vae_path,)

class StreamDiffusionCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "checkpoint": (folder_paths.get_filename_list("checkpoints"), ),
            }
        }

    RETURN_TYPES = ("SDMODEL",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Loads a model from the ComfyUI checkpoint directory. Interchangeable with StreamDiffusionLPCheckpointLoader."
    def load_checkpoint(self, checkpoint):
        checkpoint_path = folder_paths.get_full_path("checkpoints", checkpoint)
        return (checkpoint_path,)

class StreamDiffusionAdvancedConfig:
    @classmethod
    def INPUT_TYPES(s):
        defaults = _get_wrapper_params()
        
        return {
            "required": {
                # Acceleration settings
                "warmup": ("INT", {"default": defaults["warmup"], "min": 0, "max": 100, 
                    "tooltip": "The number of warmup steps to perform before actual inference. Increasing this may improve stability at the cost of speed."}),
                "do_add_noise": ("BOOLEAN", {"default": defaults["do_add_noise"], 
                    "tooltip": "Whether to add noise during denoising steps. Enable this to allow the model to generate diverse outputs."}),
                "use_denoising_batch": ("BOOLEAN", {"default": defaults["use_denoising_batch"], 
                    "tooltip": "Whether to use batch denoising for performance optimization."}),
                
                # Similarity filter settings
                "enable_similar_image_filter": ("BOOLEAN", {"default": defaults["enable_similar_image_filter"], 
                    "tooltip": "Enable filtering out images that are too similar to previous outputs."}),
                "similar_image_filter_threshold": ("FLOAT", {"default": defaults["similar_image_filter_threshold"], "min": 0.0, "max": 1.0, 
                    "tooltip": "Threshold determining how similar an image must be to previous outputs to be filtered out (0.0 to 1.0)."}),
                "similar_image_filter_max_skip_frame": ("INT", {"default": defaults["similar_image_filter_max_skip_frame"], "min": 0, "max": 100, 
                    "tooltip": "Maximum number of frames to skip when filtering similar images."}),
                
                # Device settings
                "device": (["cuda", "cpu"], {"default": defaults["device"], 
                    "tooltip": "Device to run inference on. CPU will be significantly slower."}),
                "dtype": (["float16", "float32"], {"default": "float16" if defaults["dtype"] == torch.float16 else "float32", 
                    "tooltip": "Data type for inference. float16 uses less memory but may be less precise."}),
                "device_ids": ("STRING", {"default": str(defaults["device_ids"] or ""), 
                    "tooltip": "Comma-separated list of device IDs for multi-GPU support. Leave empty for single GPU."}),
                "use_safety_checker": ("BOOLEAN", {"default": defaults["use_safety_checker"], 
                    "tooltip": "Enable safety checker to filter NSFW content. May impact performance."}),
                "engine_dir": ("STRING", {"default": defaults["engine_dir"], 
                    "tooltip": "Directory for TensorRT engine files when using tensorrt acceleration."}),
            }
        }

    RETURN_TYPES = ("ADVANCED_CONFIG",)
    OUTPUT_TOOLTIPS = ("Combined configuration settings for acceleration, similarity filtering, and device parameters.",)
    FUNCTION = "get_config"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Configures advanced settings for StreamDiffusion including acceleration, similarity filtering, and device settings."

    def get_config(self, warmup, do_add_noise, use_denoising_batch,
                  enable_similar_image_filter, similar_image_filter_threshold, 
                  similar_image_filter_max_skip_frame,
                  device, dtype, device_ids, use_safety_checker, engine_dir):
        
        # Convert device_ids string to list if provided
        device_ids = [int(x.strip()) for x in device_ids.split(",")] if device_ids.strip() else None
        
        # Combine all settings into a single config dictionary
        advanced_config = {
            # Acceleration settings
            "warmup": warmup,
            "do_add_noise": do_add_noise,
            "use_denoising_batch": use_denoising_batch,
            
            # Similarity filter settings
            "enable_similar_image_filter": enable_similar_image_filter,
            "similar_image_filter_threshold": similar_image_filter_threshold,
            "similar_image_filter_max_skip_frame": similar_image_filter_max_skip_frame,
            
            # Device settings
            "device": device,
            "dtype": torch.float32 if dtype == "float32" else torch.float16,
            "device_ids": device_ids,
            "use_safety_checker": use_safety_checker,
            "engine_dir": engine_dir,
        }
        
        return (advanced_config,)

class StreamDiffusionSampler:
    _cached_config = None  # Will store config
    _cached_params = None  # Will store params
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stream_model": ("STREAM_MODEL", {"tooltip": "The configured StreamDiffusion model to use for generation."}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "The text prompt to guide image generation."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Text prompt specifying undesired aspects to avoid in the generated image."}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100, "tooltip": "The number of denoising steps. More steps often yield better results but take longer."}),
                "guidance_scale": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 20.0, "step": 0.01, "tooltip": "Controls the strength of the guidance. Higher values make the image more closely match the prompt."}),
                "delta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1, "tooltip": "Delta multiplier for virtual residual noise, affecting image diversity."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "The input image for image-to-image generation mode."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The generated image.",)
    FUNCTION = "generate"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Generates images using the configured StreamDiffusion model and specified prompts and settings."

    def generate(self, stream_model, prompt, negative_prompt, num_inference_steps, 
                guidance_scale, delta, image=None):
        
        #stream_model is a tuple of (model, config)
        model, config = stream_model
        
        current_params = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'delta': delta
        }

        needs_prepare = self._cached_params != current_params
        needs_warmup = self._cached_config != config
        
        if needs_prepare:
            self._cached_params = current_params
            model.prepare(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                delta=delta
            )
            
        if model.mode == "img2img" and image is not None:
            # Handle batch of images
            batch_size = image.shape[0]
            outputs = []
            
            for i in range(batch_size):
                # Convert from BHWC to BCHW format for model input
                image_tensor = image[i].permute(2, 0, 1).unsqueeze(0)
                # Warmup model
                if needs_warmup:
                    self._cached_config = config
                    for _ in range(model.batch_size - 1):
                        model(image=image_tensor)
                    needs_warmup = False
                    

                output = model(image=image_tensor)
                
                # Convert CHW to BHWC
                output = output.permute(1, 2, 0)
                output = output.unsqueeze(0)  # Add batch dimension
                
                outputs.append(output)
            
            # Stack outputs
            output_tensor = torch.cat(outputs, dim=0)
            
        else:
            # Text to image generation
            output = model.txt2img()
            # Convert CHW to BHWC
            output = output.permute(1, 2, 0)
            output = output.unsqueeze(0)  # Add batch dimension
            
        return (output_tensor,)

class StreamDiffusionConfigMixin:
    @staticmethod
    def get_optional_inputs():
        return {
            "opt_lora_dict": ("LORA_DICT", {"tooltip": "Optional dictionary of LoRA models to apply."}),
            "opt_advanced_config": ("ADVANCED_CONFIG", {"tooltip": "Optional advanced configuration for performance, filtering, and device settings."}),
        }

    @staticmethod
    def apply_optional_configs(config, opt_lora_dict=None, opt_advanced_config=None):
        if opt_lora_dict:
            config["lora_dict"] = opt_lora_dict

        if opt_advanced_config:
            config.update(opt_advanced_config)
        
        return config

class StreamDiffusionConfig(StreamDiffusionConfigMixin):
    @classmethod
    def INPUT_TYPES(s):
        defaults = _get_wrapper_params()
        return {
            "required": {
                "model": ("SDMODEL", {"tooltip": "The StreamDiffusion model to use for generation."}),
                "t_index_list": ("STRING", {"default": "39,35,30", "tooltip": "Comma-separated list of t_index values determining at which steps to output images."}),
                "mode": (["img2img", "txt2img"], {"default": defaults["mode"], "tooltip": "Generation mode: image-to-image or text-to-image. Note: txt2img requires cfg_type of 'none'"}),
                "width": ("INT", {"default": defaults["width"], "min": 64, "max": 2048, "tooltip": "The width of the generated images."}),
                "height": ("INT", {"default": defaults["height"], "min": 64, "max": 2048, "tooltip": "The height of the generated images."}),
                "acceleration": (["none", "xformers", "tensorrt"], {"default": "none", "tooltip": "Acceleration method to optimize performance."}),
                "frame_buffer_size": ("INT", {"default": defaults["frame_buffer_size"], "min": 1, "max": 16, "tooltip": "Size of the frame buffer for batch denoising. Increasing this can improve performance at the cost of higher memory usage."}),
                "use_tiny_vae": ("BOOLEAN", {"default": defaults["use_tiny_vae"], "tooltip": "Use a TinyVAE model for faster decoding with slight quality tradeoff."}),
                "cfg_type": (["none", "full", "self", "initialize"], {"default": defaults["cfg_type"], "tooltip": """Classifier-Free Guidance type to control how guidance is applied:
- none: No guidance, fastest but may reduce quality
- full: Full guidance on every step, highest quality but slowest
- self: Self-guidance using previous frame, good balance of speed/quality
- initialize: Only apply guidance on first frame, fast with decent quality"""}),
                "use_lcm_lora": ("BOOLEAN", {"default": True, "tooltip": "Enable use of LCM-LoRA for latent consistency."}),
                "seed": ("INT", {"default": defaults["seed"], "min": -1, "max": 100000000, "tooltip": "Seed for generation. Use -1 for random seed."}),
            },
            "optional": s.get_optional_inputs()
        }

    RETURN_TYPES = ("STREAM_MODEL",)
    OUTPUT_TOOLTIPS = ("The configured StreamDiffusion model.",)
    FUNCTION = "load_model"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = """
This configures a model for StreamDiffusion. The model can be configured with or without acceleration. With TensorRT acceleration enabled, this node will run a TensorRT engine with the supplied parameters. 
If a suitable engine does not exist, it will be created. This can be used with any given checkpoint from either StreamDiffusionCheckpointLoader or StreamDiffusionLPCheckpointLoader or StreamDiffusionLPModelLoader.
    """

    def load_model(self, model, t_index_list, mode, width, height, acceleration, 
                  frame_buffer_size, use_tiny_vae, cfg_type, use_lcm_lora, seed,
                  **optional_configs):
        
        t_index_list = [int(x.strip()) for x in t_index_list.split(",")]
        defaults = _get_wrapper_params()
        
        # Build base configuration
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
            "seed": seed,
            "engine_dir": defaults["engine_dir"],
            "output_type": defaults["output_type"]
        }

        # Apply optional configs using mixin method
        config = self.apply_optional_configs(config, **optional_configs)

        model_name = os.path.splitext(os.path.basename(model))[0] if os.path.isfile(model) else model.split('/')[-1]
        parent_name = f"{model_name}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--mode-{mode}--t_index-{len(t_index_list)}--buffer-{frame_buffer_size}"
        config["engine_dir"] = os.path.join(
            config["engine_dir"],
            parent_name
        )
        wrapper = (StreamDiffusionWrapper(**config), config)

        return (wrapper,)

#TODO: confirm this works well with container deployment
class StreamDiffusionPrebuiltConfig(StreamDiffusionConfigMixin):
    @classmethod
    def INPUT_TYPES(s):
        defaults = _get_wrapper_params()
        return {
            "required": {
                "engine_config": (get_engine_configs(), {"tooltip": "Select from available prebuilt engine configurations"}),
                "t_index_list": ("STRING", {"default": "39,35,30", "tooltip": "Comma-separated list of t_index values. Must match the number of steps used to build the engine."}),
                "mode": (["img2img", "txt2img"], {"default": defaults["mode"], "tooltip": "Generation mode: image-to-image or text-to-image"}),
                "frame_buffer_size": ("INT", {"default": defaults["frame_buffer_size"], "min": 1, "max": 16, "tooltip": "Size of the frame buffer. Must match what was used when building engines."}),
                "width": ("INT", {"default": defaults["width"], "min": 64, "max": 2048, "tooltip": "Must match the width used when building engines"}),
                "height": ("INT", {"default": defaults["height"], "min": 64, "max": 2048, "tooltip": "Must match the height used when building engines"}),
            },
            "optional": {
                "model": ("SDMODEL", ),
                **s.get_optional_inputs(),
            }
        }

    RETURN_TYPES = ("STREAM_MODEL",)
    FUNCTION = "configure_model"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Configures a model for StreamDiffusion using existing TensorRT engines."

    def configure_model(self, engine_config, t_index_list, mode, frame_buffer_size, width, height, model=None, **optional_configs):


        # Convert t_index_list from string to list of ints
        t_index_list = [int(x.strip()) for x in t_index_list.split(",")]
        
        # Get the engine directory path
        engine_dir = os.path.join(ENGINE_DIR, engine_config)
        
        defaults = _get_wrapper_params()
        
        # Build base configuration
        config = {
            "model_id_or_path": model if model is not None else engine_config,
            "mode": mode,
            "acceleration": "tensorrt",
            "frame_buffer_size": frame_buffer_size,
            "t_index_list": t_index_list,
            "width": width,
            "height": height,
            "use_denoising_batch": True,
            "use_tiny_vae": True,  # Assuming TinyVAE was used in engine building
            "output_type": defaults["output_type"]
        }

        # Apply optional configs using mixin method
        config = self.apply_optional_configs(config, **optional_configs)
        config["engine_dir"] = engine_dir
        wrapper = (StreamDiffusionWrapper(**config), config)
        return (wrapper,)

NODE_CLASS_MAPPINGS = {
    "StreamDiffusionConfig": StreamDiffusionConfig,
    "StreamDiffusionSampler": StreamDiffusionSampler,
    "StreamDiffusionPrebuiltConfig": StreamDiffusionPrebuiltConfig,
    "StreamDiffusionLoraLoader": StreamDiffusionLoraLoader,
    "StreamDiffusionAdvancedConfig": StreamDiffusionAdvancedConfig,
    "StreamDiffusionCheckpointLoader": StreamDiffusionCheckpointLoader,
    "StreamDiffusionTensorRTEngineLoader": StreamDiffusionTensorRTEngineLoader,
    "StreamDiffusionLPCheckpointLoader": StreamDiffusionLPCheckpointLoader,   
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StreamDiffusionConfig": "StreamDiffusionConfig",
    "StreamDiffusionSampler": "StreamDiffusionSampler",
    "StreamDiffusionPrebuiltConfig": "StreamDiffusionPrebuiltConfig",
    "StreamDiffusionLoraLoader": "StreamDiffusionLoraLoader",
    "StreamDiffusionAdvancedConfig": "StreamDiffusionAdvancedConfig",
    "StreamDiffusionCheckpointLoader": "StreamDiffusionCheckpointLoader",
    "StreamDiffusionTensorRTEngineLoader": "StreamDiffusionTensorRTEngineLoader",
    "StreamDiffusionLPCheckpointLoader": "StreamDiffusionLPCheckpointLoader",
}