import os
import torch
import folder_paths
from diffusers.models import ControlNetModel
from .controlnet_support import ControlNetSupport

def get_controlnet_models():
    """Get list of available controlnet models from the controlnet directory"""
    controlnet_dir = os.path.join(folder_paths.models_dir, "controlnet")
    models = []
    
    # Try to create the directory if it doesn't exist
    if not os.path.exists(controlnet_dir):
        try:
            os.makedirs(controlnet_dir, exist_ok=True)
            print(f"Created controlnet directory at {controlnet_dir}")
        except Exception as e:
            print(f"Warning: Could not create controlnet directory: {e}")
            return models
    
    # Add some well-known Hugging Face models
    default_models = [
        "lllyasviel/control_v11p_sd15_canny",         
        "lllyasviel/control_v11f1p_sd15_depth",       
        "lllyasviel/control_v11p_sd15_openpose",      
        "lllyasviel/control_v11p_sd15_scribble",      
        "lllyasviel/sd-controlnet-hed",               
        "lllyasviel/control_v11p_sd15_mlsd",          
        "lllyasviel/control_v11p_sd15_normalbae",     
        "lllyasviel/control_v11p_sd15_seg",           
        "lllyasviel/control_v11p_sd15_lineart",       
        "lllyasviel/control_v11p_sd15s2_lineart_anime", 
        "monster-labs/control_v1p_sd15_qrcode_monster", 
        "monster-labs/control_v1p_sd15_qrcode_monster/v2",  # QR code model v2 (in v2 subfolder)
        "lllyasviel/control_v11p_sd15_inpaint",       
        "lllyasviel/control_v11e_sd15_shuffle",       
        "lllyasviel/control_v11e_sd15_ip2p",          
        "lllyasviel/control_v11f1e_sd15_tile"         
    ]
    models.extend(default_models)
    
    # List valid Diffusers model directories (containing config.json)
    if os.path.exists(controlnet_dir):
        for item in os.listdir(controlnet_dir):
            item_path = os.path.join(controlnet_dir, item)
            if os.path.isdir(item_path):
                # Check if directory contains config.json (required for Diffusers)
                config_path = os.path.join(item_path, "config.json")
                if os.path.exists(config_path):
                    models.append(item)
    
    return models

class StreamDiffusionControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "controlnet_name": (get_controlnet_models(), {
                    "tooltip": "Select a ControlNet model from your controlnet directory"
                }),
            },
            "optional": {
                "controlnet_path": ("STRING", {"default": "", "multiline": False, 
                                              "tooltip": "Optional: Path to a ControlNet model directory or safetensors file (overrides selection)"}),
            }
        }

    RETURN_TYPES = ("CONTROLNET",)
    OUTPUT_TOOLTIPS = ("The loaded ControlNet model configuration.",)
    FUNCTION = "load_controlnet"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Loads a ControlNet model for use with StreamDiffusion"

    def load_controlnet(self, controlnet_name, controlnet_path=""):
        """
        Load a ControlNet model from the selected name or specified path
        
        Parameters
        ----------
        controlnet_name : str
            Name of the ControlNet model to load from the controlnet directory
        controlnet_path : str
            Optional path to override the selection from the dropdown
            
        Returns
        -------
        CONTROLNET
            The loaded ControlNet model configuration
        """
        # Use the custom path if provided, otherwise use the selected model
        path_to_use = controlnet_path if controlnet_path else controlnet_name
        
        try:
            # Handle Hugging Face model IDs (they contain a slash)
            if "/" in path_to_use:
                print(f"Loading ControlNet from Hugging Face: {path_to_use}")
                # Check if this is a model with a subfolder path like "monster-labs/control_v1p_sd15_qrcode_monster/v2"
                if path_to_use.count("/") > 1:
                    # Split into repo_id and subfolder
                    parts = path_to_use.split("/")
                    repo_id = "/".join(parts[:2])  # e.g., "monster-labs/control_v1p_sd15_qrcode_monster"
                    subfolder = "/".join(parts[2:])  # e.g., "v2"
                    print(f"Loading from Hugging Face with subfolder: repo_id={repo_id}, subfolder={subfolder}")
                    controlnet = ControlNetModel.from_pretrained(
                        repo_id,
                        subfolder=subfolder,
                        torch_dtype=torch.float16
                    ).to("cuda")
                else:
                    # Normal Hugging Face model ID
                    controlnet = ControlNetModel.from_pretrained(
                        path_to_use, 
                        torch_dtype=torch.float16
                    ).to("cuda")
                return ((controlnet,),)
            
            # Handle local model directory with config.json
            controlnet_dir = os.path.join(folder_paths.models_dir, "controlnet")
            full_path = os.path.join(controlnet_dir, path_to_use)
            
            if os.path.exists(full_path) and os.path.isdir(full_path):
                config_path = os.path.join(full_path, "config.json")
                if os.path.exists(config_path):
                    print(f"Loading ControlNet from local directory: {full_path}")
                    controlnet = ControlNetModel.from_pretrained(
                        full_path, 
                        torch_dtype=torch.float16
                    ).to("cuda")
                    return ((controlnet,),)
            
            # If a direct path was provided and exists
            if os.path.exists(path_to_use):
                if os.path.isdir(path_to_use):
                    config_path = os.path.join(path_to_use, "config.json")
                    if os.path.exists(config_path):
                        print(f"Loading ControlNet from path: {path_to_use}")
                        controlnet = ControlNetModel.from_pretrained(
                            path_to_use, 
                            torch_dtype=torch.float16
                        ).to("cuda")
                        return ((controlnet,),)
            
            # If we get here, try as a Hugging Face model ID
            print(f"Path not found, attempting to load as Hugging Face model ID: {path_to_use}")
            controlnet = ControlNetModel.from_pretrained(
                path_to_use, 
                torch_dtype=torch.float16
            ).to("cuda")
            return ((controlnet,),)
            
        except Exception as e:
            print(f"Error loading ControlNet model: {e}")
            raise ValueError(f"Failed to load ControlNet model '{path_to_use}': {e}")

class StreamDiffusionControlNetPreprocessorBase: 
    """Base class for ControlNet preprocessors"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The preprocessed image suitable for ControlNet input.",)
    FUNCTION = "preprocess"
    CATEGORY = "StreamDiffusion/ControlNet"
    
    def preprocess(self, image):
        """
        Process an image for ControlNet input
        
        Parameters
        ----------
        image : torch.Tensor
            Input image tensor [B, H, W, C]
            
        Returns
        -------
        torch.Tensor
            Processed image tensor suitable for ControlNet input
        """
        # Base class just returns the input image
        # Child classes should implement their own preprocessing logic
        return (image,)

class StreamDiffusionControlNetConfigNode:
    """Node to apply ControlNet configuration to a StreamDiffusion model"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stream_model": ("STREAM_MODEL",),
                "controlnet": ("CONTROLNET",),
                "control_image": ("IMAGE",),
                "conditioning_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01,
                                               "tooltip": "Controls the strength of the ControlNet conditioning"}),
            },
            "optional": {
                "clear_previous_controlnets": ("BOOLEAN", {"default": False, 
                                                          "tooltip": "When enabled, removes all previous ControlNets before adding this one"}),
            }
        }
    
    RETURN_TYPES = ("STREAM_MODEL",)
    OUTPUT_TOOLTIPS = ("The StreamDiffusion model configured with ControlNet.",)
    FUNCTION = "configure_controlnet"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Configures a StreamDiffusion model with a ControlNet. Multiple nodes can be chained to add several ControlNets."
    
    def configure_controlnet(self, stream_model, controlnet, control_image, conditioning_scale=1.0, clear_previous_controlnets=False):
        """
        Apply ControlNet configuration to a StreamDiffusion model
        
        Parameters
        ----------
        stream_model : tuple
            The StreamDiffusion model (wrapper, config)
        controlnet : tuple
            The ControlNet model (without conditioning scale)
        control_image : torch.Tensor
            The conditioning image for ControlNet
        conditioning_scale : float
            Strength of the ControlNet conditioning
        clear_previous_controlnets : bool
            When True, removes all previously added ControlNets
            
        Returns
        -------
        tuple
            The updated StreamDiffusion model with ControlNet configured
        """
        # Unpack inputs
        wrapper, config = stream_model
        controlnet_model = controlnet[0]
        
        # Clear previous ControlNets if requested
        if clear_previous_controlnets:
            wrapper.clear_controlnets()
        
        # Prepare the ControlNet image
        # Handle batched control images properly
        if control_image is not None:
            # Get batch size
            batch_size = control_image.shape[0]
            
            if batch_size > 1:
                # For batched processing, we'll prepare all images but only add the first one initially
                # The wrapper will handle the batch during inference
                processed_control_images = []
                
                for i in range(batch_size):
                    # Convert from BHWC to BCHW format for model input
                    img = control_image[i].permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                    # Convert to the right device and dtype
                    img = img.to(device=wrapper.device, dtype=wrapper.dtype)
                    processed_control_images.append(img)
                
                # Store all processed images in the wrapper for batch processing
                wrapper.add_controlnet(
                    controlnet_model, 
                    processed_control_images[0],  # Initially set first image
                    conditioning_scale
                )
                
                # Store the batch of control images for later use during inference
                # Get the index of the controlnet we just added
                controlnet_idx = len(wrapper.controlnets) - 1
                # Store the batch for later use
                wrapper.controlnet_batch_images = getattr(wrapper, 'controlnet_batch_images', {})
                wrapper.controlnet_batch_images[controlnet_idx] = processed_control_images
            else:
                # For single image, process normally
                control_img = control_image[0].permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                control_img = control_img.to(device=wrapper.device, dtype=wrapper.dtype)
                wrapper.add_controlnet(controlnet_model, control_img, conditioning_scale)
        
        # Return the updated model
        return ((wrapper, config),)

# Node list for registration
NODE_CLASS_MAPPINGS = {
    "StreamDiffusionControlNetLoader": StreamDiffusionControlNetLoader,
    "StreamDiffusionControlNetConfigNode": StreamDiffusionControlNetConfigNode,
    "StreamDiffusionControlNetPreprocessorBase": StreamDiffusionControlNetPreprocessorBase,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StreamDiffusionControlNetLoader": "Load ControlNet Model",
    "StreamDiffusionControlNetConfigNode": "Configure ControlNet",
    "StreamDiffusionControlNetPreprocessorBase": "ControlNet Base Preprocessor",
} 