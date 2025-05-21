# Copied from StreamDiffusion/utils/wrapper.py

import gc
import os
import traceback
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union, Tuple

import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from PIL import Image
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

# Import for ControlNet support
from diffusers.models import ControlNetModel

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class StreamDiffusionWrapper:
    def __init__(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        mode: Literal["img2img", "txt2img"] = "img2img",
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        warmup: int = 10,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        do_add_noise: bool = True,
        device_ids: Optional[List[int]] = None,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        use_denoising_batch: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        use_safety_checker: bool = False,
        engine_dir: Optional[Union[str, Path]] = "engines",
        # Added parameters for ControlNet
        controlnet: Optional[ControlNetModel] = None,
        controlnet_conditioning_scale: float = 1.0,
    ):
        """
        Initializes the StreamDiffusionWrapper.

        Parameters
        ----------
        model_id_or_path : str
            The model id or path to load.
        t_index_list : List[int]
            The t_index_list to use for inference.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        mode : Literal["img2img", "txt2img"], optional
            txt2img or img2img, by default "img2img".
        output_type : Literal["pil", "pt", "np", "latent"], optional
            The output type of image, by default "pil".
        lcm_lora_id : Optional[str], optional
            The lcm_lora_id to load, by default None.
            If None, the default LCM-LoRA
            ("latent-consistency/lcm-lora-sdv1-5") will be used.
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
            If None, the default TinyVAE
            ("madebyollin/taesd") will be used.
        device : Literal["cpu", "cuda"], optional
            The device to use for inference, by default "cuda".
        dtype : torch.dtype, optional
            The dtype for inference, by default torch.float16.
        frame_buffer_size : int, optional
            The frame buffer size for denoising batch, by default 1.
        width : int, optional
            The width of the image, by default 512.
        height : int, optional
            The height of the image, by default 512.
        warmup : int, optional
            The number of warmup steps to perform, by default 10.
        acceleration : Literal["none", "xformers", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        device_ids : Optional[List[int]], optional
            The device ids to use for DataParallel, by default None.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        enable_similar_image_filter : bool, optional
            Whether to enable similar image filter or not,
            by default False.
        similar_image_filter_threshold : float, optional
            The threshold for similar image filter, by default 0.98.
        similar_image_filter_max_skip_frame : int, optional
            The max skip frame for similar image filter, by default 10.
        use_denoising_batch : bool, optional
            Whether to use denoising batch or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.
        use_safety_checker : bool, optional
            Whether to use safety checker or not, by default False.
        controlnet : Optional[ControlNetModel], optional
            The ControlNet model to use, by default None.
        controlnet_conditioning_scale : float, optional
            The scale for the ControlNet output, by default 1.0.
        """
        if acceleration == "tensorrt":
            import streamdiffusion._hf_tracing_patches as _tp
            _tp.apply_all_patches()

        self.sd_turbo = "turbo" in model_id_or_path

        if mode == "txt2img":
            if cfg_type != "none":
                raise ValueError(
                    f"txt2img mode accepts only cfg_type = 'none', but got {cfg_type}"
                )
            if use_denoising_batch and frame_buffer_size > 1:
                if not self.sd_turbo:
                    raise ValueError(
                        "txt2img mode cannot use denoising batch with frame_buffer_size > 1."
                    )

        if mode == "img2img":
            if not use_denoising_batch:
                raise NotImplementedError(
                    "img2img mode must use denoising batch for now."
                )

        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.mode = mode
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size
        self.batch_size = (
            len(t_index_list) * frame_buffer_size
            if use_denoising_batch
            else frame_buffer_size
        )

        self.use_denoising_batch = use_denoising_batch
        self.use_safety_checker = use_safety_checker
        
        # Initialize lists to store multiple ControlNets
        self.controlnets = []
        self.controlnet_images = []
        self.controlnet_scales = []
        
        # Store legacy single ControlNet (if provided) for backwards compatibility
        if controlnet is not None:
            self.add_controlnet(controlnet, None, controlnet_conditioning_scale)
        
        # Store ControlNet model and parameters
        self.controlnet = controlnet
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.controlnet_image = None

        self.stream: StreamDiffusion = self._load_model(
            model_id_or_path=model_id_or_path,
            lora_dict=lora_dict,
            lcm_lora_id=lcm_lora_id,
            vae_id=vae_id,
            t_index_list=t_index_list,
            acceleration=acceleration,
            warmup=warmup,
            do_add_noise=do_add_noise,
            use_lcm_lora=use_lcm_lora,
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            seed=seed,
            engine_dir=engine_dir,
        )

        if device_ids is not None:
            self.stream.unet = torch.nn.DataParallel(
                self.stream.unet, device_ids=device_ids
            )

        if enable_similar_image_filter:
            self.stream.enable_similar_image_filter(
                similar_image_filter_threshold, similar_image_filter_max_skip_frame
            )
    
    def add_controlnet(self, controlnet, control_image=None, scale=1.0):
        """
        Add a ControlNet model to the pipeline
        
        Parameters
        ----------
        controlnet : ControlNetModel
            The ControlNet model to add
        control_image : torch.Tensor, optional
            The control image tensor, by default None
        scale : float, optional
            The conditioning scale, by default 1.0
        """
        self.controlnets.append(controlnet)
        self.controlnet_images.append(control_image)  # Can be None initially
        self.controlnet_scales.append(scale)
    
    def clear_controlnets(self):
        """Remove all ControlNet models from the pipeline"""
        self.controlnets = []
        self.controlnet_images = []
        self.controlnet_scales = []
        
    def update_controlnet_image_for_batch(self, batch_index):
        """
        Update ControlNet images for a specific batch index
        
        Parameters
        ----------
        batch_index : int
            The index in the batch to use for ControlNet images
            
        Returns
        -------
        bool
            True if any ControlNet images were updated, False otherwise
        """
        # Check if we have batch images stored
        if not hasattr(self, 'controlnet_batch_images') or not self.controlnet_batch_images:
            return False
            
        updated = False
        # For each ControlNet that has batch images stored
        for controlnet_idx, batch_images in self.controlnet_batch_images.items():
            if controlnet_idx < len(self.controlnet_images) and batch_index < len(batch_images):
                # Update the image for this ControlNet to the one for the current batch index
                self.controlnet_images[controlnet_idx] = batch_images[batch_index]
                updated = True
                
        return updated

    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        controlnet_image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
    ) -> None:
        """
        Prepares the model for inference.

        Parameters
        ----------
        prompt : str
            The prompt to generate images from.
        num_inference_steps : int, optional
            The number of inference steps to perform, by default 50.
        guidance_scale : float, optional
            The guidance scale to use, by default 1.2.
        delta : float, optional
            The delta multiplier of virtual residual noise,
            by default 1.0.
        controlnet_image : Optional[Union[str, Image.Image, torch.Tensor]], optional
            The conditioning image for ControlNet, by default None.
        controlnet_image : Optional[Union[str, Image.Image, torch.Tensor]], optional
            The conditioning image for ControlNet, by default None.
        """
        self.stream.prepare(
            prompt,
            negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
        )
        
        # Process and store the controlnet image if provided (legacy support)
        if controlnet_image is not None and len(self.controlnets) > 0:
            if isinstance(controlnet_image, str):
                controlnet_image = Image.open(controlnet_image).convert("RGB").resize((self.width, self.height))
            if isinstance(controlnet_image, Image.Image):
                controlnet_image = controlnet_image.convert("RGB").resize((self.width, self.height))
                
            # Store the conditioning image in the first slot (for legacy support)
            self.controlnet_images[0] = self.prepare_controlnet_image(controlnet_image)

    def prepare_controlnet_image(self, image):
        """
        Prepare an image for ControlNet input
        
        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The input control image
            
        Returns
        -------
        torch.Tensor
            Processed image tensor ready for ControlNet
        """
        # This function would depend on the specific ControlNet implementation
        # For now, we'll use a simplified version assuming the image is already preprocessed
        if isinstance(image, torch.Tensor):
            return image.to(device=self.device, dtype=self.dtype)
        
        # For PIL images or file paths, preprocess appropriately
        # You may need a more complex preprocessing pipeline depending on the ControlNet
        from diffusers.utils import load_image
        if isinstance(image, str):
            image = load_image(image)
        
        # Convert to tensor and return
        image = self.stream.pipe.image_processor.preprocess(image).to(device=self.device, dtype=self.dtype)
        return image

    def get_controlnet_outputs(self, x_t_latent, t_list, prompt_embeds):
        """
        Process input through all ControlNets to get combined conditioning
        
        Parameters
        ----------
        x_t_latent : torch.Tensor
            Input latent tensor
        t_list : torch.Tensor
            Timestep tensor
        prompt_embeds : torch.Tensor
            Text embeddings
            
        Returns
        -------
        Tuple[List[torch.Tensor], torch.Tensor]
            Combined outputs from all ControlNets:
            - down_block_res_samples: List of tensors for each resolution level
            - mid_block_res_sample: Tensor for mid-block
        """
        if not self.controlnets or not any(img is not None for img in self.controlnet_images):
            return None, None
            
        # Initialize with empty tensors or None
        down_block_res_samples = None
        mid_block_res_sample = None
        
        # Process each ControlNet
        for i, (controlnet, control_image, scale) in enumerate(
            zip(self.controlnets, self.controlnet_images, self.controlnet_scales)
        ):
            if controlnet is None or control_image is None or scale == 0:
                continue
                
            # Forward pass through ControlNet
            down_samples, mid_sample = controlnet(
                x_t_latent,
                t_list,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=control_image,
                conditioning_scale=scale,
                return_dict=False,
            )
            
            # Combine outputs
            if down_block_res_samples is None:
                down_block_res_samples = down_samples
                mid_block_res_sample = mid_sample
            else:
                # Add contributions from this ControlNet
                for j in range(len(down_block_res_samples)):
                    down_block_res_samples[j] += down_samples[j]
                mid_block_res_sample += mid_sample
        
        return down_block_res_samples, mid_block_res_sample

    def __call__(
        self,
        image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
        prompt: Optional[str] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Performs img2img or txt2img based on the mode.

        Parameters
        ----------
        image : Optional[Union[str, Image.Image, torch.Tensor]]
            The image to generate from.
        prompt : Optional[str]
            The prompt to generate images from.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if self.mode == "img2img":
            return self.img2img(image, prompt)
        else:
            return self.txt2img(prompt)

    def txt2img(
        self, prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs txt2img.

        Parameters
        ----------
        prompt : Optional[str]
            The prompt to generate images from.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        # Modify StreamDiffusion's txt2img method to use ControlNet if available
        if self.controlnets and any(img is not None for img in self.controlnet_images):
            # Monkey patch the unet_step method to use ControlNet
            original_unet_step = self.stream.unet_step
            
            def patched_unet_step(x_t_latent, t_list, idx=None):
                if self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "initialize"):
                    x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
                    t_list_expanded = torch.concat([t_list[0:1], t_list], dim=0)
                elif self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "full"):
                    x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
                    t_list_expanded = torch.concat([t_list, t_list], dim=0)
                else:
                    x_t_latent_plus_uc = x_t_latent
                    t_list_expanded = t_list
                
                # Get combined ControlNet outputs
                down_block_res_samples, mid_block_res_sample = self.get_controlnet_outputs(
                    x_t_latent_plus_uc, t_list_expanded, self.stream.prompt_embeds
                )
                
                # Call UNet with ControlNet outputs
                model_pred = self.stream.unet(
                    x_t_latent_plus_uc,
                    t_list_expanded,
                    encoder_hidden_states=self.stream.prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]
                
                # Continue with the original logic
                if self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "initialize"):
                    noise_pred_text = model_pred[1:]
                    self.stream.stock_noise = torch.concat(
                        [model_pred[0:1], self.stream.stock_noise[1:]], dim=0
                    )
                elif self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "full"):
                    noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
                else:
                    noise_pred_text = model_pred
                
                if self.stream.guidance_scale > 1.0 and (
                    self.stream.cfg_type == "self" or self.stream.cfg_type == "initialize"
                ):
                    noise_pred_uncond = self.stream.stock_noise * self.stream.delta
                
                if self.stream.guidance_scale > 1.0 and self.stream.cfg_type != "none":
                    model_pred = noise_pred_uncond + self.stream.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                else:
                    model_pred = noise_pred_text
                
                # Compute the previous noisy sample x_t -> x_t-1
                if self.stream.use_denoising_batch:
                    denoised_batch = self.stream.scheduler_step_batch(model_pred, x_t_latent, idx)
                    if self.stream.cfg_type == "self" or self.stream.cfg_type == "initialize":
                        scaled_noise = self.stream.beta_prod_t_sqrt * self.stream.stock_noise
                        delta_x = self.stream.scheduler_step_batch(model_pred, scaled_noise, idx)
                        alpha_next = torch.concat(
                            [
                                self.stream.alpha_prod_t_sqrt[1:],
                                torch.ones_like(self.stream.alpha_prod_t_sqrt[0:1]),
                            ],
                            dim=0,
                        )
                        delta_x = alpha_next * delta_x
                        beta_next = torch.concat(
                            [
                                self.stream.beta_prod_t_sqrt[1:],
                                torch.ones_like(self.stream.beta_prod_t_sqrt[0:1]),
                            ],
                            dim=0,
                        )
                        delta_x = delta_x / beta_next
                        init_noise = torch.concat(
                            [self.stream.init_noise[1:], self.stream.init_noise[0:1]], dim=0
                        )
                        self.stream.stock_noise = init_noise + delta_x
                else:
                    denoised_batch = self.stream.scheduler_step_batch(model_pred, x_t_latent, idx)
                
                return denoised_batch, model_pred
            
            # Replace with patched method
            self.stream.unet_step = patched_unet_step
            
            try:
                if self.sd_turbo:
                    image_tensor = self.stream.txt2img_sd_turbo(self.batch_size)
                else:
                    image_tensor = self.stream.txt2img(self.frame_buffer_size)
            finally:
                # Restore original method
                self.stream.unet_step = original_unet_step
        else:
            # Use original txt2img if no ControlNet
            if self.sd_turbo:
                image_tensor = self.stream.txt2img_sd_turbo(self.batch_size)
            else:
                image_tensor = self.stream.txt2img(self.frame_buffer_size)
                
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image

    def img2img(
        self, image: Union[str, Image.Image, torch.Tensor], prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs img2img.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to generate from.

        Returns
        -------
        Image.Image
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if isinstance(image, str) or isinstance(image, Image.Image):
            image = self.preprocess_image(image)

        # Monkey patch the unet_step method if ControlNet is available
        if self.controlnets and any(img is not None for img in self.controlnet_images):
            original_unet_step = self.stream.unet_step
            
            def patched_unet_step(x_t_latent, t_list, idx=None):
                if self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "initialize"):
                    x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
                    t_list_expanded = torch.concat([t_list[0:1], t_list], dim=0)
                elif self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "full"):
                    x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
                    t_list_expanded = torch.concat([t_list, t_list], dim=0)
                else:
                    x_t_latent_plus_uc = x_t_latent
                    t_list_expanded = t_list
                
                # Get combined ControlNet outputs
                down_block_res_samples, mid_block_res_sample = self.get_controlnet_outputs(
                    x_t_latent_plus_uc, t_list_expanded, self.stream.prompt_embeds
                )
                
                # Call UNet with ControlNet outputs
                model_pred = self.stream.unet(
                    x_t_latent_plus_uc,
                    t_list_expanded,
                    encoder_hidden_states=self.stream.prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]
                
                # Continue with the original logic
                if self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "initialize"):
                    noise_pred_text = model_pred[1:]
                    self.stream.stock_noise = torch.concat(
                        [model_pred[0:1], self.stream.stock_noise[1:]], dim=0
                    )
                elif self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "full"):
                    noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
                else:
                    noise_pred_text = model_pred
                
                if self.stream.guidance_scale > 1.0 and (
                    self.stream.cfg_type == "self" or self.stream.cfg_type == "initialize"
                ):
                    noise_pred_uncond = self.stream.stock_noise * self.stream.delta
                
                if self.stream.guidance_scale > 1.0 and self.stream.cfg_type != "none":
                    model_pred = noise_pred_uncond + self.stream.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                else:
                    model_pred = noise_pred_text
                
                # Compute the previous noisy sample x_t -> x_t-1
                if self.stream.use_denoising_batch:
                    denoised_batch = self.stream.scheduler_step_batch(model_pred, x_t_latent, idx)
                    if self.stream.cfg_type == "self" or self.stream.cfg_type == "initialize":
                        scaled_noise = self.stream.beta_prod_t_sqrt * self.stream.stock_noise
                        delta_x = self.stream.scheduler_step_batch(model_pred, scaled_noise, idx)
                        alpha_next = torch.concat(
                            [
                                self.stream.alpha_prod_t_sqrt[1:],
                                torch.ones_like(self.stream.alpha_prod_t_sqrt[0:1]),
                            ],
                            dim=0,
                        )
                        delta_x = alpha_next * delta_x
                        beta_next = torch.concat(
                            [
                                self.stream.beta_prod_t_sqrt[1:],
                                torch.ones_like(self.stream.beta_prod_t_sqrt[0:1]),
                            ],
                            dim=0,
                        )
                        delta_x = delta_x / beta_next
                        init_noise = torch.concat(
                            [self.stream.init_noise[1:], self.stream.init_noise[0:1]], dim=0
                        )
                        self.stream.stock_noise = init_noise + delta_x
                else:
                    denoised_batch = self.stream.scheduler_step_batch(model_pred, x_t_latent, idx)
                
                return denoised_batch, model_pred
            
            # Replace with patched method
            self.stream.unet_step = patched_unet_step
            
            try:
                image_tensor = self.stream(image)
            finally:
                # Restore original method
                self.stream.unet_step = original_unet_step
        else:
            # Use original img2img if no ControlNet
            image_tensor = self.stream(image)
            
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image

    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocesses the image.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to preprocess.

        Returns
        -------
        torch.Tensor
            The preprocessed image.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB").resize((self.width, self.height))
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((self.width, self.height))

        return self.stream.image_processor.preprocess(
            image, self.height, self.width
        ).to(device=self.device, dtype=self.dtype)

    def postprocess_image(
        self, image_tensor: torch.Tensor, output_type: str = "pil"
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Postprocesses the image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The image tensor to postprocess.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The postprocessed image.
        """
        if self.frame_buffer_size > 1:
            return postprocess_image(image_tensor.cpu(), output_type=output_type)
        else:
            return postprocess_image(image_tensor.cpu(), output_type=output_type)[0]

    def _load_model(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        warmup: int = 10,
        do_add_noise: bool = True,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        engine_dir: Optional[Union[str, Path]] = "engines",
    ) -> StreamDiffusion:
        """
        Loads the model.

        This method does the following:

        1. Loads the model from the model_id_or_path.
        2. Loads and fuses the LCM-LoRA model from the lcm_lora_id if needed.
        3. Loads the VAE model from the vae_id if needed.
        4. Enables acceleration if needed.
        5. Prepares the model for inference.
        6. Load the safety checker if needed.

        Parameters
        ----------
        model_id_or_path : str
            The model id or path to load.
        t_index_list : List[int]
            The t_index_list to use for inference.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        lcm_lora_id : Optional[str], optional
            The lcm_lora_id to load, by default None.
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
        acceleration : Literal["none", "xfomers", "sfast", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        warmup : int, optional
            The number of warmup steps to perform, by default 10.
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.

        Returns
        -------
        StreamDiffusion
            The loaded model.
        """

        try:  # Load from local directory
            pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
                model_id_or_path,
            ).to(device=self.device, dtype=self.dtype)

        except ValueError:  # Load from huggingface
            pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file(
                model_id_or_path,
            ).to(device=self.device, dtype=self.dtype)
        except Exception:  # No model found
            traceback.print_exc()
            print("Model load has failed. Doesn't exist.")
            exit()

        stream = StreamDiffusion(
            pipe=pipe,
            t_index_list=t_index_list,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            do_add_noise=do_add_noise,
            frame_buffer_size=self.frame_buffer_size,
            use_denoising_batch=self.use_denoising_batch,
            cfg_type=cfg_type,
        )
        if not self.sd_turbo:
            if use_lcm_lora:
                if lcm_lora_id is not None:
                    stream.load_lcm_lora(
                        pretrained_model_name_or_path_or_dict=lcm_lora_id
                    )
                else:
                    stream.load_lcm_lora()
                stream.fuse_lora()

            if lora_dict is not None:
                for lora_name, lora_scale in lora_dict.items():
                    stream.load_lora(lora_name)
                    stream.fuse_lora(lora_scale=lora_scale)
                    print(f"Use LoRA: {lora_name} in weights {lora_scale}")

        if use_tiny_vae:
            if vae_id is not None:
                stream.vae = AutoencoderTiny.from_pretrained(vae_id).to(
                    device=pipe.device, dtype=pipe.dtype
                )
            else:
                stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
                    device=pipe.device, dtype=pipe.dtype
                )

        try:
            if acceleration == "xformers":
                stream.pipe.enable_xformers_memory_efficient_attention()
            if acceleration == "tensorrt":
                from polygraphy import cuda
                from streamdiffusion.acceleration.tensorrt import (
                    TorchVAEEncoder,
                    compile_unet,
                    compile_vae_decoder,
                    compile_vae_encoder,
                )
                from streamdiffusion.acceleration.tensorrt.engine import (
                    AutoencoderKLEngine,
                    UNet2DConditionModelEngine,
                )
                from streamdiffusion.acceleration.tensorrt.models import (
                    VAE,
                    UNet,
                    VAEEncoder,
                )

                def create_prefix(
                    model_id_or_path: str,
                    max_batch_size: int,
                    min_batch_size: int,
                ):
                    maybe_path = Path(model_id_or_path)
                    if maybe_path.exists():
                        return f"{maybe_path.stem}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}--mode-{self.mode}"
                    else:
                        return f"{model_id_or_path}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}--mode-{self.mode}"

                engine_dir = Path(engine_dir)
                unet_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                    ),
                    "unet.engine",
                )
                vae_encoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    ),
                    "vae_encoder.engine",
                )
                vae_decoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    ),
                    "vae_decoder.engine",
                )

                if not os.path.exists(unet_path):
                    os.makedirs(os.path.dirname(unet_path), exist_ok=True)
                    unet_model = UNet(
                        fp16=True,
                        device=stream.device,
                        max_batch_size=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                        embedding_dim=stream.text_encoder.config.hidden_size,
                        unet_dim=stream.unet.config.in_channels,
                    )
                    compile_unet(
                        stream.unet,
                        unet_model,
                        unet_path + ".onnx",
                        unet_path + ".opt.onnx",
                        unet_path,
                        opt_batch_size=stream.trt_unet_batch_size,
                    )

                if not os.path.exists(vae_decoder_path):
                    os.makedirs(os.path.dirname(vae_decoder_path), exist_ok=True)
                    stream.vae.forward = stream.vae.decode
                    vae_decoder_model = VAE(
                        device=stream.device,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    compile_vae_decoder(
                        stream.vae,
                        vae_decoder_model,
                        vae_decoder_path + ".onnx",
                        vae_decoder_path + ".opt.onnx",
                        vae_decoder_path,
                        opt_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    delattr(stream.vae, "forward")

                if not os.path.exists(vae_encoder_path):
                    os.makedirs(os.path.dirname(vae_encoder_path), exist_ok=True)
                    vae_encoder = TorchVAEEncoder(stream.vae).to(torch.device("cuda"))
                    vae_encoder_model = VAEEncoder(
                        device=stream.device,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    compile_vae_encoder(
                        vae_encoder,
                        vae_encoder_model,
                        vae_encoder_path + ".onnx",
                        vae_encoder_path + ".opt.onnx",
                        vae_encoder_path,
                        opt_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )

                cuda_stream = cuda.Stream()

                vae_config = stream.vae.config
                vae_dtype = stream.vae.dtype

                stream.unet = UNet2DConditionModelEngine(
                    unet_path, cuda_stream, use_cuda_graph=False
                )
                stream.vae = AutoencoderKLEngine(
                    vae_encoder_path,
                    vae_decoder_path,
                    cuda_stream,
                    stream.pipe.vae_scale_factor,
                    use_cuda_graph=False,
                )
                setattr(stream.vae, "config", vae_config)
                setattr(stream.vae, "dtype", vae_dtype)

                gc.collect()
                torch.cuda.empty_cache()

                print("TensorRT acceleration enabled.")
            if acceleration == "sfast":
                from streamdiffusion.acceleration.sfast import (
                    accelerate_with_stable_fast,
                )

                stream = accelerate_with_stable_fast(stream)
                print("StableFast acceleration enabled.")
        except Exception:
            traceback.print_exc()
            print("Acceleration has failed. Falling back to normal mode.")

        if seed < 0:  # Random seed
            seed = np.random.randint(0, 1000000)

        stream.prepare(
            "",
            "",
            num_inference_steps=50,
            guidance_scale=1.1
            if stream.cfg_type in ["full", "self", "initialize"]
            else 1.0,
            generator=torch.manual_seed(seed),
            seed=seed,
        )

        if self.use_safety_checker:
            from diffusers.pipelines.stable_diffusion.safety_checker import (
                StableDiffusionSafetyChecker,
            )
            from transformers import CLIPFeatureExtractor

            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(pipe.device)
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.nsfw_fallback_img = Image.new("RGB", (512, 512), (0, 0, 0))

        return stream
