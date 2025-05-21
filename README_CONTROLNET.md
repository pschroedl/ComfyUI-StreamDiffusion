# ControlNet Support for StreamDiffusion in ComfyUI

This extension adds ControlNet support to StreamDiffusion in ComfyUI, allowing you to guide the image generation process with conditioning images.

## Installation

1. The ControlNet support is built into the ComfyUI-StreamDiffusion extension.
2. Make sure you have the necessary dependencies installed:
   ```
   pip install diffusers>=0.26.0 controlnet-aux transformers
   ```

3. For ControlNet models, you have two options:
   - Use the built-in Hugging Face model IDs (automatically available in the dropdown)
   - Download Diffusers-format ControlNet models that include a config.json file

4. Popular Hugging Face models automatically available:
   - `lllyasviel/sd-controlnet-canny`
   - `lllyasviel/sd-controlnet-depth`
   - `lllyasviel/sd-controlnet-openpose`
   - `lllyasviel/sd-controlnet-scribble`
   - `lllyasviel/sd-controlnet-hed`
   - `lllyasviel/sd-controlnet-mlsd`
   - `lllyasviel/sd-controlnet-normal`
   - `lllyasviel/sd-controlnet-seg`
   - `lllyasviel/control_v11p_sd15_lineart`
   - `lllyasviel/control_v11p_sd15s2_lineart_anime`

5. For local models, place them in the `models/controlnet` directory. The directory structure should be:
   ```
   ComfyUI/models/controlnet/
   ├── model_name1/
   │   ├── config.json  (required)
   │   └── diffusion_pytorch_model.safetensors
   ├── model_name2/
   │   ├── config.json  (required)
   │   └── ...
   └── ...
   ```

## Important: Model Format Requirements

This implementation uses Diffusers to load ControlNet models, which requires:

1. A Hugging Face model ID
   - OR -
2. A local model directory containing both:
   - The model weights (.safetensors, .bin, etc.)
   - A config.json file describing the model architecture

**Note:** Standard ComfyUI ControlNet models (single .safetensors files without config.json) are not directly compatible. 

## Usage

### Basic Workflow

1. Load a ControlNet model using the `Load ControlNet Model` node:
   - Select a model from the dropdown (Hugging Face models or local directories with config.json)
   - Or specify a custom Hugging Face model ID in the optional field
2. Prepare your conditioning image using appropriate preprocessors
3. Configure your StreamDiffusion model with ControlNet using the `Configure ControlNet` node:
   - Adjust the conditioning scale to control the strength of the ControlNet effect
4. Run the generation as usual with the StreamDiffusion sampler

### Using Multiple ControlNets

You can chain multiple ControlNets together for combined effects:

1. Create the first ControlNet configuration:
   - Load ControlNet model → Prepare control image → Configure ControlNet

2. Chain additional ControlNets:
   - Take the output STREAM_MODEL from the first ControlNet
   - Connect it to another Configure ControlNet node with a different ControlNet model and control image
   - Each ControlNet in the chain adds its effect to the generation

3. Control options for each ControlNet:
   - Adjust individual conditioning scales to balance the influence of each ControlNet
   - Use the "clear_previous_controlnets" option (set to True) if you want to replace rather than combine

4. Examples of useful combinations:
   - Depth + Canny: Control both overall structure and edge details
   - Pose + Face: Control body position and facial features
   - Scribble + Color: Control both line drawing and colorization

### Example Workflow

1. **Load ControlNet Model**
   - Use the `Load ControlNet Model` node
   - Select a model from the dropdown

2. **Prepare Conditioning Image**
   - Load an image
   - Process it with appropriate preprocessors (ComfyUI's built-in ones work well)
   - For example, for a canny edge ControlNet, use the Canny preprocessor

3. **Configure StreamDiffusion with ControlNet**
   - Connect your StreamDiffusion model to the `Configure ControlNet` node
   - Connect your ControlNet model to the `Configure ControlNet` node
   - Connect your processed conditioning image to the `Configure ControlNet` node
   - Set the conditioning scale (0.0-5.0) to adjust the strength of the ControlNet effect:
     - Higher values (1.0-5.0): Strong conditioning, closely follows the control image
     - Medium values (0.5-1.0): Balanced conditioning, good for most use cases
     - Lower values (0.1-0.5): Subtle conditioning, gives the model more freedom
     - Zero (0.0): Effectively disables the ControlNet without having to rebuild the workflow

4. **Add Additional ControlNets (Optional)**
   - Connect the output of the first Configure ControlNet node to another Configure ControlNet node
   - Use a different ControlNet model and control image
   - Adjust the conditioning scale for this additional ControlNet
   - Repeat as needed for more ControlNets

5. **Generate Images**
   - Connect the output of the final Configure ControlNet node to the `StreamDiffusion Sampler`
   - Set your prompt and other parameters
   - Run the generation

## Performance Considerations

When using multiple ControlNets:

- Each additional ControlNet increases VRAM usage
- Processing time will increase with each added ControlNet
- For real-time applications, limit to 1-2 ControlNets with lower resolutions
- Adjust conditioning scales carefully - using too many strong ControlNets can conflict

## Troubleshooting

- **Missing config.json**: If you try to use a standard ComfyUI ControlNet model and encounter an error like "not a valid JSON file", this means the model doesn't have the required config.json file. Use one of the built-in Hugging Face models instead.

- **Memory Issues**: ControlNet models increase VRAM usage. If you encounter out-of-memory errors, try:
  - Reducing your image resolution
  - Using a smaller model
  - Using fewer steps (lower t_index_list values)
  - Reducing the number of ControlNets
  
- **Speed Issues**: ControlNet will slow down generation. For real-time applications:
  - Use a smaller ControlNet model
  - Reduce the conditioning scale
  - Consider using TensorRT acceleration
  - Use fewer ControlNets

- **Model Compatibility**: Ensure your ControlNet model is compatible with your base model. For example:
  - SD 1.5 ControlNet models work best with SD 1.5 base models
  - SD 2.0/2.1 ControlNet models work best with SD 2.0/2.1 base models

## Credits

- Based on the implementation described in [StreamDiffusion Issue #132](https://github.com/cumulo-autumn/StreamDiffusion/issues/132)
- Thanks to the StreamDiffusion team for creating the base library 