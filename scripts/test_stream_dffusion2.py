import torch
from PIL import Image
from streamdiffusionwrapper import StreamDiffusionWrapper
import numpy as np

def test_txt2img():
    # Initialize wrapper for txt2img
    wrapper = StreamDiffusionWrapper(
        model_id_or_path="KBlueLeaf/kohaku-v2.1",
        t_index_list=[0],
        mode="txt2img",
        output_type="pil",
        device="cuda",
        dtype=torch.float16,
        frame_buffer_size=1,
        width=512,
        height=512,
        warmup=1,
        acceleration="tensorrt",
        use_lcm_lora=True,  # Enable LCM-LoRA for faster inference
        cfg_type="none",
    )

    # Test txt2img generation
    prompt = "1girl, white hair, golden eyes, detailed face, masterpiece"
    negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    
    wrapper.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=7.0
    )
    image = wrapper.txt2img()
    
    # Save the output
    if isinstance(image, list):
        for i, img in enumerate(image):
            img.save(f"kohaku_txt2img_output_{i}.png")
    else:
        image.save("kohaku_txt2img_output.png")

def test_img2img():
    # Initialize wrapper for img2img with more appropriate settings
    wrapper = StreamDiffusionWrapper(
        model_id_or_path="KBlueLeaf/kohaku-v2.1",
        t_index_list=[22, 32, 45],
        mode="img2img",
        output_type="pil",
        device="cuda",
        dtype=torch.float16,
        frame_buffer_size=1,
        width=512,
        height=512,
        warmup=10,
        acceleration="tensorrt",
        use_lcm_lora=True,
        cfg_type="self",
        do_add_noise=True,
        use_denoising_batch=True,  # Ensure this is set to True
    )

    # Load and preprocess input image
    try:
        input_image = Image.open("ryan.png").convert("RGB")
        input_image = input_image.resize((512, 512))
        input_tensor = wrapper.preprocess_image(input_image)
    except FileNotFoundError:
        print("Error: ryan.png not found in the current directory!")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    prompt = "1girl, white hair, golden eyes, detailed face, masterpiece"
    negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    
    wrapper.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=1.2,
        num_inference_steps=50,
        delta=0.5
    )
    
    print("Starting img2img generation...")
    
    # Warmup loop
    for _ in range(wrapper.batch_size - 1):
        wrapper(image=input_tensor)
    
    # Generate the final output
    output_image = wrapper(image=input_tensor)
    
    print("Generation completed")
    
    # Save output
    input_image.save("input_image.png")
    if isinstance(output_image, list):
        for i, img in enumerate(output_image):
            img.save(f"kohaku_img2img_output_{i}.png")
    else:
        output_image.save("kohaku_img2img_output.png") 

if __name__ == "__main__":
    #print("Testing txt2img with Kohaku v2.1...")
   # test_txt2img()
    #print("txt2img test completed!")

    print("\nTesting img2img with Kohaku v2.1...")
    test_img2img()
    print("img2img test completed!") 