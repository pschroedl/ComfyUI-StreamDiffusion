from setuptools import setup

setup(
    name="ComfyUI-StreamDiffusion",
    version="0.1.0",
    description="ComfyUI-StreamDiffusion project dependencies",
    install_requires=[
        "torch>=2.3.0",
        "torchvision>=0.16.0",
        "xformers",
        "huggingface-hub>=0.25.0",
        "diffusers>=0.31.0",
        "protobuf>=4.25.3",
    ],
    python_requires=">=3.10",
)
