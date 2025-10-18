import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Diffusers version check passed.")
