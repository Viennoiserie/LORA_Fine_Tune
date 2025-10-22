import os
import torch
import shutil
import inspect
import subprocess

from huggingface_hub import snapshot_download
from diffusers import StableDiffusionXLPipeline

# region : 1 - Prepare folders 

CHECKPOINT_PATH = r"C:\Users\thoma\Documents\Thomas - SSD\LORA_Fine_Tune\Models\Safetensors\pony_V3.safetensors"
OUTPUT_DIR = r"C:\Users\thoma\Documents\Thomas - SSD\LORA_Fine_Tune\Models\Diffusers\Pony_V3"

LOCAL_BASE_DIR = r"C:\Users\thoma\Documents\Thomas - SSD\LORA_Fine_Tune\Models\SDXL"
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "cuda"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOCAL_BASE_DIR, exist_ok=True)

print("Starting SafeTensors → Diffusers conversion...")
print("------------------------------------------------")

# endregion

# region : 2 - Ensure the SDXL base model is available

if not any(os.scandir(LOCAL_BASE_DIR)):

    print(f"Base model not found locally. Downloading {BASE_MODEL}...")

    base_model_dir = snapshot_download(repo_id=BASE_MODEL,
                                       local_dir=LOCAL_BASE_DIR,
                                       local_dir_use_symlinks=False,
                                       revision="main")
    
    print(f"Downloaded SDXL base model to: {base_model_dir}")

else:
    base_model_dir = LOCAL_BASE_DIR
    print(f"Using existing SDXL base model at: {base_model_dir}")

# endregion

# region : 3 - Ensure diffusers converter is available

try:
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

except ImportError:

    print("Installing/updating diffusers...")
    subprocess.run(["pip", "install", "-U", "diffusers", "transformers", "accelerate", "safetensors"], check=True)
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

# endregion

# region : 4 - Convert the model checkpoint

print("\nConverting your .safetensors checkpoint to Diffusers format...")

sig = inspect.signature(download_from_original_stable_diffusion_ckpt)
arg_names = list(sig.parameters.keys())

possible_kwargs = {"original_config_file": None,   
                   "extract_ema": True,
                   "from_safetensors": True,
                   "to_safetensors": True,  
                   "image_size": 1024}

kwargs = {k: v for k, v in possible_kwargs.items() if k in arg_names}

if "checkpoint_path" in arg_names:
    kwargs["checkpoint_path"] = CHECKPOINT_PATH

elif "ckpt_path" in arg_names:
    kwargs["ckpt_path"] = CHECKPOINT_PATH

elif "original_ckpt_path" in arg_names:
    kwargs["original_ckpt_path"] = CHECKPOINT_PATH

elif "checkpoint_path_or_dict" in arg_names:
    kwargs["checkpoint_path_or_dict"] = CHECKPOINT_PATH

else:
    raise RuntimeError(f"Cannot determine argument name for this diffusers version: {arg_names}")

print(f"→ Using argument keys: {list(kwargs.keys())}")
model = download_from_original_stable_diffusion_ckpt(**kwargs)
model.save_pretrained(OUTPUT_DIR)

print(f"Conversion complete! Saved to: {OUTPUT_DIR}")

# endregion

# region : 5 - Copy tokenizer & VAE from the base model 

for subfolder in ["tokenizer", "tokenizer_2", "vae"]:

    src = os.path.join(base_model_dir, subfolder)
    dst = os.path.join(OUTPUT_DIR, subfolder)

    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"Copied {subfolder} from base model.")

    else:
        print(f"Missing {subfolder} in base model — skipping.")

# endregion

# region : 6 - Verify basic loadability

print("\nVerifying model loadability (Diffusers test)...")

try:
    pipe = StableDiffusionXLPipeline.from_pretrained(OUTPUT_DIR, torch_dtype=torch.float16).to(DEVICE)
    print("Model loaded successfully in Diffusers — ready for fine-tuning!")

except Exception as e:
    print("Verification failed — please check the output folder:")
    print(e)

print(f"Converted model path: {OUTPUT_DIR}")

# endregion