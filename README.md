# Stable Diffusion XL - LoRA Fine-Tuning Pipeline

This repository contains a full pipeline for training a LoRA adapter on a Stable Diffusion XL (SDXL) model using custom image-caption pairs, with tools for data preparation, training, and deployment via Hugging Face Hub.

---

## 📌 Overview

- **Image Generation Pipeline**: Load and run SDXL with pre-trained weights.
- **Dataset Creation**: Manually prompt images and save structured prompts in `.parquet`.
- **Upload Dataset**: Push dataset to Hugging Face Hub.
- **Training (LoRA)**: Fine-tune SDXL with LoRA using Hugging Face’s `diffusers` and `accelerate`.
- **Deployment**: Push fine-tuned LoRA weights to Hugging Face Hub.

---

## 📦 Setup Instructions

```bash
# Step 0: Move scripts to pod environment, then:
cd Scripts

# Step 1: Clone diffusers repo
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .

# Step 2: Install dependencies
cd ..
pip install -r requirements.txt

# Step 3: Authenticate
huggingface-cli login
wandb login  # optional if using W&B

# Step 4: Configure Accelerate
accelerate config default

# Step 5: Run training
bash lora_training.sh
```

---

## 📁 Components

### 1. **Image Generation**

```python
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="Viennoiserie/Pony",
                             filename="pony_v50.safetensors",
                             token="hf_xxx")

pipe = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
pipe.enable_model_cpu_offload()
```

---

### 2. **Dataset Preparation**

```bash
# Manual prompting via terminal
python dataset_pipeline.py
```

- Opens each image and asks for a description.
- Saves data to `dataset.parquet`.

---

### 3. **Upload to Hugging Face**

```python
from datasets import Dataset, Image

df = pd.read_parquet("Data/dataset.parquet")
df["image"] = df["image"].apply(lambda x: os.path.join("Dataset", os.path.basename(x)))

dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("image", Image())
dataset.push_to_hub("Viennoiserie/SDXL_Dataset")
```

---

### 4. **Training with LoRA**

Run this with `accelerate launch`:
```bash
bash lora_training.sh
```

The training script:
- Loads the SDXL base model and dataset.
- Adds LoRA adapters to UNet and optionally the text encoders.
- Logs training metrics to TensorBoard/W&B.
- Saves adapters and optionally pushes to the Hub.

Example prompt used:
```bash
--validation_prompt="A blond man on the beach"
```

> ⚠️ **Note**: The prompt is used for validation image generation. You can change it based on your dataset.

---

## 📝 Requirements

```
scipy
packaging
tokenizers
safetensors
peft>=0.7.1
datasets>=2.16.0
accelerate>=0.27.0
transformers>=4.36.0
diffusers>=0.35.0.dev0  
torch>=2.1.0
torchvision
bitsandbytes  
xformers>=0.0.23
tqdm
wandb
Pillow
matplotlib
numpy
```

---

## 📤 Hugging Face Hub Upload

Models are pushed to:
```
Viennoiserie/SDXL_Dataset  # Dataset
AI_SDXL_Test               # LoRA model
```

---

## ✅ Checklist Before Training

- [x] Dataset prepared in `.parquet` format.
- [x] Hugging Face Hub token configured.
- [x] `wandb login` if using W&B.
- [x] GPU-enabled pod setup with enough VRAM (>=16GB recommended).
- [x] `lora_training.sh` modified with your custom paths and dataset.

---