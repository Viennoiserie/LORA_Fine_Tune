import os
import pandas as pd

from dotenv import load_dotenv
from huggingface_hub import login
from datasets import Dataset, Image

# --- HF TOKEN --- #

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

# --- Directory Variables --- #

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Data", "images"))

IMAGE_DIR = os.path.join(BASE_DIR, "character")
DATASET_PATH = os.path.join(BASE_DIR, "dataset.parquet")

# --- Uploading Data --- #

df = pd.read_parquet(DATASET_PATH)

if "image" not in df.columns:
    raise ValueError("La colonne 'image' est absente du fichier parquet. Vérifie ton dataset.parquet.")

# Verify file integrity
df["image"] = df["image"].apply(lambda x: os.path.join(IMAGE_DIR, os.path.basename(x)))
df["exists"] = df["image"].apply(os.path.exists)
df = df[df["exists"]]

print(f"\n{len(df)} images prêtes à être envoyées.")

if len(df) == 0:
    raise ValueError("Aucune image trouvée. Vérifie IMAGE_DIR et dataset.parquet.")

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("image", Image())

# --- Push dataset to HF Hub --- #

dataset.push_to_hub("Viennoiserie/Training_Dataset")
