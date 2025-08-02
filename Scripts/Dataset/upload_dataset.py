import os
import pandas as pd

from huggingface_hub import login
from datasets import Dataset, Image

login(token="")

# Chargement des données
df = pd.read_parquet("Data/dataset.parquet")
df["image"] = df["image"].apply(lambda x: os.path.join("Dataset", os.path.basename(x)))

df = df[df["image"].apply(os.path.exists)]
print(f"{len(df)} images prêtes à être envoyées.")

# Conversion en Dataset Hugging Face
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("image", Image())

# Upload vers le Hub
dataset.push_to_hub("Viennoiserie/SDXL_Dataset")