import pandas as pd

df = pd.read_parquet("C:/Users/thoma/Documents/Thomas - SSD/LoRA - Fine Tune/Example Dataset/oldbookillustrations-small/data/train-00000-of-00001.parquet")
print(df.head())
