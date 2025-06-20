---
language:
- en
size_categories:
- 1K<n<10K
task_categories:
- text-to-image
dataset_info:
  features:
  - name: image
    dtype: image
  - name: text
    dtype: string
  splits:
  - name: train
    num_bytes: 409697476
    num_examples: 1000
  download_size: 407436946
  dataset_size: 409697476
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
tags:
- art
---

Truncated version of [gigant/oldbookillustrations](https://huggingface.co/datasets/gigant/oldbookillustrations)

Contains every 4th image until 1,000 images were saved, and their alt_text (handling null alt_text: no caption)

See original dataset for more information, and [Old Book Illustrations website](https://www.oldbookillustrations.com/) for terms of use, policy, and more.