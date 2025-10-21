import os
from datasets import load_dataset, load_from_disk, Dataset
import tinygrad
import numpy as np

mnist_dir = "mnist_datasets"
# download and load mnist datasets from huggingface
if not os.path.exists(f'./{mnist_dir}'):
  ds = load_dataset("ylecun/mnist")
  # saving the datasets
  ds.save_to_disk(mnist_dir)
else:
  ds = load_from_disk(mnist_dir)


train_images = np.array([np.array(img) for img in ds['train']['image']])
train_labels = np.array(ds['train']['label'])
test_images = np.array([np.array(img) for img in ds['test']['image']])
test_labels = np.array(ds['test']['label'])

print(train_images.shape)
