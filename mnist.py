import os
from datasets import load_dataset, load_from_disk, Dataset
from tinygrad import Device, Tensor, nn
import numpy as np

dev = Device.DEFAULT
print(dev)

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

train_images = Tensor((train_images.astype(np.float32) / 255.0)[:, None, :, :], device=dev)
train_labels = Tensor(train_labels.astype(np.int64), device=dev)
test_images = Tensor((test_images.astype(np.float32) / 255.0)[:, None, :, :], device=dev, requires_grad=False)
test_labels = Tensor(test_labels.astype(np.int64), device=dev)

# print(train_images.shape)
# print(train_labels.shape)


class Model:
  def __init__(self):
    self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
    self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
    self.l3 = nn.Linear(1600, 10)
    
  def __call__(self, x: Tensor) -> Tensor:
    x = self.l1(x).relu().max_pool2d((2, 2))
    x = self.l2(x).relu().max_pool2d((2, 2))
    return self.l3(x.flatten(1).dropout(0.5))
    

X_train, Y_train, X_test, Y_test = train_images, train_labels, test_images, test_labels
# print(X_train.shape, X_train.dtype, Y_train.shape, Y_train.dtype)
model = Model()
acc = (model(X_test).argmax(axis=1) == Y_test).mean()
print(acc.item())

# for i in range(3):
#   img = train_images[i, 0]
#   for line in img:
#     for e in line:
#       print(e, end=' ')
#     print()
#   print("===============")