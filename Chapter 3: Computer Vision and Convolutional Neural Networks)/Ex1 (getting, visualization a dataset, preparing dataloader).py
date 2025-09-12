import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("TkAgg")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Current device: {device}")

# Setup training data
train_data = datasets.FashionMNIST(
    root="data",            # where to download data to?
    train=True,             # do we want the training dataset?
    download=True,          # do we want to download yes/no?
    transform=ToTensor(),   # how do we want to transform the data
    target_transform=None   # how do we want to transform the labels/targets?
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

image, label = train_data[0]

print(f"Number of training samples: {len(train_data)}")
print(f"Number of testing samples: {len(test_data)}\n")

class_names = train_data.classes
class_idx = train_data.class_to_idx
print(f"Number of classes: {len(class_names)}\nClasses: {class_idx}")

# Check the shape of our image
print(f"Image shape: {image.shape} and it's label: {class_names[label]}: {label}")

# Visualization

# def make_plot(image, label):
#     plt.imshow(image.squeeze(), cmap="gray")
#     plt.title(class_names[label])
#     plt.show()

torch.manual_seed(42)
fig = plt.figure(figsize=(9,9))
row,col = 4, 4
for i in range(1, row*col+1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(row, col, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.grid(False)
    plt.axis(False)
plt.show()

# Prepare dataloader

# It's more computationally efficient, as in, your computing hardware may not be able to look (store in memory) at 60000 images in one hit.
# So we brake it dow to `n` images at a time (batch size of `n`).
# It gives our neural network more chances to update it's gradients per epoch.

BATCH_SIZE = 32

class DataWrapper(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

