import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

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