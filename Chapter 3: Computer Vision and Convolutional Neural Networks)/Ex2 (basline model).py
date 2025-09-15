import torch
import torchmetrics
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("TkAgg")
import requests; from pathlib import Path

# When starting to build a series of machine learning modelling experiments it's best practice to start with a baseline model.
# A `baseline model` is a simple model you will try and improve upon with subsequent models/experiments.
# In other words: start simply and add complexity when necessary.

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

class_names = train_data.classes
BATCH_SIZE = 32
N_CLASSES = len(class_names)

train_loader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Length of train loader: {len(train_loader)} batches of {BATCH_SIZE}")
print(f"Length of test loader: {len(test_loader)} batches of {BATCH_SIZE}")

train_features_batch, train_labels_batch = next(iter(train_loader))

# Create a flatten layer
flatten_model = nn.Flatten()

# Get a single sample
x = train_features_batch[0]

# Flatten the sample
output = flatten_model(x) # perform a forward pass

print(f"Shape before flattening: {x.shape} -> [color_channels, height, width]")
print(f"Sha[e after flattening: {output.shape} -> [color_channels, height * width")

class FashionModel(nn.Module):
    def __init__(self,
                 input_shape: int,
                 output_shape: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=output_shape)
        )

        # Conv2d(1→32, 3×3) + ReLU
        # Conv2d(32→64, 3×3) + ReLU
        # MaxPool(2×2)
        # Dropout
        # Flatten
        # Linear(12544 → 128) + ReLU
        # Dropout
        # Linear(128 → 10)

    def forward(self, x):
        return self.layers(x)

input_shape = x.shape[1] * x.shape[2]
model = FashionModel(input_shape=input_shape, output_shape=N_CLASSES)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr=.1)
accuracy = torchmetrics.Accuracy('multiclass', num_classes=N_CLASSES)

# load helper functions
if Path('helper_functions.py').is_file():
    print("Skipping download. . . ")
else:
    print("Downloading file. . . ")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_functions.py", "wb") as file:
        file.write(request.content)

