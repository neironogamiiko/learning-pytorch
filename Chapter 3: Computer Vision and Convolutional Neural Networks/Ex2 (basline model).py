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
from timeit import default_timer as timer
from tqdm.auto import tqdm

# When starting to build a series of machine learning modelling experiments it's best practice to start with a baseline model.
# A `baseline model` is a simple model you will try and improve upon with subsequent models/experiments.
# In other words: start simply and add complexity when necessary.

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Current device: {device}")

torch.manual_seed(42)

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
EPOCHS = 3

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

def print_time(start: float,
               end: float,
               device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time} seconds")
    return total_time

train_time_on_gpu = timer()

# 1. Loop through epochs
# 2. Loop through training batches, perform training steps, calculate the train loss per batch
# 3. loop through testing batches, perform testing steps, calculate the test loss per batch
# 4. Print out what's happening.

train_time_on_cpu = timer()

for epoch in tqdm(range(EPOCHS)):
    print(f"\nEpoch: {epoch}\n-----")
    train_loss, train_accuracy = 0, 0
    for batch, (X,y) in enumerate(train_loader):
        model.train()
        train_predictions = model(X)

        loss = criterion(train_predictions, y)
        train_loss += loss

        train_accuracy += accuracy(train_predictions.argmax(dim=1), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_loader.dataset)} samples.")
    accuracy.reset()
    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)

    test_loss, test_accuracy = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in test_loader:
            test_predictions = model(X)
            test_loss += criterion(test_predictions, y)
            test_accuracy += accuracy(test_predictions.argmax(dim=1), y)
        accuracy.reset()
        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)

    print(f"Train loss: {train_loss:.4f} | Train accuracy: {train_accuracy*100:.2f}% | Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy*100:.2f}%")

end_time_on_cpu = timer()
end_time_on_gpu = timer()
print_time(train_time_on_cpu, end_time_on_cpu, device=str(next(model.parameters())))

def eval_model(model : nn.Module,
               data_loader : DataLoader,
               criterion : nn.Module,
               accuracy):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            predictions = model(X)

            loss += criterion(predictions, y)
            acc += accuracy(predictions.argmax(dim=1), y)

        accuracy.reset()
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc*100}

model_results = eval_model(model, test_loader, criterion, accuracy)
print(model_results)