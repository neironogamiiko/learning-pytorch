import torch
import torchmetrics
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from pathlib import Path
from tqdm.auto import tqdm
import os; import gc

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.cuda.empty_cache()
gc.collect()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
scaler = torch.amp.GradScaler(device="cuda")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")

BATCH_SIZE = 32
SEED = 42
N_CLASSES = 10

(torch.cuda.manual_seed if device == "cuda" else torch.manual_seed)(SEED)

# Data preparing

class Data:
    def __init__(self):
        train_data = datasets.FashionMNIST(
            root="data",  # where to download data to?
            train=True,  # do we want the training dataset?
            download=True,  # do we want to download yes/no?
            transform=ToTensor(),  # how do we want to transform the data
            target_transform=None  # how do we want to transform the labels/targets?
        )

        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
            target_transform=None
        )

        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        self.test_loader = DataLoader(
            dataset=test_data,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        self.fully_connected_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, N_CLASSES)
        )

    def forward(self, x):
        return self.fully_connected_layers(self.convolution_layers(x))

data = Data()
model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=.001)
accuracy = torchmetrics.Accuracy("multiclass", num_classes=N_CLASSES).to(device)

def train(epochs:int, model:nn.Module, train_loader:DataLoader):
    for epoch in tqdm(range(epochs)):
        # train mode
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # forward pass
            train_logits = model(x_batch)
            train_predictions = torch.argmax(train_logits, dim=1)

            # loss/accuracy
            loss = criterion(train_logits, y_batch)
            train_loss += loss.item()
            accuracy.update(train_predictions, y_batch)

            # optimizer zero grad
            optimizer.zero_grad()

            # loss backward
            loss.backward()

            # optimizer step
            optimizer.step()
        train_loss /= len(train_loader)
        train_accuracy = accuracy.compute().item()
        accuracy.reset()

        predictions, test_loss, test_accuracy = evaluate(model, data.test_loader)
        print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train Accuracy: {train_accuracy*100:.2f} | Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy*100:.2f}")

def evaluate(model:nn.Module, test_loader:DataLoader):
    all_predictions = []
    model.eval()
    test_loss = 0
    with torch.inference_mode():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            test_logits = model(x_batch)
            test_predictions = torch.argmax(test_logits, dim=1)

            loss = criterion(test_logits, y_batch)
            test_loss += loss.item()
            accuracy.update(test_predictions, y_batch)

            all_predictions.append(test_predictions)
        test_loss /= len(test_loader)
        test_accuracy = accuracy.compute().item()
        accuracy.reset()

    predictions = torch.cat(all_predictions, dim=0)
    return predictions, test_loss, test_accuracy

train(20, model, data.train_loader)
predictions, test_loss, test_accuracy = evaluate(model, data.test_loader)

print(f"Final loss: {test_loss:.4f} | Final accuracy: {test_accuracy*100:.2f}%")
print(predictions)

SAVE_PATH = Path("/home/frasero/PycharmProjects/Models")
MODEL_NAME = "MulticlassificationFashionMNIST(state_dict).pth"
FULL_PATH = SAVE_PATH / MODEL_NAME
torch.save(model.state_dict(), FULL_PATH)
print(f"Saving model's parameters to: {FULL_PATH}")