import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torchmetrics
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("TkAgg")
from helper_functions import plot_decision_boundary
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Current device: {device}")

N_SAMPLES = 1000
SEED = 42
BATCH_SIZE = 32
N_CLASSES = 3
NEED_SAVE = True

torch.manual_seed(SEED); torch.cuda.manual_seed(SEED)

class DataWrapper(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

class Data:
    def __init__(self):
        X1, y1 = make_moons(n_samples=N_SAMPLES, noise=0.1, random_state=SEED)
        X2, y2 = make_moons(n_samples=int(N_SAMPLES/2), noise=0.1, random_state=int(SEED/2))
        X2 += np.array([2.0, .5])
        y2[:] = 2

        X = np.vstack([X1, X2])
        y = np.hstack([y1, y2])

        plt.figure(figsize=(7, 7))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='k')
        plt.title("Data")
        plt.show()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,
                                                                                 test_size=.2,
                                                                                 random_state=SEED)

        self.X_train = torch.from_numpy(self.X_train).type(torch.float32).to(device)
        self.X_test = torch.from_numpy(self.X_test).type(torch.float32).to(device)
        self.y_train = torch.from_numpy(self.y_train).type(torch.long).to(device)
        self.y_test = torch.from_numpy(self.y_test).type(torch.long).to(device)

        self.train_loader = DataLoader(
            DataWrapper(self.X_train, self.y_train),
            batch_size=BATCH_SIZE, shuffle=True
        )
        self.test_loader = DataLoader(
            DataWrapper(self.X_test, self.y_test),
            batch_size=BATCH_SIZE, shuffle=False
        )

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),

        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.layers(x)

data = Data()
model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), .001)
accuracy_metric = torchmetrics.Accuracy("multiclass", num_classes=N_CLASSES).to(device)

def test(model: Model, test_loader: DataLoader):
    all_predictions = []
    model.eval()
    epoch_loss, epoch_accuracy = 0, 0
    with torch.inference_mode():
        for x_batch, y_batch in test_loader:
            test_logits = model(x_batch)
            test_predictions = torch.argmax(test_logits, dim=1)

            test_loss = criterion(test_logits, y_batch)
            test_accuracy = accuracy_metric(test_predictions, y_batch)

            epoch_loss += test_loss.item()
            epoch_accuracy += test_accuracy.item()
            all_predictions.append(test_predictions)
        epoch_loss /= len(test_loader)
        epoch_accuracy /= len(test_loader)
        accuracy_metric.reset()
        # print(f"Test loss: {epoch_loss:.4f} | Test accuracy: {epoch_accuracy * 100:.2f}%")

    predictions = torch.cat(all_predictions, dim=0)
    return predictions, epoch_loss, epoch_accuracy

def train(epochs:int, model:Model, train_loader:DataLoader):
    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_accuracy = 0, 0
        for x_batch, y_batch in train_loader:
            logits = model(x_batch)
            predictions = torch.argmax(logits, dim=1)
            loss = criterion(logits, y_batch)
            accuracy = accuracy_metric(predictions, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
        epoch_loss /= len(train_loader)
        epoch_accuracy /= len(train_loader)
        accuracy_metric.reset()

        prediction, test_loss, test_accuracy = test(model,data.test_loader)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy*100:.2f}% | Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy*100:.2f}%")

    return prediction

def make_plot():
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, data.X_train, data.y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, data.X_test, data.y_test)
    plt.show()

predictions = train(301, model, data.train_loader)
result_accuracy = (predictions == data.y_test).float().mean()
print(f"Model final accuracy: {result_accuracy*100:.2f}%")
make_plot()

if NEED_SAVE:
    SAVE_PATH = Path("/home/frasero/PycharmProjects/Models")
    MODEL_NAME = "MulticlassificationGPTTask(state_dict).pth"
    FULL_PATH = SAVE_PATH / MODEL_NAME
    torch.save(model.state_dict(), FULL_PATH)
    print(f"Saving model's parameters to: {FULL_PATH}")