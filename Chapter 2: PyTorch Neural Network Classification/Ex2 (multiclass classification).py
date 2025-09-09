import torch
from torch import nn, optim, inference_mode
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("TkAgg")
from helper_functions import plot_decision_boundary

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Current device: {device}")

NUM_CLASSES = 4
NUM_FEATURES = 2
SEED = 42

class DataWrapper(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

class Data:
    def __init__(self):
        X_blob, y_blob = make_blobs(n_samples=1000,
                                    n_features=NUM_FEATURES,
                                    centers=NUM_CLASSES,
                                    cluster_std=1.5,
                                    random_state=SEED)

        plt.figure(figsize=(10,7))
        plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
        plt.show()

        X_blob = torch.from_numpy(X_blob).type(torch.float32).to(device)
        y_blob = torch.from_numpy(y_blob).type(torch.LongTensor).to(device)

        self.X_blob_train, self.X_blob_test, self.y_blob_train, self.y_blob_test = train_test_split(X_blob, y_blob,
                                                                                                    test_size=.2,
                                                                                                    random_state=SEED)

        self.train_loader = DataLoader(
            DataWrapper(self.X_blob_train, self.y_blob_train),
            batch_size=32, shuffle=True
        )

        self.test_loader = DataLoader(
            DataWrapper(self.X_blob_test, self.y_blob_test),
            batch_size=32, shuffle=False
        )

class MulticlassClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

def accuracy_metric(y_true, y_prediction):
    correct = torch.eq(y_true, y_prediction).sum().item()
    accuracy = (correct / len(y_prediction)) * 100
    return accuracy

data = Data()

model = MulticlassClassification().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr=.1)

epochs = 101
for epoch in range(epochs):
    model.train()
    for x_batch, y_batch in data.train_loader:
        # forward
        y_logits = model(x_batch)
        y_predictions = torch.softmax(y_logits, dim=1).argmax(dim=1)

        # loss/accuracy
        loss = criterion(y_logits, y_batch)
        accuracy = accuracy_metric(y_batch, y_predictions)

        # optimizer zero grad
        optimizer.zero_grad()

        # backpropagation
        loss.backward()

        # step the optimizer
        optimizer.step()

    all_predictions = []
    model.eval()
    with inference_mode():
        for x_batch, y_batch in data.test_loader:
            test_logits = model(x_batch)
            test_predictions = torch.softmax(test_logits, dim=1).argmax(dim=1)

            test_loss = criterion(test_logits, y_batch)
            test_accuracy = accuracy_metric(y_batch, test_predictions)

            all_predictions.append(test_predictions)
    all_predictions = torch.cat(all_predictions, dim=0)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Accuracy: {accuracy:.2f}% | Test loss: {test_loss:.5f} | Test accuracy: {test_accuracy:.2f}%")

print(f"Test predictions:\n{all_predictions}")
print(f"True:\n{data.y_blob_test}")
print(all_predictions == data.y_blob_test)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model, data.X_blob_train, data.y_blob_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model, data.X_blob_test, data.y_blob_test)
plt.show()