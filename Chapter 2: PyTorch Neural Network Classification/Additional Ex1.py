import torch
from torch import nn, optim, inference_mode
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from helper_functions import plot_decision_boundary
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Current device: {device}")

class DataWrapper(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

class Data:
    def __init__(self, size, device):
        self.size = size
        self.device = device

        X, y = make_circles(SAMPLES, noise=.03, random_state=42)

        plt.figure(figsize=(10, 10))
        plt.title('Data')
        plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
        plt.show()

        X = torch.from_numpy(X).type(torch.float32).to(self.device)
        y = torch.from_numpy(y).type(torch.float32).to(self.device)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
                                                                                test_size=.2, random_state=42)

        self.train_loader = DataLoader(
            DataWrapper(self.X_train, self.y_train),
            batch_size=16, shuffle=True
        )

        self.test_loader = DataLoader(
            DataWrapper(self.X_test, self.y_test),
            batch_size=16, shuffle=False
        )

def accuracy_metric(y_true, y_prediction):
    correct = torch.eq(y_true, y_prediction).sum().item()
    accuracy = (correct / len(y_prediction)) * 100
    return accuracy

SAMPLES = 2000
data = Data(SAMPLES, device)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )

    def forward(self, x):
        return self.layers(x)

model = Model().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(params=model.parameters(), lr=.1)


epochs = 300
for epoch in range(epochs):
    model.train()
    for x_batch, y_batch in data.train_loader:
        # forward
        y_logits = model(x_batch).squeeze()
        y_predictions = torch.round(torch.sigmoid(y_logits))

        # loss/accuracy
        loss = criterion(y_logits, y_batch)
        accuracy = accuracy_metric(y_batch, y_predictions)

        # optimizer zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        # optimizer step
        optimizer.step()

    # testing
    all_predictions = []
    model.eval()
    with inference_mode():
        for x_batch, y_batch in data.test_loader:
            test_logits = model(x_batch).squeeze()
            test_predictions = torch.round(torch.sigmoid(test_logits))

            test_loss = criterion(test_logits, y_batch)
            test_accuracy = accuracy_metric(y_batch, test_predictions)

            all_predictions.append(test_predictions)

    all_predictions = torch.cat(all_predictions, dim=0)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss:.5f} | Train accyracy: {accuracy:.2f}% | Test loss: {test_loss:.5f} | Test accuracy: {test_accuracy:.2f}%")

print(f"Test predictions:\n{all_predictions}")

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model, data.X_train, data.y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model, data.X_test, data.y_test)
plt.show()

SAVE_PATH = Path("/home/frasero/PycharmProjects/Models")
MODEL_NAME = "CircleBinaryClassificator(state_dict).pth"
MODEL_SAVE_PATH = SAVE_PATH / MODEL_NAME
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Saving model's parameters to: {MODEL_SAVE_PATH}")