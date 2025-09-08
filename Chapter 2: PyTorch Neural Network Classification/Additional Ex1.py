import torch
from torch import nn, optim, inference_mode
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from helper_functions import plot_decision_boundary
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Current device: {device}")

def accuracy_metric(y_true, y_prediction):
    correct = torch.eq(y_true, y_prediction).sum().item()
    accuracy = (correct / len(y_prediction)) * 100
    return accuracy

SAMPLES = 2000
X, y = make_circles(SAMPLES, noise=.03,random_state=42)

plt.figure(figsize=(10,10))
plt.title('Data')
plt.scatter(x=X[:,0], y=X[:,1], c=y, cmap=plt.cm.RdYlBu)
plt.show()

X = torch.from_numpy(X).type(torch.float32).to(device)
y = torch.from_numpy(y).type(torch.float32).to(device)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

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


epochs = 10001
for epoch in range(epochs):
    model.train()
    # forward
    y_logits = model(X_train).squeeze()
    y_predictions = torch.round(torch.sigmoid(y_logits))

    # loss/accuracy
    loss = criterion(y_logits, y_train)
    accuracy = accuracy_metric(y_train, y_predictions)

    # optimizer zero grad
    optimizer.zero_grad()

    # loss backward
    loss.backward()

    # optimizer step
    optimizer.step()

    # testing
    model.eval()
    with inference_mode():
        test_logits = model(X_test).squeeze()
        test_predictions = torch.round(torch.sigmoid(test_logits))

        test_loss = criterion(test_logits, y_test)
        test_accuracy = accuracy_metric(y_test, test_predictions)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {accuracy:.2f}% | Test loss: {test_loss:.5f} | test accuracy: {test_accuracy:.2f}%")

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
plt.show()

SAVE_PATH = Path("/home/frasero/PycharmProjects/Models")
MODEL_NAME = "CircleBinaryClassificator(state_dict).pth"
MODEL_SAVE_PATH = SAVE_PATH / MODEL_NAME
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Saving model's parameters to: {MODEL_SAVE_PATH}")