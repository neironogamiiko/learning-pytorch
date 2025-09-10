import torch
from torch import nn, optim, manual_seed
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("TkAgg")
from helper_functions import plot_decision_boundary
from torchmetrics.classification import BinaryAccuracy
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Current device: {device}")

# 1. Make a binary classification dataset with Scikit-Learn's make_moons() function.
# For consistency, the dataset should have 1000 samples and a random_state=42.
# Turn the data into PyTorch tensors. Split the data into training and test sets using train_test_split with 80% training and 20% testing.

N_SAMPLES = 1000
SEED = 42
BATCH_SIZE = 32
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
        X, y = make_moons(n_samples=N_SAMPLES,
                          noise=.2,
                          random_state=SEED)

        plt.figure(figsize=(10, 7))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
        plt.show()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,
                                                            test_size=.2,
                                                            random_state=SEED)

        self.X_train = torch.from_numpy(self.X_train).type(torch.float32).to(device)
        self.X_test = torch.from_numpy(self.X_test).type(torch.float32).to(device)
        self.y_train = torch.from_numpy(self.y_train).type(torch.float32).to(device)
        self.y_test = torch.from_numpy(self.y_test).type(torch.float32).to(device)

        self.train_loader = DataLoader(
            DataWrapper(self.X_train, self.y_train),
            batch_size=BATCH_SIZE, shuffle=True
        )

        self.test_loader = DataLoader(
            DataWrapper(self.X_test, self.y_test),
            batch_size=BATCH_SIZE, shuffle=False
        )

data = Data()

# 2. Build a model by subclassing nn.Module that incorporates non-linear activation functions and is capable of fitting the data you created in 1.
# Feel free to use any combination of PyTorch layers (linear and non-linear) you want.

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.layers(x)

model = Model().to(device)

# 3. Setup a binary classification compatible loss function and optimizer to use when training the model.
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(params=model.parameters(), lr=.001)

# 4. Create a training and testing loop to fit the model you created in 2 to the data you created in 1.
# To measure model accuracy, you can create your own accuracy function or use the accuracy function in TorchMetrics.
# Train the model for long enough for it to reach over 96% accuracy.
# The training loop should output progress every 10 epochs of the model's training and test set loss and accuracy.
accuracy = BinaryAccuracy().to(device)

def train(epochs : int, model: Model, data: Data):
    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_acc = 0, 0
        for x_batch, y_batch in data.train_loader:
            train_logits = model(x_batch).view(-1)
            train_predictions = torch.round(torch.sigmoid(train_logits))
            train_loss = criterion(train_logits, y_batch)
            train_accuracy = accuracy(train_predictions, y_batch.int())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            epoch_loss += train_loss.item()
            epoch_acc += train_accuracy.item()
        epoch_loss /= len(data.train_loader)
        epoch_acc /= len(data.train_loader)
        accuracy.reset()
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Train loss: {epoch_loss:.4f} | Train accuracy: {epoch_acc:.4f}")

def evaluate(model: Model, data : Data) -> torch.Tensor:
    all_predictions = []
    model.eval()
    epoch_loss, epoch_acc = 0, 0
    with torch.inference_mode():
        for x_batch, y_batch in data.test_loader:
            test_logits = model(x_batch).view(-1)
            test_predictions = torch.round(torch.sigmoid(test_logits))

            test_loss = criterion(test_logits, y_batch)
            test_accuracy = accuracy(test_predictions, y_batch.int())

            epoch_loss += test_loss.item()
            epoch_acc += test_accuracy.item()
            all_predictions.append(test_predictions)

        epoch_loss /= len(data.test_loader)
        epoch_acc /= len(data.test_loader)
        accuracy.reset()
        print(f"Test loss: {epoch_loss:.4f} | Test accuracy: {epoch_acc:.4f}")

    all_predictions = torch.cat(all_predictions, dim=0)

    return all_predictions

def make_plot():
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, data.X_train, data.y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, data.X_test, data.y_test)
    plt.show()

# 5. Make predictions with your trained model and plot them using the plot_decision_boundary() function created in this notebook.
train(301, model, data)
predictions = evaluate(model, data)
make_plot()

print(predictions == data.y_test)

if NEED_SAVE:
    SAVE_PATH = Path("/home/frasero/PycharmProjects/Models")
    MODEL_NAME = "BinaryClassificatorForNoiseMoonData(state_dict).pth"
    FULL_PATH = SAVE_PATH / MODEL_NAME
    torch.save(model.state_dict(), FULL_PATH)
    print(f"Saving model's parameters to: {FULL_PATH}")