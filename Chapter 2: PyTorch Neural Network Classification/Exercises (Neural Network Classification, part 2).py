import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("TkAgg")
from torchmetrics.classification import Accuracy
from helper_functions import plot_decision_boundary
from pathlib import Path
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Current device: {device}")

SEED = 42
BATCH_SIZE = 32
N_CLASSES = 3
NEED_SAVE = True

torch.manual_seed(SEED); torch.cuda.manual_seed(SEED)

# Create a multi-class dataset using the spirals data creation function from CS231n (see below for the code).

class DataWrapper(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

class Data:
    def __init__(self):

        # Code for creating a spiral dataset from CS231n
        N = 300 # number of points per class
        D = 2 # dimensionality

        X = np.zeros((N*N_CLASSES,D)) # data matrix (each row = single example)
        y = np.zeros(N*N_CLASSES, dtype='uint8') # class labels

        for j in range(N_CLASSES):
          ix = range(N*j,N*(j+1))
          r = np.linspace(0.0,1,N) # radius
          t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
          X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
          y[ix] = j

        # lets visualize the data
        plt.figure(figsize=(10,10))
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
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

data = Data()

# Construct a model capable of fitting the data (you may need a combination of linear and non-linear layers).

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.layers(x)

model = Model().to(device)
# Build a loss function and optimizer capable of handling multi-class data (optional extension: use the Adam optimizer instead of SGD, you may have to experiment with different values of the learning rate to get it working).
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=.001)

accuracy = Accuracy(task="multiclass", num_classes=N_CLASSES).to(device)

# Make a training and testing loop for the multi-class data and train a model on it to reach over 95% testing accuracy (you can use any accuracy measuring function here that you like).

def train(epochs:int, model:Model, train_loader:DataLoader):
    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_accuracy = 0, 0
        for x_batch, y_batch in train_loader:
            train_logits = model(x_batch)
            train_predictions = torch.argmax(train_logits, dim=1)
            train_loss = criterion(train_logits, y_batch)
            train_accuracy = accuracy(train_predictions, y_batch)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            epoch_loss += train_loss.item()
            epoch_accuracy += train_accuracy.item()

        epoch_loss /= len(train_loader)
        epoch_accuracy /= len(train_loader)
        accuracy.reset()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Train loss: {epoch_loss:.4f} | Train accuracy: {epoch_accuracy*100:.2f}%")

def test(model:Model, test_loader:DataLoader) -> torch.Tensor:
    all_predictions = []
    model.eval()
    epoch_loss, epoch_accuracy = 0, 0
    with torch.inference_mode():
        for x_batch, y_batch in test_loader:
            test_logits = model(x_batch)
            test_predictions = torch.argmax(test_logits, dim=1)

            test_loss = criterion(test_logits, y_batch)
            test_accuracy = accuracy(test_predictions, y_batch)

            epoch_loss += test_loss.item()
            epoch_accuracy += test_accuracy.item()

            all_predictions.append(test_predictions)

        epoch_loss /= len(test_loader)
        epoch_accuracy /= len(test_loader)
        accuracy.reset()
        print(f"Test loss: {epoch_loss:.4f} | Test accuracy: {test_accuracy*100:.2f}%")

    predictions = torch.cat(all_predictions, dim=0)
    return predictions

def make_plot():
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, data.X_train, data.y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, data.X_test, data.y_test)
    plt.show()

train(301, model, data.train_loader)
predictions = test(model, data.test_loader)

print(f"Predictions:\n{predictions}")
print(predictions==data.y_test)

result_accuracy = (predictions == data.y_test).float().mean()
print(f"Test accuracy: {result_accuracy*100:.2f}%")
# Plot the decision boundaries on the spirals dataset from your model predictions, the plot_decision_boundary() function should work for this dataset too.
make_plot()

if NEED_SAVE:
    SAVE_PATH = Path("/home/frasero/PycharmProjects/Models")
    MODEL_NAME = "MulticlassClassificatorForNoiseCS321nData(state_dict).pth"
    FULL_PATH = SAVE_PATH / MODEL_NAME
    torch.save(model.state_dict(), FULL_PATH)
    print(f"Saving model's parameters to: {FULL_PATH}")