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
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Current device: {device}")

SEED = 42
BATCH_SIZE = 32

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
        K = 3 # number of classes

        X = np.zeros((N*K,D)) # data matrix (each row = single example)
        y = np.zeros(N*K, dtype='uint8') # class labels

        for j in range(K):
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

# Construct a model capable of fitting the data (you may need a combination of linear and non-linear layers).



# Build a loss function and optimizer capable of handling multi-class data (optional extension: use the Adam optimizer instead of SGD, you may have to experiment with different values of the learning rate to get it working).
# Make a training and testing loop for the multi-class data and train a model on it to reach over 95% testing accuracy (you can use any accuracy measuring function here that you like).
# Plot the decision boundaries on the spirals dataset from your model predictions, the plot_decision_boundary() function should work for this dataset too.