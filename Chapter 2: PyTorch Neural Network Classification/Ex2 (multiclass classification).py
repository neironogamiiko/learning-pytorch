import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("TkAgg")

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
        y_blob = torch.from_numpy(y_blob).type(torch.float32).to(device)

        self.X_blob_train, self.X_blob_test, self.y_blob_train, self.y_blob_test = train_test_split(X_blob, y_blob,
                                                                                                    test_size=.2,
                                                                                                    random_state=SEED)

        self.train_loader = DataLoader(
            DataWrapper(self.X_blob_train, self.y_blob_train),
            batch_size=16, shuffle=True
        )

        self.test_loader = DataLoader(
            DataWrapper(self.X_blob_test, self.y_blob_test),
            batch_size=16, shuffle=False
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

data = Data()

model = MulticlassClassification().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr=.1)



