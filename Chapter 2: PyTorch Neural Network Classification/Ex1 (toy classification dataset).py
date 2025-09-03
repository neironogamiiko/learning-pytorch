import torch; from torch import nn
from sklearn.datasets import make_circles
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
    print(f"Current device name: {torch.cuda.get_device_name()}")
    print(f"Current device id: {torch.cuda.current_device()}")
    device = 'cuda'
else:
    device = 'cpu'
    print(f"Current device: {device}")

# make 1000 samples
N = 1000
X, y = make_circles(N, noise=.03,random_state=42)

plt.figure()
plt.scatter(x=X[:,0], y=X[:,1], c=y, cmap=plt.cm.RdYlBu)
plt.show()

X = torch.from_numpy(X).type(torch.float32).to(device)
y = torch.from_numpy(y).type(torch.float32).to(device)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.2, random_state=42)

class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=2, out_features=5),
            nn.Linear(in_features=5, out_features=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

model = ClassificationModel().to(device)
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=.01)

