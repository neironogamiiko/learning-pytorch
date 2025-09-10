import torch;
from torch import nn
import os; from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("TkAgg")
from torch.utils.data import Dataset, DataLoader

if torch.cuda.is_available():
    print(f"Current device ID: {torch.cuda.current_device()}")
    print(f"Current device name: {torch.cuda.get_device_name()}")
    device = 'cuda'
    os.system("nvidia-smi")
    torch.cuda.manual_seed(42)
else:
    device = 'cpu'
    print(f"Current device: {device}")
    torch.manual_seed(42)

# # 1. Data preparation
class DataWrapper(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class SyntheticDataset:
    def __init__(self, size, device):
        self.size = size
        self.device = device

        X = torch.empty(self.size, 1, device=self.device).uniform_(-5,5)
        noise = torch.randn(self.size, 1, dtype=torch.float32, device=self.device)
        y = 2 * X + 1 + noise

        train_split = int(.8 * len(X))
        self.X_train, self.y_train = X[:train_split], y[:train_split]
        self.X_test, self.y_test = X[train_split:], y[train_split:]

        self.train_loader = DataLoader(
            DataWrapper(self.X_train, self.y_train),
            batch_size=16, shuffle=True
        )

        self.test_loader = DataLoader(
            DataWrapper(self.X_test, self.y_test),
            batch_size=16, shuffle=False
        )

# 2. Build model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# 3. Train model
def train(model : Model, epochs, criterion, optimizer, train_loader):
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            y_predictions = model(x_batch)
            train_loss = criterion(y_predictions, y_batch)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Train loss: {train_loss}")


# 4. Make predictions
def evaluate(model : Model, criterion, test_loader):
    all_predictions = []

    model.eval()
    with torch.inference_mode():
        for x_batch, y_batch in test_loader:
            test_predictions = model(x_batch)
            test_loss = criterion(test_predictions, y_batch)
            all_predictions.append(test_predictions)
    all_predictions = torch.cat(all_predictions, dim=0)
    print(f"Test loss: {test_loss}\nTest predictions:\n{all_predictions}")
    print(f"Model's parameters: {model.state_dict()}")

    return all_predictions

# 5. Save & Load
def save_model(model : Model, entire_flag=False):
    SAVE_PATH = Path("/home/frasero/PycharmProjects/Models")
    MODEL_NAME = "ExcerciseChatGPT(state_dict)"
    MODEL_SAVE_PATH = SAVE_PATH / MODEL_NAME
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)
    print(f"Saving model's parameters (state_dict) to: {MODEL_SAVE_PATH}")

    if entire_flag:
        MODEL_NAME = "ExcerciseChatGPT(entire)"
        MODEL_SAVE_PATH = SAVE_PATH / MODEL_NAME
        torch.save(obj=model, f=MODEL_SAVE_PATH)
        print(f"Saving entire model to: {MODEL_SAVE_PATH}")

def load_model(LOAD_PATH, entire_flag=False) -> Model:
    model = Model()
    model.load_state_dict(torch.load(LOAD_PATH))
    model.to(device)

    if entire_flag:
        model = torch.load(LOAD_PATH)

    return model

# 6. Plot data
def plot_data(data : SyntheticDataset):
    plt.figure(figsize=(10, 7))
    plt.scatter(data.X_train.cpu(), data.y_train.cpu(), c='b', s=4, label='Training data')
    plt.scatter(data.X_test.cpu(), data.y_test.cpu(), c='g', s=4, label='Test data')
    plt.scatter(data.X_test.cpu(), test_predictions.cpu(), c='r', s=10, label='Predicted data')
    plt.title("y = 2 * X + 1 + noise")
    plt.legend()
    plt.show()

data = SyntheticDataset(100, device)
model = Model().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.01)
train(model, 1001, criterion, optimizer, data.train_loader)
test_predictions = evaluate(model, criterion, data.test_loader)
plot_data(data)