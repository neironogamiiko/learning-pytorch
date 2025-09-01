import torch
from torch import nn
import os; from pathlib import Path
import numpy

if torch.cuda.is_available():
    print(f"Current device ID: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    device = 'cuda'

    print("\nFull info about device:\n")
    os.system("nvidia-smi")
else:
    device = 'cpu'

# creating some data with linear regression formula
weight, bias = .7, .3

# create range values
start, end, step = 0, 1, .02

# create features and labels (X, y)
X = torch.arange(start, end, step, device=device).unsqueeze(1)
y = weight * X + bias

# split data to train and test sets
train_split = int(.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(f"\nNumber of X train data: {len(X_train)}\n"
      f"Number of y train data: {len(y_train)}\n"
      f"Number of X test data: {len(X_test)}\n"
      f"Number of y test data: {len(y_test)}\n")

# building a model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1) # 1 X value equals 1 y value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

torch.manual_seed(42)
model = LinearRegression()
print(f"Model's structure: {model}\nModel's parameters: {model.state_dict()}")

# set the model to use the target device
model.to(device)

# loss function and optimizer
loss_function = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=.01)

# training loop
epochs = 301
for epoch in range(epochs):
    model.train()

    # forward
    y_predicted = model(X_train)

    # loss
    loss = loss_function(y_predicted, y_train)

    # optimizer zero grad
    optimizer.zero_grad()

    # backward
    loss.backward()

    # optimizer step
    optimizer.step()

    # testing
    model.eval()
    with torch.inference_mode():
        test_predictions = model(X_test)
        test_loss = loss_function(test_predictions, y_test)

    if epoch % 100 == 0:
        print(f"Current epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
print(f"Weight and bias: {model.state_dict()}")
print(f"Predicted y values:\n{test_predictions}")

# save model
SAVING_PATH = Path("/home/frasero/PycharmProjects/Models")
SAVING_PATH.mkdir(exist_ok=True,parents=True)
MODEL_NAME = "LinearRegression(entire).pth"
MODEL_SAVING_PATH = SAVING_PATH / MODEL_NAME
torch.save(obj=model, f=MODEL_SAVING_PATH)
print(f"Saving entire model to: {MODEL_SAVING_PATH}")
MODEL_NAME = "LinearRegression(state_dict).pth"
MODEL_SAVING_PATH = SAVING_PATH / MODEL_NAME
torch.save(obj=model.state_dict(),f=MODEL_SAVING_PATH)
print(f"Saving model's weight and bias to: {MODEL_SAVING_PATH}")