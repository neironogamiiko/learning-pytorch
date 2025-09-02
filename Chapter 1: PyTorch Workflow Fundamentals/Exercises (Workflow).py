import torch
from torch import nn, manual_seed
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.use("TkAgg")

# 1. Create a straight line dataset using the linear regression formula (weight * X + bias).
# Set weight=0.3 and bias=0.9 there should be at least 100 datapoints total.
weight, bias = .3, .9
X = torch.arange(0,1,.01)
y = weight * X + bias

# Split the data into 80% training, 20% testing.
train_split = int(.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Plot the training and testing data so it becomes visual.
plt.figure(figsize=(10, 7))
plt.scatter(X_train, y_train, c='b', s=4, label='Training data')
plt.scatter(X_test, y_test, c='g', s=4, label='Test data')

# 2. Build a PyTorch model by subclassing nn.Module.
class Model(nn.Module):
    # Inside should be a randomly initialized nn.Parameter() with requires_grad=True, one for weights and one for bias.
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))

    # Implement the forward() method to compute the linear regression function you used to create the dataset in 1.
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias

# Once you've constructed the model, make an instance of it and check its state_dict().
model = Model()
print(f"Model parameters: {model.state_dict()}")

# 3. Create a loss function and optimizer using nn.L1Loss() and torch.optim.SGD(params, lr) respectively.
loss_function = nn.L1Loss()
# Set the learning rate of the optimizer to be 0.01 and the parameters to optimize should be the model parameters from the model you created in 2.
optimizer = torch.optim.SGD(params=model.parameters(), lr=.01)

# Write a training loop to perform the appropriate training steps for 300 epochs.
epochs = 301
for epoch in range(epochs):
    model.train()
    y_predicted = model(X_train)
    loss = loss_function(y_predicted, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_predictions = model(X_test)
        test_loss = loss_function(test_predictions , y_test)

    # The training loop should test the model on the test dataset every 20 epochs.
    if epoch % 20 == 0:
        print(f"Current epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
print(f"Weight and bias: {model.state_dict()}")
print(f"Predicted y values:\n{test_predictions}")

plt.scatter(X_test, test_predictions.detach().numpy(), c='r', s=4, label='Predicted data')
plt.title("Linear Regression Fit")
plt.legend()
plt.show()

# 5. Save your trained model's state_dict() to file.
SAVE_PATH = Path("/home/frasero/PycharmProjects/Models")
SAVE_PATH.mkdir(exist_ok=True, parents=True)
MODEL_NAME = "Exercise(entire).pth"
MODEL_SAVING_PATH = SAVE_PATH / MODEL_NAME
torch.save(obj=model,f=MODEL_SAVING_PATH)
print(f"Saving entire model to: {MODEL_SAVING_PATH}")
MODEL_NAME = "Exercise(state_dict).pth"
MODEL_SAVING_PATH = SAVE_PATH / MODEL_NAME
torch.save(obj=model.state_dict(),f=MODEL_SAVING_PATH)
print(f"Saving model's parameters to: {MODEL_SAVING_PATH}")

# Create a new instance of your model class you made in 2. and load in the state_dict() you just saved to it.
model_loaded = Model()
model_loaded.load_state_dict(torch.load("/home/frasero/PycharmProjects/Models/Exercise(state_dict).pth",
                                                  weights_only=True))
print(f"Loaded model's parameters: {model_loaded.state_dict()}")

# Perform predictions on your test data with the loaded model and confirm they match the original model predictions from 4.
model_loaded.eval()
with torch.inference_mode():
    test_predictions_loaded = model_loaded(X_test)

print(f"Predicted y values from loaded model: {test_predictions_loaded}")
print(test_predictions == test_predictions_loaded)

