import torch
from torch import nn
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import numpy as np

if torch.cuda.is_available():
    print(f"Device ID: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}\n")
    device = 'cuda'

    os.system('nvidia-smi')
else:
    device = 'cpu'

def plot_predictions(train_data,train_labels,test_data,test_labels,predictions=None):
    # Figure 1: weights * x + bias
    train_data = train_data.cpu().numpy()
    train_labels = train_labels.cpu().numpy()
    test_data = test_data.cpu().numpy()
    test_labels = test_labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_labels, c='b', s=4, label='Training data')
    plt.scatter(test_data, test_labels, c='g', s=4, label='Test data')

    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')

    plt.legend(prop={"size":14})
    plt.title("weights * x + bias")


class LinearRegressionModel(nn.Module): # all costume models should also subclass torch.nn.Module
    '''
    Linear Regression Model:
    1. Start with random values of weight and bias.
    2. Look at training data and adjust the random values to better represent or get closer to the ideal values.
    '''
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, # start with a random weights and try to adjust it to the ideal weights
                                                requires_grad=True, # for gradient descent we need 'requires_grad=True'
                                                                    # can this parameter be updated via Gradient descent?
                                                dtype=torch.float32,
                                                device=device))
        self.bias = nn.Parameter(torch.randn(1, # start with a random bias and try to adjust it to the ideal bias
                                             requires_grad=True, # for gradient descent we need 'requires_grad=True'
                                                                 # can this parameter be updated via Gradient descent?
                                             dtype=torch.float32,
                                             device=device))

    # Forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias # linear regression formula


weight, bias = .7, .3
start = 0; end = 1; step = .02

X =  torch.arange(start,end,step, device=device).unsqueeze(dim=1)
y = weight * X + bias

# Splitting training and test sets
train = int(.8 * len(X))
X_train, y_train = X[:train], y[:train]
X_test, y_test = X[train:], y[train:]

torch.cuda.manual_seed(42)
model_0 = LinearRegressionModel().to(device=device)
print(f"Parameters: {model_0.state_dict()}")

with torch.inference_mode():
    y_preds = model_0(X_test)

# print(f"Ideal y values:\n{y_test}\n")
# print(f"Predicted y:\n{y_preds}")

# One way to measure how poor or how wrong your models predictions are is to use a loss function.
# Note: 'loss' function may also call 'cost' function, or 'criterion' in different areas.

# Loss function: a function to measure how wrong your model's predictions are to the ideal outputs (lower is better)
# Optimizer: takes into account the loss of a model and adjusts the model's parameters (e.g. weight & bias) to improve the loss function.

# Parameters - it's what model sets itself
# Hyperparameters - it's what data scientist sets

losses = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), # Stochastic Gradient Descent
                            lr=.01) # learning rate

# Training loop:
# 1. Loop through the data
# 2. Forwards pass (this involves data moving through our model's 'forwards()' functions) to make predictions on data
# 3. Calculate the loss (compare forward pass predictions to ground truth labels)
# 4. Optimizer zero grad
# 5. Loss backward - move backwards through the network to calculate the gradients of each of the parameters of our model with respect to the loss (Backpropagation)
# 6. Optimizer step - use the optimizer to adjust our model's parameters tp tru amd improve the loss (Gradient Descent)

epochs = 5000 # one loop through the data . . .
# epoch - it's also a hyperparameter, 'cuse we've set it ourselves

# Track different values
epoch_counter, loss_values, test_loss_values = [], [], []

# Training model
# 1. Loop through the data
for epoch in range(epochs):
    model_0.train() # train mode sets all parameters that require gradients to 'require gradients'

    # 2. Forward pass
    y_pred = model_0(X_train)

    # 3. Calculate the loss
    loss = losses(y_pred,y_train)
    print(f"Loss: {loss}")

    # 4. Optimizer zero grad
    optimizer.zero_grad()

    # 5. Loss backward (Backpropagation)
    loss.backward()

    # 6. Optimizer step (Gradient Descent)
    optimizer.step()

    # Testing
    model_0.eval() # turns off different settings in the model not needed fpr evaluation/testing (dropout/batch norm layers)

    # Test without learning
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = losses(test_pred, y_test)
    if epoch % 100 == 0:
        epoch_counter.append(epoch); loss_values.append(loss); test_loss_values.append(test_loss)

print(f"\nPredicted y:\n{y_pred}")
print(f"\nIdeal weight value: {weight}, and bias value: {bias}")
print(f"Predicted weight and bias: {model_0.state_dict()}\n")
print(f"Current epoch: {epoch_counter}\nTraining loss values: {loss_values}\nTest loss values: {test_loss_values}")

plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=test_pred)

plt.figure(figsize=(10,7))
plt.plot(epoch_counter, np.array(torch.tensor(loss_values).detach().cpu().numpy()),label="Train loss")
plt.plot(epoch_counter, np.array(torch.tensor(test_loss_values).detach().cpu().numpy()), label="Test loss")
plt.title("Training and test loss curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()