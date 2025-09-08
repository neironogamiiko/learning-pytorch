import torch
from torch import nn, optim, inference_mode
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pathlib import Path; import requests

if torch.cuda.is_available():
    print(f"Current device name: {torch.cuda.get_device_name()}")
    print(f"Current device id: {torch.cuda.current_device()}")
    device = 'cuda'
else:
    device = 'cpu'
    print(f"Current device: {device}")

    
def accuracy_metric(y, y_prediction):
    correct = torch.eq(y, y_prediction).sum().item()
    accuracy = (correct/len(y_prediction))*100
    return accuracy

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
optimizer = optim.SGD(params=model.parameters(), lr=.01)

# Our model outputs are going to be raw **logits** (raw predictions)

# We convert these logits into predictions probabilities by passing them to som kind of activation function (e.g. sigmoid for binary classification abd softmax for multiclass classification).
# Then we can convert our model's predictions probabilities to prediction labels by either rounding them or taking the `argmax()`.

# # View the first 5 outputs of the forward pass on the test data.
# model.eval()
# with inference_mode():
#     y_logits = model(X_test.to(device))[:5]
# print(f"Logits:\n{y_logits}")
#
# # Use the sigmoid activation function on our model logits to turn them into prediction probabilities
# y_prediction_probabilities = torch.sigmoid(y_logits)
# print(f"Predictions probabilities (logits):\n{y_prediction_probabilities}")
#
#
# # Find predicted labels
# y_prediction = torch.round(y_prediction_probabilities)
# print(f"Predicted labels:\n{y_prediction}")

# # Full
# model.eval()
# with inference_mode():
#     y_prediction_labels = torch.round(torch.sigmoid(model(X_test.to(device))[:5]))

# print(torch.eq(y_prediction.squeeze(), y_prediction_labels.squeeze()))

torch.manual_seed(42); torch.cuda.manual_seed(42)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

epochs = 301
for epoch in range(epochs):
    model.train()
    # Forward pass. The model goes through all the training data once, performing it's `forward()` function calculations (`model(X_train)`).
    y_logits = model(X_train).squeeze()
    y_predictions = torch.round(torch.sigmoid(y_logits))
    # Calculate the loss. The model's outputs (predictions) are compared to the ground truth and evaluated tp see how wrong they are (`loss = loss_fn(y_pred,y_train)`)
    loss = loss_function(y_logits, y_train) # nn.BCEWithLogits expects raw logits as input
    accuracy = accuracy_metric(y_train, y_predictions)
    # Zero gradients. The optimizers gradients are set to zero (they are accumulated by default) so they can be recalculated for the specific training step (`optimizer.zero_grad()`)
    optimizer.zero_grad()
    # Backward pass. Computes the gradient of the loss with respect for every model parameters to be updated (each parameter with `requiers_grad=True`). This is known as backpropagation, hence "backwards" (`loss.backward()`)
    loss.backward()
    # Step the optimizer (gradient descent). Update the parameters with `requieres_grad=True`) with respect to the loss gradients in order to improve them (`optimizer.step()`).
    optimizer.step()

    # testing
    model.eval()
    with inference_mode():
        test_logits = model(X_test).squeeze()
        test_predictions = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_function(test_logits, y_test)
        test_accuracy = accuracy_metric(y_test, test_predictions)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.6f} | Accuracy: {accuracy:.2f}% | Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.2f}%")

# From the metrics it looks like our model isn't learning anything and just flipping the coin.
# To inspect it let's make some predictions and make them visual.

if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_functions.py", "wb") as file:
        file.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
plt.show()

# Improving model (from a model's perspective, 'cuse they deal directly with the model, rather than the data).
# 1. Add more layers: give model more chances to learn about patterns in the data.
# 2. Add more hidden units: go from 5 hidden units to 10 hidden units
# 3. Fit for longer (more epochs).
# 4. Changing the activation function.
# 5. Change the learning rate.
# 6. Change the loss function.

# Let's improve with adding more hidden units: 5->10
# Increase the number of layers: 2->3
# Increase the number of epochs: 300->3000

class CirclesModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

model_circles = CirclesModel().to(device)

loss_function_circles = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(params=model_circles.parameters(), lr=.1, momentum=.9)

torch.manual_seed(42); torch.cuda.manual_seed(42)

epochs = 3001
for epoch in range(epochs):
    model_circles.train()

    #forward
    y_logits = model_circles(X_train).squeeze()
    y_predictions = torch.round(torch.sigmoid(y_logits))

    # loss/accuracy
    loss = loss_function_circles(y_logits, y_train)
    accuracy = accuracy_metric(y_train, y_predictions)

    # optimizer zero grad
    optimizer.zero_grad()

    # loss backwards
    loss.backward()

    # optimizer step
    optimizer.step()

    # testing
    model.eval()
    with inference_mode():
        # forward
        test_logits = model_circles(X_test).squeeze()
        test_predictions = torch.round(torch.sigmoid(test_logits))

        # loss
        test_loss = loss_function_circles(test_logits, y_test)
        test_accuracy = accuracy_metric(y_test, test_predictions)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {accuracy:.2f}% | Test loss: {test_loss:.5f}, test accuracy: {test_accuracy:.2f}%")

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_circles, X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_circles, X_test, y_test)
plt.show()