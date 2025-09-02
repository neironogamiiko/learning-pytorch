import torch
from torch import nn # nn contains all of python's building blocks for neural networks
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

what_where_covering = {1: 'Data (prepare and load)',
                       2: 'Build model',
                       3: 'Fitting the model to data (training)',
                       4: 'Making predictions and evaluating a model (inference)',
                       5: 'Save and load a model',
                       6: 'Putting it all together'}

# Create known parameters
weight = .7
bias = .3

start = 0
end = 1
step = .02

X =  torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * X + bias

# Splitting training and test sets
train = int(.8 * len(X))
X_train, y_train = X[:train], y[:train]
X_test, y_test = X[train:], y[train:]

# Visualize data
def plot_predictions(train_data,train_labels,test_data,test_labels,predictions=None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_labels, c='b', s=4, label='Training data')
    plt.scatter(test_data, test_labels, c='g', s=4, label='Test data')

    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')

    plt.legend(prop={"size":14})
    plt.show()

plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None)