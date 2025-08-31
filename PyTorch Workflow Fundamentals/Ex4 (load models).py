import torch
from torch import nn
from pathlib import Path; import os

if torch.cuda.is_available():
    print(f"Device ID: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}\n")
    device = 'cuda'

    os.system('nvidia-smi')
else:
    device = 'cpu'

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

# Load model.state_dict()
MODEL_PATH = Path("/home/frasero/PycharmProjects/Models/SimpleLinearRegressionModel(state_dict).pth")

model = LinearRegressionModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

print(f"\n{model.state_dict()}\n")