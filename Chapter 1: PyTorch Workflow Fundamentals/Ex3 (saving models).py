import torch
from torch import nn
from pathlib import Path

# There are three main methods for saving and loading models
# 1. `torch.save()` - allows to save a PyTorch objects in Python's pickle format
# 2. `torch.load()` - allows to load a saved PyTorch objects
# 3. `torch.nn.Module.load_state_dict()` - allows to load a model's saved state dictionary

# Saving algorithm:

# 1. Create model's directory
MODEL_PATH = Path("/home/frasero/PycharmProjects/Models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create a model save path
MODEL_NAME = "SimpleLinearRegressionModel.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the module state_dicti
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)