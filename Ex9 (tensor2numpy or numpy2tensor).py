import torch
import numpy as np

if torch.cuda.is_available():
    print(f"Device ID: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}\n")
    device = 'cuda'
else:
    device = 'cpu'

# NumPy -> PyTorch: torch.from_numpy(ndarray)
# PyTorch -> Numpy: torch.tensor.numpy()

# numpy array to tensor
print("NumPy array to PyTorch tensor:\n")
array = np.arange(1.,10.)
tensor = torch.from_numpy(array)
print(type(array), array, array.dtype)
print(tensor)
# tensor doesn't change if we change the original array

print("\nPyTorch tensor to NumPy array:\n")
# tensor to numpy array
tensor = torch.ones(10)
array = tensor.numpy()
print(tensor, tensor.dtype)
print(type(array), array, array.dtype)
# array doesn't change if we change the original tensor