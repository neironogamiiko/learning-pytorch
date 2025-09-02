import cv2; import torch
import numpy as np

print(f"\nCUDA available: {torch.cuda.is_available()}")
print(f"Device ID: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}\n")
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Note: Tensor datatypes is one of the 3 big errors with PyTorch and Deep Learning:
# 1. Tensor not right datatype. Can use tensor.dtype
# 2. Tensor not right shape. Can use tensor.shape
# 3. Tensor not on the right device. Cna use tensor.device

float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # what datatype is the tensor (e.g. float, uint8)
                               device=cuda, # what device is tensor on
                               requires_grad=False) # whether or not to track gradients with this tensors operations
print(f"Type: {float_32_tensor.dtype}")

float_16_tensor = float_32_tensor.type(torch.float16)
print(f"Type: {float_16_tensor.dtype}")

int_32_tensor = torch.tensor([3, 6, 9],
                             dtype=torch.int32,
                             device=cuda)

print(f"Multiplication result: {float_32_tensor * int_32_tensor}\n")

some_tensor = torch.rand(3,4)
print(f"Random tensor: \n{some_tensor}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Device tensor is on: {some_tensor.device}")