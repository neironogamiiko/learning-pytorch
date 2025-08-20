import cv2; import torch
import numpy as np

print(f"\nCUDA available: {torch.cuda.is_available()}")
print(f"Device ID: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}\n")
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Element-wise multiplication
tensor = torch.tensor([1,2,3], device=cuda, dtype=torch.float32)
print(f"Element-wise multiplication: {tensor * tensor}\n")

# Matrix multiplication [n x m] * [m x n]
print(f"Matrix multiplication: {torch.matmul(tensor, tensor)}")
print(f"Also matrix multiplication: {tensor @ tensor}\n") # @ - it's the same as matmul

# Two main rules that performing matrix multiplication needs to satisfy:
# 1. The inner dimensions must match: [n x m] * [m x n]
#   (3,2) @ (2,3) - will work
#   (3,2) @ (3,2) - wouldn't work
# 2. The resulting matrix has the shape of tje outer dimensions:
#   (2,3) @ (3,2) -> (2,2)

# One of the most common errors in Deep Learning are shape errors.
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], device=cuda, dtype=torch.float32)

tensor_B = torch.tensor([[1, 2, 3],
                         [4, 5, 6]], device=cuda, dtype=torch.float32)

print(f"Matrix multiplicatin (3x2 and 2x3): \n{torch.matmul(tensor_A, tensor_B)}")
