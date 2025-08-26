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

tensor_C = torch.tensor([[1,2,3],
                         [1,2,3],
                         [1,2,3]])

tensor_D = torch.tensor([[4,5],
                         [5,6],
                         [1,3]])

print(f"\nMultiplication result:\n{torch.mm(tensor_C, tensor_D)}") # the same as torch.matmul or @

# To manipulate shape of one of the tensor - we can use transpose.
print(f"Tensor D transpose:\n{tensor_D.T}")

tensor_F = torch.tensor([[1,2],
                         [3,4],
                         [1,6]])

tensor_G = torch.tensor([[4,5],
                         [5,6],
                         [1,3]])

print(f"Original shape of tensor F: {tensor_F.shape}\nOriginal shape of tensor G: {tensor_G.shape}")
print(f"\nTensor G transpose:\n{tensor_G.T}\nand it's new shape: {tensor_G.T.shape}\n")
print(f"Multiplication result:\n{tensor_F @ tensor_G.T}")