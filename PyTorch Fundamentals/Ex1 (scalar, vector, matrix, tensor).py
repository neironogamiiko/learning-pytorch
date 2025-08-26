import cv2; import torch
import numpy as np

print(f"\nCUDA available: {torch.cuda.is_available()}")
print(f"Device ID: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}\n")
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# scalar
scalar = torch.tensor(7)
print(f"Scalar is: {scalar}")
print(f"Scalar number of dimensions: {scalar.ndim}"); print(f"Scalar value: {scalar.item()}")
print(f"Shape of scalar: {scalar.shape}")

print("**************************************\n")

# vector
vector = torch.tensor([7,7])
print(f"Vector is: {vector}"); print(f"Vector number of dimensions: {vector.ndim}")
print(f"Shape of vector: {vector.shape}")

print("**************************************\n")

# MATRIX
MATRIX = torch.tensor([[7, 8],
                       [9,10]])

print(f"Matrix is:\n{MATRIX}"); print(f"Matrix number of dimensions: {MATRIX.ndim}")
print(f"Shape of matrix: {MATRIX.shape}")

print("**************************************\n")

# TENSOR
TENSOR = torch.tensor([[[1, 2, 3],
                        [4,5,6],
                        [7,8,9]]])

print(f"Tensor is:\n{TENSOR}"); print(f"Tensor number of dimensions: {TENSOR.ndim}")
print(f"Shape of tensor: {TENSOR.shape}")

print("**************************************\n")

# Random tensors
random_TENSOR = torch.rand(1, 3, 4)
print(random_TENSOR)
# Random tensors with similar shape to an image tensor
random_image_size_TENSOR = torch.rand(size=(3, 224, 224))
print(random_image_size_TENSOR.shape, random_image_size_TENSOR.ndim)