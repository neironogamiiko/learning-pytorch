import torch; import os

if torch.cuda.is_available():
    print(f"Device ID: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}\n")
    device = 'cuda'
    
    os.system('nvidia-smi')
else:
    device = 'cpu'

# 2. Create a random tensor with shape (7, 7).
tensor_A = torch.rand(7, 7)
print(f"\nTensor A:\n{tensor_A}")

# 3. Perform a matrix multiplication on the tensor from 2 with another
# random tensor with shape (1, 7)
tensor_B = torch.rand(1,7)
print(f"\nTensor B:\n{tensor_B}")

mmAB = torch.matmul(tensor_A, tensor_B.T)
print(f"\nMultiplication result [7x7] * [1x7]:\n{mmAB}")

# 4.1. Set the random seed to 0 and do exercises 2 & 3 over again.
torch.manual_seed(0)
tensor_A = torch.rand(7,7)
print(f"\nTensor A:\n{tensor_A}")

tensor_B = torch.rand(1,7)
print(f"\nTensor B:\n{tensor_B}")

mmAB = torch.matmul(tensor_A, tensor_B.T)
print(f"\nMultiplication result with random seed 0:\n{mmAB}")

# 4.2. Set the random seed to 0 for both tensors and do exercises 2 & 3 over again.
torch.manual_seed(0)
tensor_A = torch.rand(7,7)
print(f"\nTensor A:\n{tensor_A}")

torch.manual_seed(0)
tensor_B = torch.rand(1,7)
print(f"\nTensor B:\n{tensor_B}")

mmAB = torch.mm(tensor_A, tensor_B.T)
print(f"\nMultiplication result with random seed 0 for both tensors:\n{mmAB}")

# 5. Speaking of random seeds, we saw how to set it with torch.manual_seed()
# but is there a GPU equivalent? If there is, set the GPU random seed to 1234.

torch.cuda.manual_seed(1234)

# 6. Create two random tensors of shape (2, 3) and send them both to the GPU
# Set torch.manual_seed(1234) when creating the tensors (this doesn't have to be the GPU random seed).
torch.manual_seed(1234)

tensor_A = torch.rand(2,3, device=device)
print(f"\nTensor A:\n{tensor_A}")

tensor_B = torch.rand(2,3, device=device)
print(f"\nTensor B:\n{tensor_B}")

# 7. Perform a matrix multiplication on the tensors you created in 6

tensor_C = tensor_A @ tensor_B.T
print(f"\nMultiplication result on cuda:\n{tensor_C}\n")

# 8. Find the maximum and minimum values of the output of 7.

maximum_C = torch.max(tensor_C)
print(f"Max value of tensor C: {maximum_C}")
minimum_C = torch.min(tensor_C)
print(f"Min value of tensor C: {minimum_C}")

# 9. Find the maximum and minimum index values of the output of 7.
max_index_C = torch.argmax(tensor_C)
min_index_C = torch.argmin(tensor_C)

print(f"Index of max tensor's value: {max_index_C}")
print(f"Index of min tensor's value: {min_index_C}")

# 10. Make a random tensor with shape (1, 1, 1, 10)

torch.cuda.manual_seed(7)
tensor_D = torch.rand(1,1,1,10, device=device)
print(f"\nTensor D:\n{tensor_D}\n")
# and then create a new tensor with all the 1 dimensions removed to be left
# with a tensor of shape (10).
tensor_D_squeezed = tensor_D.squeeze()
print(f"Squeezed tensor D device: {tensor_D_squeezed.device}\n")

# Set the seed to 7 when you create it and print out the first tensor
# and it's shape as well as the second tensor and it's shape.

print(f"Shape of tensor D: {tensor_D.shape}")
print(f"Shape of squeezed tensor: {tensor_D_squeezed.shape}")