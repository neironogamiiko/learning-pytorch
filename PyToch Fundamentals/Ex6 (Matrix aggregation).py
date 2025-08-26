import torch

print(f"\nCUDA available: {torch.cuda.is_available()}")
print(f"Device ID: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}\n")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tensor_A = torch.tensor([[1,2],
                         [3,4],
                         [1,6]])

tensor_B = torch.tensor([[4,5],
                         [5,6],
                         [1,3]])

tensor_C = tensor_A @ tensor_B.T

# Finding the min, max, mean, sum, etc
print(f"Min value of tensor C: {torch.min(tensor_C)}")
print(f"Max value of tensor C: {torch.max(tensor_C)}")
print(f"Mean value of tensor C: {torch.mean(tensor_C,dtype=torch.float32)}") # for mean valuer we always need float type
print(f"Mean value of tensor C: {torch.mean(tensor_C.type(torch.float32))}") # it's the same as torch.mean(tensor_C, dtype=torch.float32)
print(f"Sum of all elements: {torch.sum(tensor_C)}\n")

# Finding positional min and max (index of min/max value)
print(f"Tensor C:\n{tensor_C}\n")
print(f"Index of max value of tensor: {tensor_C.argmax()}") # find index of max value of tensor
print(f"Index of min value of tensor: {tensor_C.argmin()}") # find index of min value of tensor
