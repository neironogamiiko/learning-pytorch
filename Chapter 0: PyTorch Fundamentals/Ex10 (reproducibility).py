import torch

if torch.cuda.is_available():
    print(f"Device ID: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}\n")
    device = 'cuda'
else:
    device = 'cpu'

# Trying to take random out of random

# how a neural network learns:
# start with random numbers -> tensor operations -> update random numbers to try
# and make hem better representation of data -> again -> again -> again . . .

# to reduce the randomness in neural networks and PyTorch comes the concept of a random seed

random_tensor_A = torch.rand(3,4)
random_tensor_B = torch.rand(3,4)

print(f"Random tensor A:\n{random_tensor_A}\n")
print(f"Random tensor B:\n{random_tensor_B}\n")

print(f"Tensor A the same as tensor B:\n{random_tensor_A == random_tensor_B}\n")

# random but reproducible tensor
# set the random seed
torch.manual_seed(42)

random_tensor_C = torch.rand(3,4)
random_tensor_D = torch.rand(3,4)

print(f"Random tensor C:\n{random_tensor_C}\n")
print(f"Random tensor D:\n{random_tensor_D}\n")

print(f"Tensor C the same as tensor D:\n{random_tensor_C == random_tensor_D}\n")

# random but reproducible tensor
# set the random seed
torch.manual_seed(42)
random_tensor_F = torch.rand(3,4)

torch.manual_seed(42)
random_tensor_G = torch.rand(3,4)

print(f"Random tensor F:\n{random_tensor_F}\n")
print(f"Random tensor G:\n{random_tensor_G}\n")

print(f"Tensor F the same as tensor G:\n{random_tensor_F == random_tensor_G}\n")

