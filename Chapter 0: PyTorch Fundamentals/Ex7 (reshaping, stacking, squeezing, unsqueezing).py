import torch

if torch.cuda.is_available():
    print(f"Device ID: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}\n")
    device = 'cuda'
else:
    device = 'cpu'

# Reshaping - reshapes an input tensor to a defined shape
# View - return a view of an input tenosir of certain shape but keep the same memory as original tensor
# Stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack) // concatenation
# Squeezing - removes all '1' dimensions from a tensor
# Unsqueezing - add a '1' dimension to a target tensor
# Permute - return a view of the input with dimensions permuted (swapped) in a certain way

tensor_D = torch.arange(1.,10.)
print(f"Tensor D:\n{tensor_D}, shape of tensor: {tensor_D.shape}\n")

# add an extra dimension
D_reshaped = tensor_D.reshape(3,3)
print(f"Tensor D reshaped:\n{D_reshaped.shape}\n")

# change the view
tensor_F = tensor_D.view(3,3) # tensor_F it's just another view of tensor_D and shares the same memory with it
print(f"\nTensor F:\n{tensor_F}\nchanging tensor F changes tensor D:")               # changing tensor_F changes tensor_D

tensor_F[:,0] = 888
print(f"Tensor F:\n{tensor_F}\n")
print(f"Tensor D:\n{tensor_D}\n")

# stack tensors on top of each other
tensor_stacked = torch.stack([tensor_D,tensor_D,tensor_D,tensor_D],dim=1) # 0 - is vstack, 1 - is hstack
print(f"Stacked tensor (x3 tensor D stacked on each other):\n{tensor_stacked}\n")

D_reshaped = tensor_D.reshape(1,9)
print(f"Tensor D reshaped:\n{D_reshaped.shape}\n")

# squeeze for removing a single dimension
squeeze_D = D_reshaped.squeeze()
print(f"Shape of squeezed tensor D: {squeeze_D.shape}\n")

# unsqueeze for adding a single dimension
unsqueeze_D = squeeze_D.unsqueeze(dim=0)

print(f"Tensor D squeezed:\n{squeeze_D}\n")
print(f"Unsqueezed tensor D:\n{unsqueeze_D}\n")
print(f"Shape of unsqueezed tensor D: {unsqueeze_D.shape}\n")

# permute for rearraging the dimensions of a target tensor in a specified order
x = torch.rand(12,3,5)
print(f"Shape of random tensor X: {x.shape}\n")
print(f"Permuted tensor X:\n{torch.permute(x, (2,0,1)).shape}\n")

x_perm = x.permute(2,0,1) # permute is just another way to represent a view of tensor x_original[0,0,0] = x_perm[0,0,0]
print(f"Shape of permuted tensor X: {x_perm.shape}\n")