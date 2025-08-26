import torch

if torch.cuda.is_available():
    print(f"Device ID: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}\n")
    device = 'cuda'
else:
    device = 'cpu'

x = torch.arange(1,10).reshape(1,3,3)
print(f"Tensor X:\n{x}\nand it's shape: {x.shape}")
print(f"X[0]: {x[0]}")
print(f"X[0,0]: {x[0,0]}") # the same as x[0][0]

# index of the most inner bracket (last dimension)
print(f"X[0,0,0]: {x[0,0,0]}")

# all values of 0th and 1st dimension but only index 1 of 2nd dimension
print(f"X[:,:,1]: {x[:,:,1]}") # column

# all values of the 0 dimension but only the 1 index value of 1st and 2nd dimension
print(f"X[:,1,1]: {x[:,1,1]}")

# get index 0 of 0th and 1st dimension and all values of 2nd dimension
print(f"X[0,0,:]: {x[0,0,:]}")