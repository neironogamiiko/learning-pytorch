import cv2; import torch
import numpy as np

print(f"\nCUDA available: {torch.cuda.is_available()}")
print(f"Device ID: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}\n")
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

zeros = torch.zeros(3,4)
random_tensor = torch.rand(3,4)
ones = torch.ones(3,4)

print(f"Zeros:\n{zeros}\n")
print(f"Random:\n{random_tensor}\n")
print(f"Ones:\n{ones}\n")

# Range of tensors and tensors-like

range2ten = torch.arange(0, 11, 1) #start, end, step=1; [start; end) with step
print(f"Range:\n{range2ten}\n")

ten_zeroes = torch.zeros_like(input=range2ten)
print(f"Range:\n{ten_zeroes}\n")