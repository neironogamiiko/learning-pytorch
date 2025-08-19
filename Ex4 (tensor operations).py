import cv2; import torch
import numpy as np

print(f"\nCUDA available: {torch.cuda.is_available()}")
print(f"Device ID: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}\n")
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Addition
tensor = torch.tensor([1,2,3])
tensor += 10
print(tensor)

# Multiplication
tensor *= 10
print(tensor)

# Substract
tensor -= 10
print(tensor)

tensor = torch.tensor([1,2,3])
tensor = torch.mul(tensor, 10) # torch multiplication
print(tensor)

tensor = torch.add(tensor, 10) # torch addition
print(tensor)