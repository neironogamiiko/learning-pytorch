import torch
import os

# the easiest way - Google Colab
# use your own GPU
# use cloud computing: GCP, AWS, Azure

if torch.cuda.is_available():
    print(f"Device ID: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}\n")
    device = 'cuda'
else:
    device = 'cpu'

os.system("nvidia-smi")

tensor_cpu = torch.tensor([1,2,3])
print(f"Tensor's device: {tensor_cpu.device}")

tensor_gpu = tensor_cpu.to(device)
print(f"Tensor's device: {tensor_gpu.device}")