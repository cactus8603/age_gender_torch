import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# Print CUDA information
if cuda_available:
    cuda_device_count = torch.cuda.device_count()
    cuda_device_name = torch.cuda.get_device_name(0)
    print(f"CUDA is available.")
    print(f"Number of CUDA devices: {cuda_device_count}")
    print(f"Current CUDA device: {cuda_device_name}")
else:
    print("CUDA is not available.")
