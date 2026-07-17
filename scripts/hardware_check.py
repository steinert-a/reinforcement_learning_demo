import torch

cuda_available = torch.cuda.is_available()

if cuda_available:
    print("CUDA available")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("!! CUDA is not available !!")
