import torch
print(torch.cuda.is_available())  # should return True if GPU is available
print(torch.cuda.device_count())  # number of GPUs available
print(torch.cuda.get_device_name(0))  # name of the GPU (if available)
