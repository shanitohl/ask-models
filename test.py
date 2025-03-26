import torch
print(f"¿CUDA disponible?: {torch.cuda.is_available()}")
print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
print(f"Versión CUDA: {torch.version.cuda}")