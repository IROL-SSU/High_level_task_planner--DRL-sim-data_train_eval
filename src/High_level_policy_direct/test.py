import torch

# 1D 텐서
x = torch.tensor([0, 1, 2, 0, 3, 0, 4, 0], device="cuda")

# mask = x > 0
mask = torch.where(x > 0, x, torch.tensor(None, device="cuda"))

print(mask)