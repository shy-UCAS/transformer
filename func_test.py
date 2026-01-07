import torch
from torch.nn import functional as F

chars = torch.tensor([1, 2, 3])

print(F.one_hot(chars, num_classes=10))  # 默认dtype=torch.int64