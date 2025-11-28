import torch
torch.zeros(3, 4)
torch.ones(3, 4)
torch.rand(3, 4)
torch.randn(3, 4)
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.tensor([[7, 8, 9], [10, 11, 12]])
print(x == y)