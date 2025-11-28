import numpy as np
import torch

# matrix1 = torch.arange(20).reshape(-1,2).T
# print(matrix1)

x = torch.arange(1, 5, requires_grad=True, dtype=torch.float32)
# print(x)

# z = x * x
# # 【关键修改】告诉 PyTorch保留 z 的梯度，不要释放它
# z.retain_grad() 

# y = torch.dot(z, z)
# y.backward()

# print(f"x.grad: {x.grad}")
# print(f"z.grad: {z.grad}") # 现在这里会有值了

# 分离计算
x.grad.zero_()
y = x * x
# 将u作为常数处理
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u