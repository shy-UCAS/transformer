import torch
n_train = 50
test = 1 - torch.eye(n_train)

x_train,x_list = torch.sort(torch.rand(n_train) * 5)
print(x_train.unsqueeze(0).shape)
print([[0.0]]*2)