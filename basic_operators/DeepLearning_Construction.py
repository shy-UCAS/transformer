import torch
from torch import nn
from torch.nn import functional as F

class MySequential(nn.Module):
    '''提供了与默认Sequential类相同的功能'''
    def __init__(self, *args):
        super(MySequential, self).__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module
    
    def forward(self, x):
        for block in self._modules.values():
            x = block(x)
        return x
    
# 使用MySequential类重新实现多层感知机
net = MySequential(nn.Linear(20,256), 
                   nn.ReLU(),
                   nn.Linear(256,10))


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units))
    
    def forward(self, X):
        linear = torch.mm(X, self.weight.data) + self.bias.data
        return F.relu(linear)