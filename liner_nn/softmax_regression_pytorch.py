import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l
from softmax_regression import load_data_fashion_mnist
'''使用PyTorch的高级API更简洁地实现模型'''
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

# .apply深度优先遍历模型中的每一个子模块（Submodule），并将函数 fn 应用到每一个子模块上。
net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)