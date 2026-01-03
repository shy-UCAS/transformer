import torch
from torch import nn
from d2l import torch as d2l
from basic_operators.training_utils import *
net = nn.Sequential(
    nn.Conv2d(1,6,kernel_size=5,padding=2),nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Conv2d(6,16,kernel_size=5),nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Flatten(),
    nn.Linear(16*5*5,120),nn.ReLU(),
    nn.Linear(120,84),nn.ReLU(),
    nn.Linear(84,10)
    # 不需要手动softmax处理
)

batch_size = 2048*2
train_iter, test_iter = load_data_fashion_mnist(batch_size)
lr = 0.1
num_epochs = 30
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()