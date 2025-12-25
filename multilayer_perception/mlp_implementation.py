import torch
import torchvision

from torch import nn
from d2l import torch as d2l
from torch.utils import data
from torchvision import transforms
from liner_nn.softmax_regression import Accumulator, train_ch3

def get_dataloader_workers():
    '''使用4个进程来读取数据'''
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    '''下载Fashion-MNIST数据集，然后加载到内存中'''
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)
    return(
        data.DataLoader(mnist_train, batch_size, shuffle=True,
                        num_workers=get_dataloader_workers()),
        data.DataLoader(mnist_test, batch_size, shuffle=False,
                        num_workers=get_dataloader_workers())
    )

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens,requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs,requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs,requires_grad=True))



loss = nn.CrossEntropyLoss(reduction="none")

num_epochs = 10
lr = 0.1
updater = torch.optim.SGD([W1, b1, W2, b2], lr=lr)

if __name__ == '__main__':
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    d2l.plt.show()


