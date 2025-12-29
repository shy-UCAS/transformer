'''卷积神经网络示例'''
import torch
from torch import nn
from d2l import torch as d2l


def corr2d(X, K):
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()  
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias
    
def corr2d_multi_in(X, K):
    '''输入通道互相关运算'''
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
    
def corr2d_multi_in_out(X, k):
    '''实现一个计算多个通道的输出的互相关函数'''
    return torch.stack([corr2d_multi_in(X, K) for K in k], 0)

def corr2d_multi_in_out_1x1(X, K):
    '''使用全连接层实现1*1卷积'''
    c_in, h ,w = X.shape
    c_out = K.shape[0]
    X = X.reshape((c_in, h*w))
    K = K.reshape((c_out, c_in))
    Y = torch.matmul(K, X)
    return Y.reshape((c_out, h, w))

def pool2d(X, pool_size, mode = 'max'):
    '''汇聚操作'''
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0]-p_h+1, X.shape[1]-p_w+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i+p_h, j:j+p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i+p_h, j:j+p_w].mean()
    return Y



