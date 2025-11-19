#用于展示注意力汇聚的例子 illustrate of attention mechanism in Nadaraya-Watson kernel regression.
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import numpy as np
# torch.set_printoptions(threshold=np.inf)

n_train = 50
# 生成随机样本
# 返回的第一个参数是排序后的数组，第二个是索引
x_train,x_list = torch.sort(torch.rand(n_train) * 5)
print("x_train:",x_train)
print("x_list:",x_list)

def f(x):
    return 2 * torch.sin(x) + x ** 0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
x_test = torch.arange(0, 5, 0.01)
y_truth = f(x_test)
n_test = len(x_test)


def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)



# 平均汇聚
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
# plot_kernel_reg(y_hat)

# 非参数注意力汇聚
x_repeat= x_test.repeat_interleave(n_train).reshape(-1,n_train)
print("x repeat shape",x_repeat.shape)
print("x repeat",x_repeat)

attention_weight = nn.functional.softmax(-(x_repeat-x_train)**2/2,dim=1)
y_hat = torch.matmul(attention_weight,y_train)
plot_kernel_reg(y_hat)
print("attention weight",attention_weight)
print(f"shape of attention weight {attention_weight.unsqueeze(0).unsqueeze(0).shape}")

# 带参数的注意力汇聚
class NWKernelReg(nn.Module):
    def __init__(self, **kwargs):
        super(NWKernelReg, self).__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,),requires_grad=True))

        def forward(self, queries, keys, values):
            queries = queries.repeat_interleave(keys.shape[1]).reshape(-1,keys.shape[1])
            self.attention_weight = nn.functional.softmax(-((queries-keys)*self.w)**2/2,dim=1)

            return torch.bmm(self.attention_weight.unsqueeze(1),values.unsqueeze(-1).reshape(-1))
print("x shape",x_train.shape)        
x_tile = x_train.repeat((n_train,1))
y_tile = y_train.repeat((n_train,1))
print("x tile",x_tile)
print("y tile",y_tile)
# q取出非对角元素，k取出对角元素，v取出y_train
# keys = x_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))