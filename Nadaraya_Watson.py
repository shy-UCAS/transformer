#用于展示注意力汇聚的例子 illustrate of attention mechanism in Nadaraya-Watson kernel regression.
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import numpy as np
torch.set_printoptions(threshold=np.inf)

n_train = 50
# 生成随机样本
# 返回的第一个参数是排序后的数组，第二个是索引
x_train,x_list = torch.sort(torch.rand(n_train) * 5)
print("x_train:",x_train)
print("x_list:",x_list)

def f(x):
    return 2 * torch.sin(x) + x ** 0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
x_test = torch.arange(0, 5, 0.1)
y_truth = f(x_test)
n_test = len(x_test)


def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)
    d2l.plt.show()

# 平均汇聚
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
# plot_kernel_reg(y_hat)

# 非参数注意力汇聚
x_repeat= x_test.repeat_interleave(n_train).reshape(-1,n_train)
print("x repeat shape",x_repeat.shape)
print("x repeat",x_repeat)

attention_weight = nn.functional.softmax(-(x_repeat-x_train)**2/2,dim=1)
rows = torch.argsort(attention_weight,dim=1,descending=True)
print("attention weight",attention_weight)
print("rows",rows)