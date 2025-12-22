import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l
'''使用PyTorch的高级API更简洁地实现模型'''

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

net = nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
# 采用均方误差作为损失函数
loss = nn.MSELoss()
# 使用小批量随机梯度下降作为优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.01)

num_epochs = 3
batch_size = 10
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
data_iter = load_array((features, labels), batch_size)
for epoch in range(num_epochs):
    for x , y in data_iter:
        l = loss(net(x), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    L = loss(net(features), labels)
    with torch.no_grad():
        print(f'epoch {epoch + 1}, loss {float(L.mean()):f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)