import torch

def run_training(mode_name, use_zero_grad):
    print(f"\n[{mode_name}]")
    # 简单的线性模型 y = w * x
    # 输入 x=1, 目标 y=0
    # 初始 w=10
    # 损失 Loss = (wx - y)^2 = (w*1 - 0)^2 = w^2
    # 导数 dLoss/dw = 2w
    
    w = torch.tensor([10.0], requires_grad=True)
    x = torch.tensor([1.0])
    y_true = torch.tensor([0.0])
    
    lr = 0.1
    optimizer = torch.optim.SGD([w], lr=lr)
    
    print(f"初始权重 w: {w.item()}")
    
    for i in range(3):
        # 1. 前向传播
        y_pred = w * x
        loss = (y_pred - y_true) ** 2
        
        # 2. 清空梯度 (根据条件)
        if use_zero_grad:
            optimizer.zero_grad()
        
        # 3. 反向传播
        loss.backward()
        
        print(f"第 {i+1} 轮:")
        print(f"  计算出的梯度 (w.grad): {w.grad.item()}")
        
        # 4. 更新参数 w = w - lr * grad
        optimizer.step()
        print(f"  更新后的权重 w: {w.item()}")

# 正常情况：使用 zero_grad
run_training("正常模式 (使用 zero_grad)", use_zero_grad=True)

# 错误情况：不使用 zero_grad
run_training("错误模式 (不使用 zero_grad)", use_zero_grad=False)
