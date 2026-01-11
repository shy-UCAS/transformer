import torch
from torch import nn
from d2l import torch as d2l
from RNN.text_preprocessing import load_data_time_machine, RNNModelScratch, train_ch8

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01
    
    def three_gate():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))
    
    W_xz, W_hz, b_z = three_gate()  # Update gate parameters
    W_xr, W_hr, b_r = three_gate()  # Reset gate parameters
    W_xh, W_hh, b_h = three_gate()  # Candidate hidden state parameters

    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # Collect all parameters  附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_gru_state(batch_size, num_hiddens, device):
    # 返回一个元组格式的数据
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)  # Update gate
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)  # Reset gate
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)  # Candidate hidden state
        H = Z * H + (1 - Z) * H_tilda  # New hidden state
        Y = (H @ W_hq) + b_q  # Output
        outputs.append(Y)
    return torch.stack(outputs), (H, )

if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
    num_epochs, lr = 500, 1 
    model = RNNModelScratch(len(vocab), num_hiddens, device,
                            get_params, init_gru_state, gru)
    train_ch8(model, train_iter, vocab, lr, num_epochs, device)