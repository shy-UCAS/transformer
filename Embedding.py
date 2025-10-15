from torch import nn
from torch import Tensor
import torch
import torch.nn.functional as F
import math


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_dim):
        super(TokenEmbedding, self).__init__(vocab_size, embedding_dim, padding_idx=0)


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_len=512):
        super(PositionalEmbedding, self).__init__()
        self.encoding = torch.zeros(max_len, embedding_dim)
        # 位置编码不需要梯度
        self.encoding.require_grad = False
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        _2i = torch.arange(0, embedding_dim, 2, dtype=torch.float)
        self.encoding[:, 0::2] = torch.sin(position / (10000 ** (_2i / embedding_dim)))
        self.encoding[:, 1::2] = torch.cos(position / (10000 ** (_2i / embedding_dim)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len=512, drop_prob=0.1, device='cuda'):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEmbedding(embedding_dim, max_len)

    def forward(self, x):
        x = self.token_embedding(x) + self.position_embedding(x)
