#这是有关COBW的代码建立
import torch
import numpy as np
from torch import nn


class COBW(nn.Module):
    def __init__(self):
        super (COBW, self).__init__()
        self.embedding = nn.Embedding(1000, 64)  # 假设词汇表大小为1000，嵌入维度为64
        self.out = nn.Linear(64, 1000)  # 输出层，假设词汇表大小为1000

    def forward(self, x):
        x = self.embedding(x)
        x= x.mean(dim=1)
        x = torch.relu(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    model = COBW()
    sample_input = torch.randint(0, 1000, (1, 10))  # 假设输入是一个批次大小为1，序列长度为10的整数张量
    output = model(sample_input)
    print(output.shape)  # 输出形状应该是 (1,1000)

