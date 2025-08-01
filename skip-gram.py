#这是skip-gram的代码
import numpy as np
import torch
import torch.nn as nn

class SkipGram(nn.Module):
    def __init__(self,weight=None):
        super(SkipGram, self).__init__()
        self.in_embedding = nn.Embedding(1000, 64)  # 这个是目标词层，假设词汇表大小为1000，嵌入维度为64
        self.out_embedding = nn.Embedding(1000, 64) #这个是上下文词汇层，假设词汇表大小为1000，嵌入维度为64
        if weight is not None:
            self.in_embedding.weight = nn.Parameter(weight)
            self.out_embedding.weight = nn.Parameter(weight)
        #这里需要保证词向量组是一样的，所以可以传入一个权重参数weight来共享词向量组
    def forward(self, target, context):
        target_emb = self.in_embedding(target)#输入词向量
        context_emb = self.out_embedding(context)#目标词向量
        scores = torch.sum(target_emb*context_emb,dim=1)#算出每一个样本的点积
        return scores

if __name__ == "__main__":
    weight = torch.randn(1000, 64,requires_grad=True)# 假设词向量的权重是随机初始化的
    model = SkipGram(weight)
    target_input = torch.tensor([1,1,1,1,1,1,2,2,2,2],dtype=torch.long)  # 假设输入是一个批次大小为1，序列长度为10的整数张量
    context_input = torch.tensor([10, 3, 4, 5, 23, 45, 67, 88, 44, 34],dtype=torch.long)  # 假设上下文输入也是一个批次大小为1，序列长度为10的整数张量
    output = model(target_input, context_input)
    print(output.shape)  # 输出形状应该是 (1, 10)