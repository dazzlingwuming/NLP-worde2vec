#这是skip-gram的代码
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGram(nn.Module):
    def __init__(self,weight=None):
        super(SkipGram, self).__init__()
        self.in_embedding = nn.Embedding(1000, 64)  # 这个是目标词层，假设词汇表大小为1000，嵌入维度为64
        self.out_embedding = nn.Embedding(1000, 64) #这个是上下文词汇层，假设词汇表大小为1000，嵌入维度为64
        if weight is not None:
            self.in_embedding.weight = nn.Parameter(weight)
            self.out_embedding.weight = nn.Parameter(weight)
        #这里需要保证词向量组是一样的，所以可以传入一个权重参数weight来共享词向量组
    def forward(self, target, context,num_neg = 5):
        target_emb = self.in_embedding(target)#输入词向量
        context_emb = self.out_embedding(context)#目标词向量
        pos_scores = torch.sum(target_emb*context_emb,dim=1)#计算正样本的点积
        #生成负样本
        neg_sample = torch.randint(0, 1000, (target.size(0), num_neg), dtype=torch.long)
        for i , j  in enumerate(neg_sample):
            if j == context[i]:
                neg_sample[i] = torch.randint(0, 1000, (1,), dtype=torch.long)#这里需要保证负样本不等于正样本
        neg_emb = self.out_embedding(neg_sample)#负样本的词向量
        neg_scores = torch.sum(target_emb.unsqueeze(1) * neg_emb, dim=2) #计算负样本的点积
        pos_loss = F.logsigmoid(pos_scores).mean()
        neg_loss = F.logsigmoid(-neg_scores).sum(dim=1).mean()
        all_loss = -(pos_loss + neg_loss)
        return all_loss

if __name__ == "__main__":
    weight = torch.randn(1000, 64,requires_grad=True)# 假设词向量的权重是随机初始化的
    model = SkipGram(weight)
    target_input = torch.tensor([1,1,1,1,1,1,2,2,2,2],dtype=torch.long)  # 假设输入是一个批次大小为1，序列长度为10的整数张量
    context_input = torch.tensor([10, 3, 4, 5, 23, 45, 67, 88, 44, 34],dtype=torch.long)  # 假设上下文输入也是一个批次大小为1，序列长度为10的整数张量
    loss = model(target_input, context_input)
    print(loss.shape)  # 输出形状应该是 (10)
    #进行训练
    loss.backward()  # 计算梯度
