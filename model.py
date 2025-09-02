import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ultralytics.utils.torch_utils import model_info


def random_neg_labels(pos_labels, num_neg_labels, all_labels=None, negative_over_position = False, all_label_weigths = None):
    """
    生成负样本标签
    :param pos_labels: 正样本标签
    :param neg_labels: 需要负样本标签数量
    :param all_labels: 所有的标签
    :param all_label_weigths: 所有标签的权重
    :param negative_over_position: 负样本数量是否加上正样本数量
    :return: 产生负样本标签
    """
    pos_labels = np.array(pos_labels).reshape(-1)
    random_labels = np.random.choice(
        all_labels,
        size= num_neg_labels + (len(pos_labels) if negative_over_position else 0),#这里是保证即使取到了正样本标签，也能保证负样本标签的数量
        p = all_label_weigths
    )
    negative_labels = []
    for label in random_labels:
        if label not in pos_labels:
            negative_labels.append(label)
        if len(negative_labels) == num_neg_labels:
            break
    return negative_labels

class CBOWModel(nn.Module):
    def __init__(self,num_embeddings=1000, embedding_dim=64 , k = 5, all_label_weights= None):
        """
        CBOW模型的初始化
        :param num_embeddings: 词汇表大小
        :param embedding_dim: 映射维度大小
        :param k: 负样本数量
        :param all_label_wegihts:所有的标签权重
        这里的权重是为了处理类别不平衡的问题
        """
        super(CBOWModel, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim), requires_grad=True)
        #正态分布初始化
        nn.init.normal_(self.weight)
        self.embedding.weight = self.weight
        self.k = k
        #使用交叉熵损失函数
        self.loss = nn.CrossEntropyLoss()
        self.all_labels = np.arange(num_embeddings)#构建所有的标签的数组
        self.all_label_weights = all_label_weights
        self.negative_over_position = False #产生负样本的过程中，负样本的数量是否加上正样本的数量

    def forward(self, x, pos_labels):
        """
        前向传播
        :param x: 输入的词索引，形状为 (batch_size, context_size)批次和窗口大小
        :param pos_labels: 正样本标签，形状为 (batch_size,)这就是N个样本的中心词
        中国有很多经典的古诗词，假设窗口大小是5，那么就中心词就是古诗词中的每一个词，x就是每一个词的前后各两个词
        :return: 损失值
        """
        x = self.embedding(x)#将词索引映射为词向量，形状为 (batch_size, context_size, embedding_dim)
        x = x.mean(dim=1)  # 对上下文词向量取平均，形状为 (batch_size, embedding_dim)
        pos_emb = self.embedding(pos_labels)  # 获取正样本词向量，形状为 (batch_size, embedding_dim)
        #负采样
        #提取正样本标签
        pos_labels_np = pos_labels.cpu().numpy()
        #随机产生k个负样本标签
        neg_labes = random_neg_labels(pos_labels,
                                      num_neg_labels = self.k ,
                                      all_labels=self.all_labels,
                                      all_label_weigths=self.all_label_weights,
                                      negative_over_position=self.negative_over_position
                                      )
        #负样本的形状是 (batch_size, k)
        neg_labes = torch.tensor(neg_labes, dtype=torch.long, device=x.device)  # 转为Tensor
        #从参数中提取负样本词向量
        neg_emb = self.embedding(neg_labes)
        #计算正样本得分
        pos_scores = torch.sum(x * pos_emb, dim=1, keepdim=True)  #[N,E] * [N,E] -> [N,1]
        #计算负样本得分
        neg_scores = torch.matmul(x, neg_emb.t())  # 形状为 (batch_size, k) #[N,E] * [E,k] -> [N,k]
        #将正样本得分和负样本得分拼接在一起
        all_scores = torch.cat([pos_scores, neg_scores], dim=1)  # [N, 1+k]
        #构建标签
        target = torch.zeros(all_scores.size(0), dtype=torch.long, device=all_scores.device)
        #计算损失
        loss = self.loss(all_scores, target)
        return loss
    #这里是通过负采样减少了输出投影层的计算复杂度

class SkipGramModel(nn.Module):
    def __init__(self, num_embeddings=1000, embedding_dim=64, k=8, all_label_weights=None):
        """
        SkipGram模型的初始化
        :param num_embeddings: 词汇表大小
        :param embedding_dim: 映射维度大小
        :param k: 负样本数量
        :param all_label_wegihts:所有的标签权重
        这里的权重是为了处理类别不平衡的问题
        """
        super(SkipGramModel, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # 初始化：服从均值0、标准差0.01的正态分布（Word2Vec常用初始化）
        nn.init.normal_(self.embedding.weight, mean=0, std=0.01)
        self.k = k
        # 使用交叉熵损失函数
        self.loss = nn.CrossEntropyLoss()
        self.all_labels = np.arange(num_embeddings)  # 构建所有的标签的数组
        self.all_label_weights = all_label_weights
        self.negative_over_position = False  # 产生负样本的过程中，负样本的数量是否加上正样本的数量

    def forward(self, center_words, context_words):
        """
        前向传播
        :param center_words: 中心词索引，形状(batch_size,)
        :param context_words: 正样本上下文词索引，形状(batch_size,)
        中国有很多经典的古诗词，假设窗口大小是5，那么就中心词就是古诗词中的每一个词，x就是每一个词的前后各两个词
        :return: 损失值
        """
        # 从同一嵌入矩阵中提取中心词和上下文词向量
        center_emb = self.embedding(center_words)  # (batch_size, embedding_dim)
        context_emb = self.embedding(context_words)  # (batch_size, context_size, embedding_dim)#这里存在两种不同的上下文词向量，一种是多个词，一种是一个词，一个词的大小是(batch_size, embedding_dim)

        # 负采样
        # 提取正样本标签
        pos_labels_np = context_words.cpu().numpy().reshape(-1) # 展平为(batch_size*window_size,)
        # 随机产生k个负样本标签
        neg_labes = random_neg_labels(pos_labels_np,
                                      num_neg_labels=self.k,
                                      all_labels=self.all_labels,
                                      all_label_weigths=self.all_label_weights,
                                      negative_over_position=self.negative_over_position
                                      )
        # 负样本的形状是 (batch_size*window_size, k)
        neg_labes = torch.tensor(neg_labes, dtype=torch.long, device=center_words.device)  # 转为Tensor

        # 从参数中提取负样本词向量
        neg_emb = self.embedding(neg_labes)# (batch_size*window_size, k, embedding_dim)

        #计算正样本得分
        pos_scores = torch.sum(center_emb.unsqueeze(1) * context_emb, dim=-1,keepdim=True)#首先将中心词扩展为(batch_size, 1, embedding_dim)，然后与上下文词向量逐元素相乘并在embedding_dim维度上求和，得到(batch_size, context_size)
        N , M ,_ = pos_scores.shape# N是批次大小，M是上下文窗口大小，3,4
        # 计算负样本得分
        neg_scores = torch.matmul(center_emb, neg_emb.t())  # 形状为 (batch_size, k) #[N,E] * [E,k] -> [N,k]（3,8）
        #将负样本得分调整为 (N, M, k)
        neg_scores = torch.tile(neg_scores.unsqueeze(1), (1, M, 1))# (N, k) -> (N, M, k)（3,4,8）
        # 这里使用tile函数将负样本得分沿着新的维度M进行复制，以匹配正样本得分的形状
        # 将正样本得分和负样本得分拼接在一起
        all_scores = torch.concat([pos_scores, neg_scores], dim=-1)  # [N,M, 1+k]
        all_scores = all_scores.transpose(1,2) #(N, 1+k, M)
        # 构建标签
        target = torch.zeros((N ,M), dtype=torch.long, device=all_scores.device)#假设正确标签值是0
        # 计算损失
        loss = self.loss(all_scores, target)
        return loss

if __name__ == '__main__':
    x = torch.tensor([[1, 2, 3,4], [4, 5, 6,5], [7, 8, 9,12]])
    y = torch.tensor([10, 20, 30])
    # model = CBOWModel(100, 4)
    # output = model(x, y)
    model = SkipGramModel(100, 4)
    output = model(y , x)
    print(output)

