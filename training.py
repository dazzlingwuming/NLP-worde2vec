"""
模型训练相关代码
"""
import os
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CBOWModel , SkipGramModel
from data_building import Vocabulary , VocabDataset


def restore_network (model, save_path):
    """
    如果存在断点，则从断点继续训练
    :param model: 选择模型
    :param save_path:  模型保存路径
    这里假设模型保存的格式为 model_epoch_{epoch}.pth
    通过检查该目录下的文件，找到最新的模型文件，并加载到模型中
    :return: 模型加载的参数
    """
    if not os.path.exists(save_path):
        return
    model_files = [f for f in os.listdir(save_path) if f.startswith("model_epoch_") and f.endswith(".pth")]
    if not model_files:
        return
    model_files.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))  # 按照epoch排序
    latest_model_file = model_files[-1]
    model_path = os.path.join(save_path, latest_model_file)
    model.load_state_dict(torch.load(model_path))
    print(f"Restored model from {model_path}")
    pass

def train_model(vocab_file ,
                data_loader,
                num_epochs=5,
                learning_rate=0.001,
                device='cpu',
                cbow = False ,
                dataset_file= None,
                batch_size = 64,
                shuffle=True,
                num_workers = 0 ,
                collate_fn=None,
                save_interval=100,
                save_path="data/save"):
    #0.路径确定
    if save_path :
        os.makedirs(save_path, exist_ok=True)
    #1.数据构建
    vocab = Vocabulary.load(vocab_file)  # 假设词汇表已经保存为vocab.pkl
    dataset = VocabDataset(dataset_file)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size, #给定的批次大小
                            shuffle=shuffle , #是否打乱数据
                            num_workers=num_workers , #使用的子进程数
                            collate_fn=None #给定如何将多个样本合并为一个批次的函数
                            )

    #2.模型对象，损失函数，优化器
    word_freq = vocab.fetch_token_negative_frequency()
    word_freq = np.array(word_freq)
    model = SkipGramModel if not cbow else CBOWModel
    model = model(num_embeddings=vocab.vocab_size, embedding_dim=64, k=8, all_label_weights=word_freq)
    model.to(device)
    #如果训练中断，可以从断点继续训练
    restore_network (model, save_path)
    train_opt = optim.SGD(model.parameters(), lr=learning_rate)


    #3.模型训练
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (inputs, targets) in enumerate(dataloader):
            #前向传播
            loss  = model(inputs.to(device), targets.to(device))
            train_opt.zero_grad()
            loss.backward()
            train_opt.step()
            total_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item() :.4f}")
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), f"{save_path}/model_epoch_{epoch + 1}.pth")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {total_loss / len(dataloader) :.4f}")
    #4.保存模型
    torch.save(model.state_dict(), f"{save_path}/model_final.pth")
#         :param center_words: 目标词索引，形状为 (batch_size,)
#         :return: 损失值
#         """
    #5.保存单词映射和单词向量
    vocab.save(f"{save_path}/vocab.pkl")
    with open(f"{save_path}/embedding_table.pkl", "wb") as f:
        pickle.dump(model.weght.detach().cpu().numpy(), f)



if __name__ == "__main__":
    # 进行模型训练
    train_model("data/原始数据/vocab.pkl", None, num_epochs=1000, learning_rate=0.001, device='cpu', cbow=False ,dataset_file="data/训练数据/train.txt",save_interval=100,)

    pass
