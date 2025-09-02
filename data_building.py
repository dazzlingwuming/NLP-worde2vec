#建立文本数据集
import os
import pickle
from venv import create
import jieba
import torch
from networkx.algorithms.distance_measures import periphery
from pygments.lexer import words
from torchgen.api.dispatcher import jit_arguments
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

#加载jiaba词典
jieba_user_dict_path = os.path.join(os.path.dirname(__file__), 'data','jieba_dict.txt')
jieba.load_userdict(jieba_user_dict_path)
#文本分词
def split_text(text):
    #使用jieba分词
    return list(jieba.cut(text))
#读取文件内容，进行分词，并将分词结果写入新文件
def split_flie_content(read_flie, out_flie):
    """
    读取文件内容，进行分词，并将分词结果写入新文件
    :param read_flie: 输入文件路径
    :param out_flie: 输出文件路径
    :return: None
    """
    if read_flie is None or out_flie is None:
        raise ValueError("文件路径不能为空")
    if not os.path.exists(read_flie):
        raise ValueError("输入文件路径不存在")
    with open(read_flie, 'r', encoding='utf-8') as read_flier , open(out_flie, 'w', encoding='utf-8') as out_flier:
        for line in read_flier:
            line = line.strip()
            if len(line) <= 5: #忽略长度小于5的行
                continue
            #文本分词
            words = split_text(line)
            #按空格连接分词结果，并写入新文件
            out_flier.writelines(' '.join(words) + '\n')

#构建单词表
def build_vocab(file_path,min_freq,PAD_token="<PAD>",UNK_token="<UNK>"):
    """
    构建词汇表
    :param file_path: 分词后文件路径
    :param PAD_token: 用于填充的特殊标记
    :param UNK_token: 用于表示未知词的特殊标记
    :return: word2id, id2word, word_freq
    """
    #解析文件，统计词频
    words_freq = {}
    total_freq = 0
    with open(file_path , "r" , encoding="utf-8") as reader:
        for line in reader:
            line = line.strip().lower()
            words = line.split(" ")#这个是按空格分词的
            for word in words:
                if word in words_freq:
                    words_freq[word] += 1
                else:
                    words_freq[word] = 1
                total_freq += 1
    token2id = {PAD_token:0, UNK_token:1}
    id2token = {0:PAD_token, 1:UNK_token}
    token2freq = {}
    index = 2
    for word , freq in words_freq.items():
        if freq >= min_freq:
            token2id[word] = index
            id2token[index] = word
            token2freq[word] = freq
            index += 1
    return Vocabulary(token2id, id2token, token2freq , total_freq , PAD_token="<PAD>", UNK_token="<UNK>")
#词汇表类
class Vocabulary:
    def __init__(self,token2id, id2token, token2freq , total_freq , PAD_token="<PAD>", UNK_token= "<UNK>"):
        self.vocab_size = len(token2id) #词汇表大小
        self.token2id = token2id
        self.id2token = id2token
        self.token2freq = token2freq
        self.total_freq = total_freq
        self.PAD_token = PAD_token
        self.PAD_id = self.token2id[PAD_token]
        self.UNK_token = UNK_token
        self.UNK_id = self.token2id[UNK_token]

    def save(self , file_path):
        """
        保存词汇表到文件
        :param file_path: 文件路径
        :return: None
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb",) as writer:
            obj = {
                "vocab_size": self.vocab_size,
                "token2id": self.token2id,
                "id2token": self.id2token,
                "token2freq": self.token2freq,
                "total_freq": self.total_freq,
                "PAD_token": self.PAD_token,
                "UNK_token": self.UNK_token
            }
            pickle.dump(obj , writer)

    @staticmethod
    def load(file_path):
        """
        从文件加载词汇表
        :param file_path: 文件路径
        :return: Vocabulary对象
        """
        if not os.path.exists(file_path):
            raise ValueError("文件路径不存在")
        with open(file_path, "rb") as reader:
            obj = pickle.load(reader)
            return Vocabulary(
                obj["token2id"],
                obj["id2token"],
                obj["token2freq"],
                obj["total_freq"],
                obj["PAD_token"],
                obj["UNK_token"]
            )

    def token2id_way(self,token):
        """
        将单词转换为索引
        :param token: 单词
        :return: 索引
        """
        return self.token2id.get(token, self.UNK_id)#.get(token, self.UNK_id) 是字典的查找方法，如果 token 存在于字典中，返回对应的编号；如果不存在，则返回默认值 self.UNK_id。
    def id2token_way(self, idx):
        """
        将索引转换为单词
        :param idx:
        :return:
        """
        return self.id2token.get(idx, self.UNK_token)
    def token2freq_way(self,token):
        """
        将单词转换为词频
        :param token: 单词
        :return: 词频
        """
        return self.token2freq.get(token, "无")

    def fetch_token_negative_frequency(self):
        """
        获取词典中每个单词的负采样频率
        :param vocab: 词典
        :return: 负采样频率列表
        """
        freq = []
        for i in range(self.vocab_size):
            token = self.id2token_way(i)
            token_freq = self.token2freq_way(token)
            if token_freq == "无":
                token_freq = 1
            freq.append(token_freq ** 0.75)
        freq_sum = sum(freq)
        freq = [f / freq_sum for f in freq]

        return freq

#构建训练数据
def build_training_data(file_path , out_path , vocab , window_siaze = 5):
    """
    构建训练数据
    :param file_path: 输入文件路径
    :param out_path: 输出文件路径
    :param vocab: 词典
    :param window_siaze: 窗口大小
    :return:
    """
    os.makedirs(out_path, exist_ok=True)
    with open(file_path , "r", encoding="utf-8") as reader, open(out_path + "/train.txt", "w", encoding="utf-8") as writer:
        for sentence in reader:
            words = sentence.strip().split(" ")
            words = [str(vocab.token2id_way(word)) if word in vocab.token2id else str(vocab.UNK_id) for word in words]
            if len(words) < window_siaze:
                continue
            m1 = (window_siaze - 1) // 2
            m2 = (window_siaze - 1) - m1

            for i in range(m1 , len(words) - m2):
                target_word = words[i]
                context_words = words[i - m1:i] + words[i + 1:i + 1 + m2]
                writer.write(f"{' '.join(context_words)} {target_word}\n")

class VocabDataset(Dataset):
    def __init__(self , data_file ):
        super(VocabDataset, self).__init__()
        if not os.path.exists(data_file):
            raise ValueError("文件路径不存在")
        self.data = []
        with open(data_file , "r", encoding="utf-8") as reader:
            for line in reader:
                words = list(map(int, line.strip().split(" ")))
                self.data.append((words[:-1], words[-1]))#前面是上下文词，后面是目标词

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        peripheral_words, target_word = self.data[idx]
        return torch.tensor(peripheral_words), torch.tensor(target_word)#建议 tensor 化，因为 PyTorch 的 Dataset 和 DataLoader 需要返回张量（torch.Tensor）




if __name__ == "__main__":
    #测试代码
    # split_flie_content("data/原始数据/三体.txt", "data/分词后结果/三体_分词.txt")
    # vocab = build_vocab("data/分词后结果/三体_分词.txt" , min_freq=10 , PAD_token="<PAD>", UNK_token="<UNK>")
    # vocab.save("data/原始数据/vocab.pkl")
    # loaded_vocab = Vocabulary.load("data/原始数据/vocab.pkl")
    # print(vocab.token2id_way("叶文洁"))
    # print(vocab.token2freq_way("叶文洁"))
    # print(loaded_vocab.token2id_way("叶文洁"))
    # print(loaded_vocab.token2freq_way("叶文洁"))
    # build_training_data("data/分词后结果/三体_分词.txt", "data/训练数据", Vocabulary.load("data/原始数据/vocab.pkl"), window_siaze=5)
    dataset_word2vec = VocabDataset("data/训练数据/train.txt")
    print(len(dataset_word2vec), dataset_word2vec[4])
    pass

