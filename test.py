#用于测试文件
if __name__ == "__main__":
    from gensim.models import FastText
    sentences = [["我", "爱", "自然", "语言"], ["处理", "文本", "数据"]]
    model = FastText(sentences, vector_size=100, window=5, min_count=1, min_n=3, max_n=6)
    print(model.wv["自然"])  # 获取 "自然" 的词向量
    print(model.wv["自然数据"])  # 即使未训练，也能生成向量
