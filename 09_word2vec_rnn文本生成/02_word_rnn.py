"""
用RNN做文本生成,不再用char级别，用word级别来做。
用温斯顿丘吉尔的人物传记作为我们的学习语料。
(各种中文语料可以自行网上查找， 英文的小说语料可以从古登堡计划网站下载txt平文本：
https://www.gutenberg.org/wiki/Category:Bookshelf)
实现的功能是：
    文本预测就是，给了前面的单词以后，下一个单词是谁？
        比如，hello from the other, 给出 side
先导入各种库
"""
import os
import numpy as np
import nltk
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from gensim.models.word2vec import Word2Vec


def predict_next(input_array):
    x = np.reshape(input_array, (-1, seq_length, 128))
    y = model.predict(x)
    return y


def string_to_index(raw_input):
    raw_input = raw_input.lower()
    input_stream = nltk.word_tokenize(raw_input)
    res = []
    for word in input_stream[(len(input_stream) - seq_length):]:
        res.append(w2v_model[word])
    return res


def y_to_word(y):
    word = w2v_model.most_similar(positive=y, topn=1)
    return word


def generate_article(init, rounds=30):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_word(predict_next(string_to_index(in_string)))
        in_string += ' ' + n[0][0]
    return in_string


if __name__ == "__main__":
    # 读入文本
    raw_text = ''
    for file in os.listdir("./input_data/"):
        if file.endswith(".txt"):
            raw_text += open("./input_data/" + file, errors='ignore',encoding='utf-8').read() + '\n\n'
    # raw_text = open('./input_data/Winston_Churchil.txt').read()
    raw_text = raw_text.lower()
    # 句子分割模式
    sentensor = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sentensor.tokenize(raw_text)

    # 再对句子分词
    corpus = []
    for sen in sents:
        corpus.append(nltk.word_tokenize(sen))

    # 这里打印,只是为了看一下
    print(len(corpus))
    print(corpus[:3])

    # 模型
    w2v_model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=4)

    # 处理training data，把源数据变成一个长长的x，好让LSTM学会predict下一个单词
    raw_input = [item for sublist in corpus for item in sublist]
    text_stream = []
    vocab = w2v_model.wv.vocab
    for word in raw_input:
        if word in vocab:
            text_stream.append(word)

    # 构造训练测试集，需要把raw_text变成可以用来训练的x, y:
    seq_length = 10
    x = []
    y = []
    for i in range(0, len(text_stream) - seq_length):
        given = text_stream[i:i + seq_length]
        predict = text_stream[i + seq_length]
        x.append(np.array([w2v_model[word] for word in given]))
        y.append(w2v_model[predict])

    x = np.reshape(x, (-1, seq_length, 128))
    y = np.reshape(y, (-1, 128))

    # 模型建造：LSTM模型构建
    model = Sequential()
    model.add(LSTM(256, dropout_W=0.2, dropout_U=0.2, input_shape=(seq_length, 128)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam')

    # 进行训练，要花费较长时间,可以调整epochs参数值可改进性能
    print("进行训练，要花费较长时间...")
    model.fit(x, y, epochs=20, batch_size=4096)

    # 保存模型：默认保存为h5格式，包括模型网络和权重
    # 如只保存模型的结构，而不包含其权重或配置信息，可用:JSON或YAML格式
    # json_string = model.to_json()
    # yaml_string = model.to_yaml()
    # 加载：
    #       from keras.models import model_from_json
    #       model = model_from_json(json_string)
    #       model reconstruction from YAML
    #       model = model_from_yaml(yaml_string)
    # 权重的保存和加载
    #       model.save_weights(path)
    #       model.load_weights(path)
    model.save("../out_data/word_rnn.h5")
    # 加载模型
    # model.load_model("../out_data/word_rnn.h5")

    # 检验模型
    init = 'Language Models allow us to measure how likely a sentence is, which is an important for Machine'
    article = generate_article(init)
    print(article)
