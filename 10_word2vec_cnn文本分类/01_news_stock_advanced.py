"""
用每日新闻预测金融市场变化（进阶版）
Kaggle竞赛：https://www.kaggle.com/aaron7sun/stocknews
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import date
from nltk.tokenize import word_tokenize
# 停止词
from nltk.corpus import stopwords
import re
# lemma
from nltk.stem import WordNetLemmatizer
from gensim.models.word2vec import Word2Vec

from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from keras.layers import Convolution2D

# n_filter = 16
# filter_length = 4
# con2d = Convolution2D(n_filter,filter_length,filter_length,input_shape=(1,256, 128))

# 数字
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


# 特殊符号
def isSymbol(inputString):
    return bool(re.match(r'[^\w]', inputString))


def check(word):
    """
    如果需要这个单词，则True
    如果应该去除，则False
    """
    word = word.lower()
    stop = stopwords.words('english')
    if word in stop:
        return False
    elif hasNumbers(word) or isSymbol(word):
        return False
    else:
        return True


# 把上面的方法综合起来
def preprocessing(sen):
    res = []
    wordnet_lemmatizer = WordNetLemmatizer()
    for word in sen:
        if check(word):
            # 这一段的用处仅仅是去除python里面byte存str时候留下的标识。。之前数据没处理好，其他case里不会有这个情况
            word = word.lower().replace("b'", '').replace('b"', '').replace('"', '').replace("'", '')
            res.append(wordnet_lemmatizer.lemmatize(word))
    return res


def load_data():
    """
    加载数据，分割为训练集和测试集
    :return:
    """
    # 先读入数据。可以在 Kaggle竞赛：https://www.kaggle.com/aaron7sun/stocknews上下载
    data = pd.read_csv('./input_data/Combined_News_DJIA.csv')
    # 分割测试 / 训练集
    train = data[data['Date'] < '2015-01-01']
    test = data[data['Date'] > '2014-12-31']
    # 每条新闻做成一个单独的句子，集合在一起：
    x_train = train[train.columns[2:]]
    corpus = x_train.values.flatten().astype(str)
    x_train = x_train.values.astype(str)
    x_train = np.array([' '.join(x) for x in x_train])

    x_test = test[test.columns[2:]]
    x_test = x_test.values.astype(str)
    x_test = np.array([' '.join(x) for x in x_test])

    y_train = train['Label'].values
    y_test = test['Label'].values

    # tokenize
    corpus = [word_tokenize(x) for x in corpus]
    x_train = [word_tokenize(x) for x in x_train]
    x_test = [word_tokenize(x) for x in x_test]

    # 把三个数据组都来处理一下：
    corpus = [preprocessing(x) for x in corpus]
    x_train = [preprocessing(x) for x in x_train]
    x_test = [preprocessing(x) for x in x_test]

    return corpus, x_train, x_test, y_train, y_test


# 得到任意text的vector
def get_vector(model, vocab, word_list):
    # 建立一个全是0的array
    res = np.zeros([128])
    count = 0
    for word in word_list:
        if word in vocab:
            res += model[word]
            count += 1
    return res / count


def training_model(corpus, x_train, x_test):
    """
    训练模型
    :return:
    """
    model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=4)
    # 先拿到全部的vocabulary
    vocab = model.wv.vocab
    wordlist_train = x_train
    wordlist_test = x_test

    x_train = [get_vector(x) for x in x_train]
    x_test = [get_vector(x) for x in x_test]

    params = [0.1, 0.5, 1, 3, 5, 7, 10, 12, 16, 20, 25, 30, 35, 40]
    test_scores = []
    for param in params:
        clf = SVR(gamma=param)
        test_score = cross_val_score(clf, x_train, y_train, cv=3, scoring='roc_auc')
        test_scores.append(np.mean(test_score))

    return model, wordlist_train, wordlist_test


# 说明，对于每天的新闻，我们会考虑前256个单词。不够的我们用[000000]补上
# vec_size 指的是我们本身vector的size
def transform_to_matrix(model, x, padding_size=256, vec_size=128):
    res = []
    for sen in x:
        matrix = []
        for i in range(padding_size):
            try:
                matrix.append(model[sen[i]].tolist())
            except:
                # 这里有两种except情况，
                # 1. 这个单词找不到
                # 2. sen没那么长
                # 不管哪种情况，我们直接贴上全是0的vec
                matrix.append([0] * vec_size)
        res.append(matrix)
    return res


if __name__ == "__main__":
    # 加载数据
    corpus, x_train, x_test, y_train, y_test = load_data()
    # 训练模型
    model, wordlist_train, wordlist_test = training_model(corpus, x_train, x_test)

    x_train = transform_to_matrix(wordlist_train)
    x_test = transform_to_matrix(wordlist_test)

    # 搞成np的数组，便于处理
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])

