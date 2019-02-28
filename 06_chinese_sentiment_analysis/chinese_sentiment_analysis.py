# 用gensim去做word2vec的处理，会用sklearn当中的SVM进行建模
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC
import sys


# 载入数据，做预处理(分词)，切分训练集与测试集
def load_file_and_preprocessing():
    # excel 需要pip install xlrd
    neg = pd.read_excel('../in_data/Chinese_sentiment_analysis_data/neg.xls', sheet_name='Sheet1', header=None,
                        index=None)
    pos = pd.read_excel('../in_data/Chinese_sentiment_analysis_data/pos.xls', sheet_name='Sheet1', header=None,
                        index=None)

    cw = lambda x: list(jieba.cut(x))
    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)

    # print pos['words']
    # use 1 for positive sentiment, 0 for negative
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)

    np.save('../out_data/Chinese_sentiment_analysis_data/y_train.npy', y_train)
    np.save('../out_data/Chinese_sentiment_analysis_data/y_test.npy', y_test)
    print("save train and test data...")
    return x_train, x_test


# 对每个句子的所有词向量取均值，来生成一个句子的vector
def build_sentence_vector(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# 计算词向量
def get_train_vecs(x_train, x_test):
    # 一般取维度300或500
    n_dim = 300
    # 初始化模型和词表
    imdb_w2v = Word2Vec(size=n_dim, min_count=10)
    imdb_w2v.build_vocab(x_train)

    print("在评论训练集上建模,可能会花费几分钟...")
    # 在评论训练集上建模(可能会花费几分钟)
    imdb_w2v.train(x_train,total_examples=imdb_w2v.corpus_count,epochs=imdb_w2v.iter)

    # 对每个句子的所有词向量取均值，来生成一个句子的vector
    train_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_train])
    # train_vecs = scale(train_vecs)

    np.save('../out_data/Chinese_sentiment_analysis_data/train_vecs.npy', train_vecs)
    # print(train_vecs.shape)
    # 在测试集上训练
    imdb_w2v.train(x_test,total_examples=imdb_w2v.corpus_count,epochs=imdb_w2v.iter)
    imdb_w2v.save('../out_data/Chinese_sentiment_analysis_data/w2v_model.pkl')
    # Build test tweet vectors then scale
    test_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_test])
    # test_vecs = scale(test_vecs)
    np.save('../out_data/Chinese_sentiment_analysis_data/test_vecs.npy', test_vecs)

    print("save w2v_model.pkl successful.")


# load data
def get_data():
    train_vecs = np.load('../out_data/Chinese_sentiment_analysis_data/train_vecs.npy')
    y_train = np.load('../out_data/Chinese_sentiment_analysis_data/y_train.npy')
    test_vecs = np.load('../out_data/Chinese_sentiment_analysis_data/test_vecs.npy')
    y_test = np.load('../out_data/Chinese_sentiment_analysis_data/y_test.npy')
    return train_vecs, y_train, test_vecs, y_test


# 训练svm模型
def svm_train(train_vecs, y_train, test_vecs, y_test):
    clf = SVC(kernel='rbf', verbose=True)
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, '../out_data/Chinese_sentiment_analysis_data/model.pkl')
    print(clf.score(test_vecs, y_test))


# 构建待预测句子的向量
def get_predict_vecs(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load('../out_data/Chinese_sentiment_analysis_data/w2v_model.pkl')
    # imdb_w2v.train(words)
    train_vecs = build_sentence_vector(words, n_dim, imdb_w2v)
    # print train_vecs.shape
    return train_vecs


# 对单个句子进行情感判断
def svm_predict(string):
    words = jieba.lcut(string)
    words_vecs = get_predict_vecs(words)
    clf = joblib.load('../out_data/Chinese_sentiment_analysis_data/model.pkl')

    result = clf.predict(words_vecs)

    if int(result[0]) == 1:
        print(string, ' positive')
    else:
        print(string, ' negative')


def prdict():
    ##对输入句子情感进行判断
    # string = '电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    string='牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    svm_predict(string)


if __name__ == "__main__":
    # 载入数据，做预处理(分词)，切分训练集与测试集
    x_train, x_test = load_file_and_preprocessing()
    # 计算词向量,
    get_train_vecs(x_train, x_test)
    # 获取数据
    train_vecs, y_train, test_vecs, y_test = get_data()
    # 训练模型
    svm_train(train_vecs,y_train,test_vecs,y_test)
    # 对输入句子情感进行判断
    prdict()