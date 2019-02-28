import os
import re
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import nltk
from nltk.corpus import stopwords

def display(text, title):
    """

    :param text:
    :param title:
    :return:
    """
    print(title)
    print("\n----------我是分割线-------------\n")
    print(text)


def clean_text(text):
    """
    清洗文本数据
    # 对影评数据做预处理，大概有以下环节：
    # 去掉html标签
    # 移除标点
    # 切分成词 / token
    # 去掉停用词
    # 重组为新的句子
    :param text:
    :return:
    """
    # bs4中的BeautifulSoup，去掉HTML标签
    text = BeautifulSoup(text, 'html.parser').get_text()
    # 只保留英文字母
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    # 去掉停用词
    words = [w for w in words if w not in eng_stopwords]
    return ' '.join(words)

if __name__ == "__main__":
    # 用pandas读入训练数据
    datafile = os.path.join('../in_data/kaggle_word2vec_data', 'labeledTrainData.tsv')
    df = pd.read_csv(datafile, sep='\t', escapechar='\\')
    print('Number of reviews: {}'.format(len(df)))

    # 对影评数据做预处理，大概有以下环节：
    # 去掉html标签
    # 移除标点
    # 切分成词 / token
    # 去掉停用词
    # 重组为新的句子
    raw_example = df['review'][1]
    display(raw_example, '原始数据')

    # bs4中的BeautifulSoup，去掉HTML标签
    example = BeautifulSoup(raw_example, 'html.parser').get_text()
    display(example, '去掉HTML标签的数据')

    # 只保留英文字母
    example_letters = re.sub(r'[^a-zA-Z]', ' ', example)
    display(example_letters, '去掉标点的数据')

    words = example_letters.lower().split()
    display(words, '纯词列表数据')

    # 如果下载了nltk的停用词库，可直接使用下面这句，否则，自己提供一个
    # words_nostop = [w for w in words if w not in stopwords.words('english')]
    stopwords = {}.fromkeys([line.rstrip() for line in open('../in_data/kaggle_word2vec_data/en_stopwords.txt')])
    words_nostop = [w for w in words if w not in stopwords]
    display(words_nostop, '去掉停用词数据')

    # 去掉重复词
    # eng_stopwords = set(stopwords.words('english'))
    eng_stopwords = set(stopwords)

    # 以上这些，可直接调用clean_text()进行一次性处理
    df['cleaned_review'] = df.review.apply(clean_text)

    # 抽取bag of words特征(用sklearn的CountVectorizer),特征向量取5000，基本够用
    vectorizer = CountVectorizer(max_features=5000)
    train_data_features = vectorizer.fit_transform(df.cleaned_review).toarray()

    print("开始训练，需要一段时间...")
    # 训练分类器,随机森林
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data_features, df.sentiment)

    # 在训练集上做个predict看看效果如何
    print(confusion_matrix(df.sentiment, forest.predict(train_data_features)))

    # 读取测试数据进行预测
    datafile = os.path.join('../in_data/kaggle_word2vec_data', 'testData.tsv')
    df = pd.read_csv(datafile, sep='\t', escapechar='\\')
    print('Number of reviews: {}'.format(len(df)))
    df['cleaned_review'] = df.review.apply(clean_text)
    test_data_features = vectorizer.transform(df.cleaned_review).toarray()
    result = forest.predict(test_data_features)
    output = pd.DataFrame({'id': df.id, 'sentiment': result})
    display(output,"预测结果")
    output.to_csv(os.path.join('../out_data/bag_of_Words_model.csv'), index=False)
    print("写入out_data/bag_of_Words_model.csv")