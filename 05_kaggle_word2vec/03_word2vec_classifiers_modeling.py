# 在word2vec上训练情感分析模型
import os
import re
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

from nltk.corpus import stopwords

from gensim.models.word2vec import Word2Vec

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import pickle


# 加载数据
def load_dataset(name, nrows=None):
    datasets = {
        'unlabeled_train': 'unlabeledTrainData.tsv',
        'labeled_train': 'labeledTrainData.tsv',
        'test': 'testData.tsv'
    }
    if name not in datasets:
        raise ValueError(name)
    data_file = os.path.join('../in_data/kaggle_word2vec_data', datasets[name])
    df = pd.read_csv(data_file, sep='\t', escapechar='\\', nrows=nrows)
    print('Number of reviews: {}'.format(len(df)))
    return df


# 预处理数据
def clean_text(text, remove_stopwords=False):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    if remove_stopwords:
        eng_stopwords = {}.fromkeys(
            [line.rstrip() for line in open('../in_data/kaggle_word2vec_data/en_stopwords.txt')])
        words = [w for w in words if w not in eng_stopwords]
    return words


def to_review_vector(review):
    words = clean_text(review, remove_stopwords=True)
    array = np.array([model[w] for w in words if w in model])
    return pd.Series(array.mean(axis=0))


def make_cluster_bag(review):
    words = clean_text(review, remove_stopwords=True)
    return (pd.Series([word_centroid_map[w] for w in words if w in wordset])
        .value_counts()
        .reindex(range(num_clusters + 1), fill_value=0))


if __name__ == "__main__":
    # 读入之前训练好的Word2Vec模型
    # path = "../out_data/kaggle_word2vec_models/kaggle_w2v.model"
    model = Word2Vec.load("../out_data/kaggle_word2vec_models/kaggle_w2v.model")
    # 根据word2vec的结果去对影评文本进行编码
    # 编码方式有一点粗暴，简单说来就是把这句话中的词的词向量做平均
    df = load_dataset('labeled_train')
    train_data_features = df.review.apply(to_review_vector)

    # 用随机森林构建分类器
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest = forest.fit(train_data_features, df.sentiment)
    # 在训练集上试试，确保模型能正常work
    confusion_matrix(df.sentiment, forest.predict(train_data_features))
    # 预测测试集结果
    df = load_dataset('test')
    test_data_features = df.review.apply(to_review_vector)
    result = forest.predict(test_data_features)
    output = pd.DataFrame({'id': df.id, 'sentiment': result})
    output.to_csv(os.path.join('../out_data/kaggle_word2vec_models', 'Word2Vec_model.csv'), index=False)
    print("使用随机森林预测结果写入  ../out_data/kaggle_word2vec_models/Word2Vec_model.csv .......")

    # 以下：kMeans对词向量进行聚类研究和编码
    word_vectors = model.wv.vectors
    num_clusters = word_vectors.shape[0] // 10
    kmeans_clustering = KMeans(n_clusters=num_clusters, n_jobs=4)
    idx = kmeans_clustering.fit_predict(word_vectors)
    word_centroid_map = dict(zip(model.wv.index2word, idx))
    filename = 'word_centroid_map_10avg.pickle'
    with open(os.path.join('../out_data/kaggle_word2vec_models', filename), 'bw') as f:
        pickle.dump(word_centroid_map, f)
    # 输出一些clusters看
    for cluster in range(0, 10):
        print("\nCluster %d" % cluster)
        print([w for w, c in word_centroid_map.items() if c == cluster])
    # 评论数据转成cluster bag vectors
    wordset = set(word_centroid_map.keys())
    df = load_dataset('labeled_train')
    train_data_features = df.review.apply(make_cluster_bag)

    # 再用随机森林算法建模
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest = forest.fit(train_data_features, df.sentiment)
    confusion_matrix(df.sentiment, forest.predict(train_data_features))
    df = load_dataset('test')
    test_data_features = df.review.apply(make_cluster_bag)
    result = forest.predict(test_data_features)
    output = pd.DataFrame({'id': df.id, 'sentiment': result})
    output.to_csv(os.path.join('../out_data/kaggle_word2vec_models', 'Word2Vec_BagOfClusters.csv'), index=False)
    print("Kmeans结合随机森林预测结果写入  ../out_data/kaggle_word2vec_models/Word2Vec_BagOfClusters.csv .......")
