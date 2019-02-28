import os
import re
import pandas as pd
from bs4 import BeautifulSoup
import nltk.data
from gensim.models.word2vec import Word2Vec
import logging


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


def split_sentences(review):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = [clean_text(s) for s in raw_sentences if s]
    return sentences


def train_model():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # 设定词向量训练的参数
    num_features = 300  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    # model_name = '{}features_{}minwords_{}context.model'.format(num_features, min_word_count, context)
    model_name = "kaggle_w2v.model"
    print('Training model...')

    model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context,
                     sample=downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model.save(os.path.join('../out_data/kaggle_word2vec_models', model_name))
    print("save model....successful.")


# 训练结果检验
def pridict_model():
    # 加载模型
    model = Word2Vec.load("../out_data/kaggle_word2vec_models/kaggle_w2v.model")
    print(model.doesnt_match("man woman child kitchen".split()))
    print(model.doesnt_match('france england germany berlin'.split()))
    print(model.most_similar("man"))
    print(model.most_similar("queen"))


if __name__ == "__main__":
    # 读入无标签数据,用于训练生成word2vec词向量
    df = load_dataset('unlabeled_train')
    sentences = sum(df.review.apply(split_sentences), [])
    print('{} reviews -> {} sentences'.format(len(df), len(sentences)))
    # 训练模型
    train_model()
    # 看训练的词向量结果如何
    pridict_model()
