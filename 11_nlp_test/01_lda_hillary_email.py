"""
LDA模型应用：一眼看穿希拉里的邮件
"""
import pandas as pd
import re
from gensim import corpora, models, similarities
from nltk.corpus import stopwords


def clean_email_text(text):
    """
    文本预处理
    :param text:
    :return:
    """
    text = text.replace('\n', " ")  # 新行，我们是不需要的
    text = re.sub(r"-", " ", text)  # 把 "-" 的两个单词，分开。（比如：july-edu ==> july edu）
    text = re.sub(r"\d+/\d+/\d+", "", text)  # 日期，对主体模型没什么意义
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text)  # 时间，没意义
    text = re.sub(r"[\w]+@[\.\w]+", "", text)  # 邮件地址，没意义
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text)  # 网址，没意义
    pure_text = ''
    # 以防还有其他特殊字符（数字）等等，我们直接把他们loop一遍，过滤掉
    for letter in text:
        # 只留下字母和空格
        if letter.isalpha() or letter == ' ':
            pure_text += letter
    # 再把那些去除特殊字符后落单的单词，直接排除。
    # 我们就只剩下有意义的单词了。
    text = ' '.join(word for word in pure_text.split() if len(word) > 1)
    return text


class LdaHillaryEmail():
    """
    LDA模型构建,用Gensim
    """

    def __init__(self, doclist):
        self.doclist = doclist

    def _remove_stop_words(self,doclist):
        """
        移除停用词
        :return:
        """
        stoplist = stopwords.words("english")
        texts = [[word for word in doc.lower().split() if word not in stoplist] for doc in doclist]
        return texts

    def create_model(self):
        texts = self._remove_stop_words(self.doclist)
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)
        self.model = lda
        return lda

    def predict(self,text):
        texts = self._remove_stop_words(text)
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        return self.model.get_document_topics(corpus)

if __name__ == "__main__":
    # 读取数据，预处理
    df = pd.read_csv("./HillaryEmail_data/HillaryEmails.csv")
    # 原邮件数据中有很多Nan的值，直接扔了。
    df = df[['Id', 'ExtractedBodyText']].dropna()

    # 对文本进行预处理
    docs = df['ExtractedBodyText']
    docs = docs.apply(lambda s: clean_email_text(s))
    doclist = docs.values

    # 创建模型
    lda_model = LdaHillaryEmail(doclist)
    lda = lda_model.create_model()
    print(lda.print_topics(num_topics=20, num_words=5))

    # 预测一下
    print("预测：\n")
    doc_topics = lda_model.predict("To all the little girls watching...never doubt that you are valuable and powerful & deserving of every chance & opportunity in the world.")
    for doc_topic in doc_topics:
        print(doc_topic)