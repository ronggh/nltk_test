import nltk
from nltk.corpus import stopwords

if __name__ == "__main__":
    sentence = " Find the English stopwords below and/or follow the links to view our other language stop word lists."
    word_list = nltk.word_tokenize(sentence)
    # 过滤
    filtered_words = [word for word in word_list if word not in stopwords.words("english")]
    print("原句：" + sentence)
    print("分词并过滤掉stopwords:")
    print(filtered_words)
