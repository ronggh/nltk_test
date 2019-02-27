import nltk
from nltk import FreqDist

def freq(s):
    """
    统计词频
    :param s:
    :return:
    """
    tokens = nltk.word_tokenize(s)
    # print(tokens)
    fdist = FreqDist(tokens)
    # is出现的次数
    # print(fdist['is'])
    # 把最常用的50个单词拿出来
    standard_freq_vector = fdist.most_common(50)

    return standard_freq_vector


def position_lookup(v):
    """
    按照出现频率⼤小, 记录下每⼀个单词的位置
    :param v:
    :return:
    """
    res = {}
    counter = 0
    for word in v:
        res[word[0]] = counter
        counter += 1
    return res


if __name__ == "__main__":
    # 做个简单的词库
    corpus = 'this is my sentence ' \
             'this is my life ' \
             'this is the day'
    standard_freq_vector = freq(corpus)
    print(standard_freq_vector)
    # 把标准的单词位置记录下来
    standard_position_dict = position_lookup(standard_freq_vector)
    # 得到一个位置对照表
    print(standard_position_dict)

    # 这时, 如果有个新句⼦:
    sentence = 'this is cool'
    # 先新建一个跟我们的标准vector同样⼤大小的向量
    size = len(standard_freq_vector)
    freq_vector = [0] * size
    # 简单的Preprocessing
    tokens = nltk.word_tokenize(sentence)
    # 对于这个新句子⾥的每一个单词
    for word in tokens:
        try:
            # 如果在我们的词库⾥里里出现过
            # 那么就在"标准位置"上+1
            freq_vector[standard_position_dict[word]] += 1
        except KeyError:
            # 如果是个新词
            # 就pass掉
            continue
    # 结果：[1, 1, 0, 0, 0, 0, 0]
    print(freq_vector)
