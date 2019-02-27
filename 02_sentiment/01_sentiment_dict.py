import nltk

def get_score(words):
    """
    计算一句话的情感总分
    :param words:
    :return:
    """
    # 形成情感词典
    sentiment_dictionary = {}
    for line in open('../in_data/AFINN/AFINN-111.txt'):
        word, score = line.split('\t')
        sentiment_dictionary[word] = int(score)

    # 把这个打分表记录在⼀一个Dict上以后
    # 跑⼀一遍整个句句⼦子，把对应的值相加
    total_score = sum(sentiment_dictionary.get(word, 0) for word in words)
    # 有值就是Dict中的值，没有就是0,于是你就得到了了⼀一个 sentiment score
    return total_score


if __name__ == "__main__":
    text = "I like movies.If you are a scientist I hope you would reference the above paper if you use it in your paper."
    score = get_score(nltk.word_tokenize(text))
    print("score:" + str(score))
