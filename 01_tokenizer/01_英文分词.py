import nltk


def tokenize_english():
    """
    英文分词
    :return:
    """
    sentence = "hello, world"
    tokens = nltk.word_tokenize(sentence)
    print(tokens)
    return None


if __name__ == "__main__":
    #
    tokenize_english()
