from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer


def porter_stemmmer(s):
    """
    词干提取：PorterStemmer
    :param s:
    :return:
    """
    return PorterStemmer().stem(s)


def snowball_stemmer(s):
    """
    词干提取：SnowballStemmer
    :param s:
    :return:
    """
    return SnowballStemmer("english").stem(s)


def lancaster_stemmer(s):
    """
    词干提取：LancasterStemmer
    :param s:
    :return:
    """
    return LancasterStemmer().stem(s)


if __name__ == "__main__":
    words = ["maximum", "presumably", "multiply", "provision"]
    for word in words:
        print("原词：" + word)
        print("\tPorterStemmer:" + porter_stemmmer(word))
        print("\tSnowballStemmer:" + snowball_stemmer(word))
        print("\tLancasterStemmer:" + lancaster_stemmer(word))
