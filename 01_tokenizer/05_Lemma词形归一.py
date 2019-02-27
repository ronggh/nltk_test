import nltk
from nltk.stem import WordNetLemmatizer


# pos，指定词性，默认是名词
def wn_lemmatizer(s, pos="n"):
    """
    词形归一
    :param s:
    :return:
    """
    return WordNetLemmatizer().lemmatize(s, pos)


if __name__ == "__main__":
    words = ["dogs", "churches", "aardwolves", "abaci", "hardrock", "indexes", "data", "Went"]
    for word in words:
        print("原词：" + word)
        print("\t 归一化：" + wn_lemmatizer(word))

    # 使用pos tag
    words2 = ["went", "are", "is", "was", "were"]
    for word in words2:
        print("原词：" + word)
        print("\t 归一化：" + wn_lemmatizer(word, pos="v"))

    # pos tag
    text = nltk.word_tokenize('what does the fox say')
    pos_txt = nltk.pos_tag(text)
    print("原句分词：")
    print(text)
    print("打上Pos Tag:")
    print(pos_txt)
