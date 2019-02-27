from nltk.tokenize import word_tokenize
# 处理正则表达式
import re


# 如没有下载punkt，则需要 nltk.download('punkt')
# punkt中包含了很多预先训练好的tokenize模型

def net_language():
    """
    包含网络语言和表情符号
    :return:
    """
    tweet = "RT @angelababy: love you baby! :D http://ah.love #168cm"
    print("未处理的分词结果：")
    print(word_tokenize(tweet))

    return None


def tokenize(s, lowercase=False):
    """
    处理网络用语、表情符号
    :param s:
    :return:
    """

    # 表情符号
    emoticons_str = r"""
    (?:
    [:=;] # 眼睛
    [oO\-]? # ⿐鼻⼦子
    [D\)\]\(\]/\\OpP] # 嘴
    )"""

    # 正则表达式
    regex_str = [
        emoticons_str,
        r'<[^>]+>',  # HTML tags
        r'(?:@[\w_]+)',  # @某⼈人
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # 话题标签
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
        r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # 数字
        r"(?:[a-z][a-z'\-_]+[a-z])",  # 含有 - 和 ‘ 的单词
        r'(?:[\w_]+)',  # 其他
        r'(?:\S)'  # 其他
    ]
    tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
    emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)
    tokens = tokens_re.findall(s)

    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


def process_net_language():
    """
    使用正则表达式处理网络语言和表情包符号
    :return:
    """
    tweet = "RT @angelababy: love you baby! :D http://ah.love #168cm"
    tokens = tokenize(tweet)
    print("使用正则处理的分词结果:")
    print(tokens)

    return None


if __name__ == "__main__":
    # 未处理
    net_language()
    # 使用正则表达式处理
    process_net_language()
