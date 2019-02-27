import jieba


def tokenize_chinese():
    """
    中文分词：使用jieba
    :return:
    """
    # 全模式
    seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
    print("Full Mode:\n", "/ ".join(seg_list))
    # 精确模式
    seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
    print("Default Mode:\n", "/ ".join(seg_list))
    # 默认是精确模式
    seg_list = jieba.cut("他来到了网易杭研大厦")
    print("默认是精确模式:\n", ", ".join(seg_list))

    # 搜索引擎模式
    seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
    print("搜索引擎模式:\n", ", ".join(seg_list))

    return None


if __name__ == "__main__":
    # 中文分词
    tokenize_chinese()
