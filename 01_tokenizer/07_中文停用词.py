import jieba


# 创建停用词list
def stopwordslist(filepath):
    """
    创建停用词list
    :param filepath:
    :return:
    """
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# 对句子进行分词
def seg_sentence(sentence):
    """
    对句子进行分词
    :param sentence:
    :return:
    """
    sentence_seged = jieba.cut(sentence.strip())
    # 这里加载停用词的路径
    stopwords = stopwordslist('../in_data/stopwords.txt')
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr


if __name__ == "__main__":
    inputs = open('../in_data/input.txt', 'r', encoding='utf-8')
    outputs = open('../out_data/output.txt', mode='w', encoding="utf-8")
    for line in inputs:
        line_seg = seg_sentence(line)  # 这里的返回值是字符串
        outputs.write(line_seg + '\n')
    outputs.close()
    inputs.close()
