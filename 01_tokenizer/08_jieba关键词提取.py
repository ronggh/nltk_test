import jieba.analyse

if __name__ == "__main__":
    # 字符串前面加u表示使用unicode编码
    content = u'中国特色社会主义是我们党领导的伟大事业，全面推进党的建设新的伟大工程，是这一伟大事业取得胜利的关键所在。党坚强有力，事业才能兴旺发达，国家才能繁荣稳定，人民才能幸福安康。党的十八大以来，我们党坚持党要管党、从严治党，凝心聚力、直击积弊、扶正祛邪，党的建设开创新局面，党风政风呈现新气象。习近平总书记围绕从严管党治党提出一系列新的重要思想，为全面推进党的建设新的伟大工程进一步指明了方向。'

    # 访问提取结果
    keywords = jieba.analyse.extract_tags(content, topK=20, withWeight=True, allowPOS=())
    for item in keywords:
        # 分别为关键词和相应的权重
        print(item[0], item[1])

    # 同样是四个参数，但allowPOS默认为('ns', 'n', 'vn', 'v')
    # 即仅提取地名、名词、动名词、动词
    keywords = jieba.analyse.textrank(content, topK=20, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
    # 访问提取结果
    for item in keywords:
        # 分别为关键词和相应的权重
        print(item[0], item[1])
