# 04_test_model.py用于测试模型训练效果
# 用法：
# python 04_test_model.py <word>
import gensim
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    inp = sys.argv[1]

    model = gensim.models.Word2Vec.load("wiki.zh.text.model")
    # 可以把足球替换成其它词，检验，如：男人，女人，青蛙，姨夫，衣服，公安局，铁道部
    result = model.most_similar(inp)
    for e in result:
        print(e[0], e[1])
