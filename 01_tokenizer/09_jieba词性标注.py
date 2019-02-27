# 加载jieba.posseg并取个别名，方便调用
import jieba.posseg as pseg

if __name__ == "__main__":
    words = pseg.cut("我爱北京天安门.根据分词结果中每个词的词性，可以初步实现命名实体识别,")
    for word, flag in words:
        # 格式化模版并传入参数
        print('%s, %s' % (word, flag))
