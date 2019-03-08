"""
用RNN做文本生成
举个小小的例子，来看看LSTM是怎么玩的,用温斯顿丘吉尔的人物传记作为我们的学习语料。
(各种中文语料可以自行网上查找， 英文的小说语料可以从古登堡计划网站下载txt平文本：
https://www.gutenberg.org/wiki/Category:Bookshelf)
实现的功能是：
    简单的文本预测就是，给了前置的字母以后，下一个字母是谁？
    比如，Winsto, 给出 n Britai 给出 n
先导入各种库
"""
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


def predict_next(input_array):
    """
    预测下一个字母
    :param input_array:
    :return:
    """
    x = numpy.reshape(input_array, (1, seq_length, 1))
    x = x / float(n_vocab)
    y = model.predict(x)
    return y


def string_to_index(raw_input):
    """

    :param raw_input:
    :return:
    """
    res = []
    for c in raw_input[(len(raw_input) - seq_length):]:
        res.append(char_to_int[c])
    return res


def y_to_char(y):
    largest_index = y.argmax()
    c = int_to_char[largest_index]
    return c


def generate_article(init, rounds=500):
    """
    生成预测文本
    :param init:
    :param rounds:
    :return:
    """
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_char(predict_next(string_to_index(in_string)))
        in_string += n
    return in_string


if __name__ == "__main__":
    # 第一步，读入文本
    raw_text = open('./input_data/Winston_Churchil.txt',encoding='utf-8').read()
    raw_text = raw_text.lower()

    # 以每个字母为层级，字母总共才26个，所以我们可以很方便的用One-Hot来编码出所有的字母（当然，可能还有些标点符号和其他noise）
    chars = sorted(list(set(raw_text)))
    # 做成两个字典数据
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    # 这里打印一个字符集个数和原文本的长度，只是为了看一下
    print("字符集个数：%d" % (len(chars)))
    print("原文本长度：%d " % (len(raw_text)))

    # 构造训练测试集，需要把raw_text变成可以用来训练的x, y:
    #  x:是前置字母集合
    # y:是后一个字母
    seq_length = 100
    x = []
    y = []
    for i in range(0, len(raw_text) - seq_length):
        given = raw_text[i:i + seq_length]
        predict = raw_text[i + seq_length]
        x.append([char_to_int[char] for char in given])
        y.append(char_to_int[predict])

    #
    n_patterns = len(x)
    n_vocab = len(chars)

    # 把x变成LSTM需要的样子
    x = numpy.reshape(x, (n_patterns, seq_length, 1))
    # 简单normal到0-1之间
    x = x / float(n_vocab)
    # output变成one-hot
    y = np_utils.to_categorical(y)

    # 模型建造：LSTM模型构建
    model = Sequential()
    model.add(LSTM(128, input_shape=(x.shape[1], x.shape[2])))
    model.add(Dropout(0.2))
    # 增加一个Dense层
    model.add(Dense(y.shape[1], activation='softmax'))
    # 交叉熵和亚当优化器
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # 进行训练，要花费较长时间，可以调整epochs参数值可改进性能
    print("进行训练，要花费相当长时间...")
    model.fit(x, y, epochs=10, batch_size=128)

    # 保存模型：默认保存为h5格式，包括模型网络和权重
    # 如只保存模型的结构，而不包含其权重或配置信息，可用:JSON或YAML格式
    # json_string = model.to_json()
    # yaml_string = model.to_yaml()
    # 加载：
    #       from keras.models import model_from_json
    #       model = model_from_json(json_string)
    #       model reconstruction from YAML
    #       model = model_from_yaml(yaml_string)
    # 权重的保存和加载
    #       model.save_weights(path)
    #       model.load_weights(path)
    #model.save("../out_data/char_rnn.h5")
    # 加载模型
    # model.load_model("../out_data/char_rnn.h5")

    # 检验模型
    init = 'Professor Michael S. Hart is the originator of the Project'
    article = generate_article(init)
    print(article)
