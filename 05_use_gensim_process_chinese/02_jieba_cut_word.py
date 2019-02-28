# 使用jieba分词，生成分词文件wiki.zh.text.seg
# 用法：
# python 02_jieba_cut_word.py wiki.zh.text wiki.zh.text.seg

import jieba
import logging
import os.path
import sys

if __name__ =="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    # check and process input arguments
    if len(sys.argv) < 3:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp = sys.argv[1:3]

    try:
        infile = open(inp, 'r', encoding="utf-8")
        outfile = open(outp, 'w', encoding="utf-8")
        print('open files...')
        line_num = 1
        line = infile.readline()

        while line:
            print('-------- processing... ', line_num, ' article----------------')
            line_seg = " ".join(jieba.cut(line))
            outfile.writelines(line_seg)
            line_num = line_num + 1
            line = infile.readline()

    finally:
        if infile:
            infile.close()
        if outfile:
            outfile.close()
