from nltk.text import TextCollection

if __name__ == "__main__":
    corpus = ['this is sentence one' , 'this is sentence two', 'this is sentence three']
    # 首先, 把所有的⽂档放到TextCollection类中。这个类会自动断句, 做统计, 做计算
    corpus_tc = TextCollection(corpus)
    # 直接就能算出tfidf
    # (term: 一句句话中的某个term, text: 这句话)
    print(corpus_tc.tf_idf('this', 'this is sentence four'))

