#! /bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardChou

分词，去停用词
去停用词可以理解成去噪声

该类既可以分词句子，也可以分词文本
"""
import jieba
import os


stopwords_path = os.path.normpath(os.path.dirname(__file__)) + "/stopwords.txt"

class WordCut(object):
    def __init__(self, stopwords_path=stopwords_path):
        """
        :stopwords_path: 停用词文件路径

        """
        self.stopwords_path = stopwords_path


    def addDictionary(self, dict_list):
        """
        添加用户自定义字典列表
        """
        map(lambda x: jieba.load_userdict(x), dict_list)


    def seg_sentence(self, sentence, stopwords_path = None):
        """
        对句子进行分词  
        """
        # print "now token sentence..."
        if stopwords_path is None:
            stopwords_path = self.stopwords_path
        def stopwordslist(filepath):
            """
            创建停用词list ,闭包 
            """
            stopwords = [line.decode('utf-8').strip() for line in open(filepath, 'r').readlines()]  
            return stopwords
        sentence_seged = jieba.cut(sentence.strip())  
        stopwords = stopwordslist(stopwords_path)  # 这里加载停用词的路径  
        outstr = ''  # 返回值是字符串
        for word in sentence_seged:  
            if word not in stopwords:  
                if word != '\t':  
                    outstr += word  
                    outstr += " "  
        return outstr 


    def seg_file(self, path, show=True, write=False, write_name='token.txt'):
        """
        对文本进行分词
        """
        print "now token file..."
        if write == True:
            write_path = '/'.join(path.split('/')[:-1]) + '/' + write_name
            w = open(write_path, 'w')
        # lines_list = []
        with open(path, 'r') as p:
            for line in p.readlines():
                line_seg = self.seg_sentence(line)
                # lines_list.append(line_seg)
                if show == True:
                    print item
                if write == True:
                    w.write(line_seg.encode('utf-8'))
                    w.write('\n')
        if write == True:
            w.close()


# stopwords_path = "stopwords.txt"
# mydict = ["/home/zcy/haiNan/texttravelgen/textCluster/mysenicdict.txt", "/home/zcy/haiNan/texttravelgen/textCluster/myfooddict.txt"]
# file_path = '/home/zcy/haiNan/texttravelgen/data/clean_comments.txt'
# sentence = "黔清宫真是像大婶说的一样，一直到文昌市，一路都没看到卖椰子的。到文昌市区，已经是下午3点，我们早已饥肠辘辘，赶忙下车寻觅午餐。到了文昌当然要吃最有名的文昌鸡，而从当地人口中得知最好吃的做法就是盐焗鸡，市区里非常好找，沿路有不少的店面。" # 默认是精确模式

# test = WordCut(stopwords_path)
# test.addDictionary(mydict)
# print test.seg_sentence(sentence)
# test.seg_file(file_path, show=False, write=True)



