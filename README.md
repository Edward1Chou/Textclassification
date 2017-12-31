## 中文文本分类对比（经典方法和CNN）

### 背景介绍

笔者实验室项目正好需要用到文本分类，作为NLP领域最经典的场景之一，文本分类积累了大量的技术实现方法，如果将是否使用深度学习技术作为标准来衡量，实现方法大致可以分成两类：

- 基于传统机器学习的文本分类
- 基于深度学习的文本分类

比如facebook之前开源的fastText属于第一类，还有业界研究上使用比较的TextCNN模型属于第二类。有一个github项目很好的把这些模型都集中到了一起，并做了一些简单的性能比较，想要进一步了解这些高大上模型的同学可以查看如下链接：

[all kinds of text classificaiton models and more with deep learning](https://github.com/brightmart/text_classification)

本文的目的主要记录笔者自己构建文本分类系统的过程，分别构建基于传统机器学习的文本分类和基于深度学习的文本分类系统，并在同一数据集上进行测试。

经典的机器学习方法采用获取tf-idf文本特征，分别喂入logistic regression分类器和随机森林分类器的思路，并对两种方法做性能对比。

基于深度学习的文本分类，这里主要采用CNN对文本分类，考虑到RNN模型相较CNN模型性能差异不大并且耗时还比较久，这里就不多做实验了。


实验过程有些比较有用的small trick分享，包括多进程分词、训练全量tf-idf、python2对中文编码的处理技巧等等，在下文都会仔细介绍。

### 食材准备

本文采用的数据集是很流行的搜狗新闻数据集，get到的时候已经是经过预处理的了，所以省去了很多数据预处理的麻烦，数据集下载链接如下：

[新闻文本分类数据集下载](http://pan.baidu.com/s/1bpq9Eub)

密码：ycyw

数据集一共包括10类新闻，每类新闻65000条文本数据，训练集50000条，测试集10000条，验证集5000条。

### 经典机器学习方法

#### 分词、去停用词

调用之前[短文本分类](https://www.jianshu.com/p/2aaf1a94b7d6)博文中提到的分词工具类，对训练集、测试集、验证集进行多进程分词，以节省时间：

```python
import multiprocessing


tmp_catalog = '/home/zhouchengyu/haiNan/textClassifier/data/cnews/'
file_list = [tmp_catalog+'cnews.train.txt', tmp_catalog+'cnews.test.txt']
write_list = [tmp_catalog+'train_token.txt', tmp_catalog+'test_token.txt']

def tokenFile(file_path, write_path):
    word_divider = WordCut()
    with open(write_path, 'w') as w:
        with open(file_path, 'r') as f:
            for line in f.readlines():
                line = line.decode('utf-8').strip()
                token_sen = word_divider.seg_sentence(line.split('\t')[1])
                w.write(line.split('\t')[0].encode('utf-8') + '\t' + token_sen.encode('utf-8') + '\n') 
    print file_path + ' has been token and token_file_name is ' + write_path

pool = multiprocessing.Pool(processes=4)
for file_path, write_path in zip(file_list, write_list):
    pool.apply_async(tokenFile, (file_path, write_path, ))
pool.close()
pool.join() # 调用join()之前必须先调用close()
print "Sub-process(es) done."
```

#### 计算tf-idf

这里有几点需要注意的，一是计算tf-idf是全量计算，所以需要将train+test+val的所有corpus都相加，再进行计算，二是为了防止文本特征过大，需要去低频词，因为是在jupyter上写的，所以测试代码的时候，先是选择最小的val数据集，成功后，再对test,train数据集迭代操作，希望不要给大家留下代码冗余的影响...[悲伤脸]。实现代码如下：

```python
def constructDataset(path):
    """
    path: file path
    rtype: lable_list and corpus_list
    """
    label_list = []
    corpus_list = []
    with open(path, 'r') as p:
        for line in p.readlines():
            label_list.append(line.split('\t')[0])
            corpus_list.append(line.split('\t')[1])
    return label_list, corpus_list
    
tmp_catalog = '/home/zhouchengyu/haiNan/textClassifier/data/cnews/'
file_path = 'val_token.txt'
val_label, val_set = constructDataset(tmp_catalog+file_path)
print len(val_set)

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


tmp_catalog = '/home/zhouchengyu/haiNan/textClassifier/data/cnews/'
write_list = [tmp_catalog+'train_token.txt', tmp_catalog+'test_token.txt']

tarin_label, train_set = constructDataset(write_list[0]) # 50000
test_label, test_set = constructDataset(write_list[1]) # 10000
# 计算tf-idf
corpus_set = train_set + val_set + test_set # 全量计算tf-idf
print "length of corpus is: " + str(len(corpus_set))
vectorizer = CountVectorizer(min_df=1e-5) # drop df < 1e-5,去低频词
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus_set))
words = vectorizer.get_feature_names()
print "how many words: {0}".format(len(words))
print "tf-idf shape: ({0},{1})".format(tfidf.shape[0], tfidf.shape[1])

"""
length of corpus is: 65000
how many words: 379000
tf-idf shape: (65000,379000)
"""
```

#### 标签数字化，抽取数据

因为本来文本就是以一定随机性抽取成3份数据集的，所以，这里就不shuffle啦，偷懒一下下。。但是如果能shuffle的话，尽量还是做这一步，坚持正途。

```python
from sklearn import preprocessing

# encode label
corpus_label = tarin_label + val_label + test_label
encoder = preprocessing.LabelEncoder()
corpus_encode_label = encoder.fit_transform(corpus_label)
train_label = corpus_encode_label[:50000]
val_label = corpus_encode_label[50000:55000]
test_label = corpus_encode_label[55000:]
# get tf-idf dataset
train_set = tfidf[:50000]
val_set = tfidf[50000:55000]
test_set = tfidf[55000:]
```

#### 喂入分类器

- logistic regression分类器

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix

# LogisticRegression classiy model
lr_model = LogisticRegression()
lr_model.fit(train_set, train_label)
print "val mean accuracy: {0}".format(lr_model.score(val_set, val_label))
y_pred = lr_model.predict(test_set)
print classification_report(test_label, y_pred)
```

分类报告如下（包括准确率、召回率、F1值）:

```text
mean accuracy: 0.9626
             precision    recall  f1-score   support

          0       1.00      1.00      1.00      1000
          1       0.99      0.98      0.98      1000
          2       0.94      0.87      0.91      1000
          3       0.91      0.91      0.91      1000
          4       0.97      0.93      0.95      1000
          5       0.97      0.98      0.98      1000
          6       0.93      0.96      0.95      1000
          7       0.99      0.97      0.98      1000
          8       0.94      0.99      0.96      1000
          9       0.95      0.99      0.97      1000

avg / total       0.96      0.96      0.96     10000
```



- Random Forest 分类器

```python
# 随机森林分类器
from sklearn.ensemble import RandomForestClassifier    


rf_model = RandomForestClassifier(n_estimators=200, random_state=1080)
rf_model.fit(train_set, train_label)
print "val mean accuracy: {0}".format(rf_model.score(val_set, val_label))
y_pred = rf_model.predict(test_set)
print classification_report(test_label, y_pred)
```

分类报告如下（包括准确率、召回率、F1值）:


```text
val mean accuracy: 0.9228
             precision    recall  f1-score   support

          0       1.00      1.00      1.00      1000
          1       0.98      0.98      0.98      1000
          2       0.89      0.57      0.69      1000
          3       0.81      0.97      0.88      1000
          4       0.95      0.89      0.92      1000
          5       0.97      0.96      0.97      1000
          6       0.85      0.94      0.89      1000
          7       0.95      0.97      0.96      1000
          8       0.95      0.97      0.96      1000
          9       0.91      0.99      0.95      1000

avg / total       0.93      0.92      0.92     10000
```

#### 分析

```text
上面采用逻辑回归分类器和随机森林分类器做对比：
可以发现，除了个别分类随机森林方法有较大进步，大部分都差于逻辑回归分类器
并且200棵树的随机森林耗时过长，比起逻辑回归分类器来说，运算效率太低
```

### CNN文本分类

这一部分主要是参考tensorflow社区的一份博客进行实验的，这里也不再赘述，博客讲的非常好，附上原文链接，前去膜拜：[NN-RNN中文文本分类，基于tensorflow](http://www.tensorflownews.com/2017/11/04/text-classification-with-cnn-and-rnn/)

#### 字符级特征提取

这里和前文差异比较大的地方，主要是提取文本特征这一块，这里的CNN模型采用的是字符级特征提取，比如data目录下cnews_loader.py中：

```python
def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                contents.append(list(content)) # 字符级特征
                labels.append(label)
            except:
                pass
    return contents, labels

def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
```

笔者做了下测试：

```python
#! /bin/env python
# -*- coding: utf-8 -*-
from collections import Counter

"""
字符级别处理,
对于中文来说，基本不是原意的字，但是也能作为一种统计特征来表征文本
"""
content1 = "你好呀大家"
content2 = "你真的好吗？"
# content = "abcdefg"
all_data = []
all_data.extend(list(content1))
all_data.extend(list(content2))
# print list(content) # 字符级别处理
# print "length: " + str(len(list(content)))
counter = Counter(all_data)
count_pairs = counter.most_common(5)
words, _ = list(zip(*count_pairs))
words = ['<PAD>'] + list(words) #['<PAD>', '\xe5', '\xbd', '\xa0', '\xe4', '\xe7']
```

这种基本不是原意的字符级别的特征，也能从统计意义上表征文本，从而作为特征，这一点需要清楚。

#### 迁移python2

github上的版本是python3的，由于笔者一直使用的是python2,所以对上述工作做了一点版本迁移，使得在如下环境下也能顺利运行：

```text
Python 2.7
TensorFlow 1.3
numpy
scikit-learn
```

除了p3和py2差异比较明显的类定义、print、除法运算外，还有就是中文编码，使用codecs模块可以很好的解决这个问题，由于是细枝末节，这里也就不展开来说了。

最终，在同一数据集上，得到的测试报告如下：

```text
Test Loss:   0.13, Test Acc:  96.06%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support

      sports       0.99      0.99      0.99      1000
     finance       0.96      0.99      0.98      1000
       house       1.00      0.99      1.00      1000
      living       0.99      0.88      0.93      1000
   education       0.90      0.93      0.92      1000
        tech       0.92      0.99      0.95      1000
     fashion       0.95      0.97      0.96      1000
      policy       0.97      0.92      0.94      1000
        game       0.97      0.97      0.97      1000
entertaiment       0.95      0.98      0.96      1000

 avg / total       0.96      0.96      0.96     10000

```

#### 分析

可以看出与传统机器学习方法相比，貌似深度学习方法优势不大，但是考虑到数据集数量不多、深度学习模型仍旧是个baseline,还可以通过进一步的调节参数，来达到更好的效果,深度学习在文本分类性能优化方面，依旧是大有可为的。



- 参考资料

[NN-RNN中文文本分类，基于tensorflow](http://www.tensorflownews.com/2017/11/04/text-classification-with-cnn-and-rnn/)

