# -*- coding: utf-8 -*-
# @Time     :2019/10/7 21:54
# @Author   :XiaoMa
# @Site     :
# @File     :rnn.py
# @Software :PyCharm
import collections
import tensorflow as tf

def build_dataset(documents):
    #外部列表表示每个文档，内部列表表示给定文档中的单词
    chars=[]
    data_list=[]

    for d in documents:
        chars.extend(d)
    print('找到%d个字符.'%len(chars))
    count=[]
    #根据频率对bigram排序（最高的排在第一位）
    count.extend(collections.Counter(chars).most_common())
    #为每个text赋ID
    dictionary=dict({'UNK':0})

    for char,c in count:
        #如果频率超过10，则仅增加一个bigram到字典中
        if c>10:
            dictionary[char]=len(dictionary)
    unk_count=0
    #遍历所有文本，我们将每个字符串单词替换为单词的ID
    for d in documents:
        data=list()
        for char in d:
            #如果单词在词典中存在，则使用单词ID，否则使用'UNK'
            if char in dictionary:
                index=dictionary[char]
            else:
                index=dictionary['UNK']
                unk_count+=1
            data.append(index)
        data_list.append(data)
    reverse_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    return data_list,count,dictionary,reverse_dictionary

import os
# 如果需要,创建一个目录
dir_name = '故事集'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def read_data(filename):
    with open(filename) as f:
        data = tf.compat.as_str(f.read())
        data = data.lower()
        data = list(data)
    return data


documents = []
global documents
num_files=100
filenames = [format(i, '03d')+'.txt' for i in range(1,101)]
for i in range(num_files):
    print('\n处理文件 %s'%os.path.join(dir_name,filenames[i]))
    chars = read_data(os.path.join(dir_name,filenames[i]))
    two_grams = [''.join(chars[ch_i:ch_i+2]) for ch_i in range(0,len(chars)-2,2)]
    documents.append(two_grams)
    print('数据大小 (字符) (文档 %d) %d' %(i,len(two_grams)))
    print('样本字符串 (文档 %d)  %s'%(i,two_grams[:50]))


data_list,count,dictionary,reverse_dictionary=build_dataset()
#定义超参数
#num_unroll:展开的步数
#batch_size
#hidden:隐层神经元数量

tf.reset_default_graph()

num_unroll=50
batch_size=64
test_batch_size=1

hidden=64
vocabulary_size=len(dictionary)
#输入大小和输出大小
in_size,out_size=vocabulary_size,vocabulary_size

#定义输入和输出
train_dataset,train_labels=[],[]
for ui in range(num_unroll):
    train_dataset.append(tf.placeholder(tf.float32,shape=[batch_size,in_size],name='train_dataset_%d'%ui))
    train_labels.append(tf.placeholder(shape=[batch_size,out_size],name='train_labels_%d'%ui))

#验证数据集
valid_dataset=tf.placeholder(tf.float32,shape=[1,in_size],name='valid_dataset')
valid_labels=tf.placeholder(tf.float32,shape=[1,out_size],name='valid_labels')

#测试数据集
test_dataset=tf.placeholder(tf.float32,shape=[test_batch_size,in_size],name='test_dataset')

#定义模型参数和其他变量
#输入层和隐层之间的权重
W_xh=tf.Variable(tf.truncated_normal([in_size,hidden],stddev=0.02,dtype=tf.float32),name='W_xh')

#隐层之间的权重
W_hh=tf.Variable(tf.truncated_normal([hidden,hidden],stddev=0.02,dtype=tf.float32),name='W_hh')

#隐层和输出层之间的权重
W_hy=tf.Variable(tf.truncated_normal([hidden,out_size],stddev=0.02),name='W_hy')

#获取无法训练变量（训练数据）中隐藏节点的之前状态
prev_train_h=tf.Variable(tf.zeros([batch_size,hidden],dtype=tf.float32),name='train_h',trainable=False)

#获取无法训练变量（验证数据）中隐藏节点的之前状态
prev_valid_h=tf.Variable(tf.zeros([1,hidden],dtype=tf.float32),name='valid_h',trainable=False)

#获取测试阶段隐藏节点的之前状态
prev_test_h=tf.Variable(tf.zeros([test_batch_size,hidden],dtype=tf.float32),name='test_h')

#定义RNN的推断
#训练得分（非标准化）值和预测（标准化）
y_scores,y_predictions=[],[]

#在num_unroll步数中为每个步长添加计算的RNN输出