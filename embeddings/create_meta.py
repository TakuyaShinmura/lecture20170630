#-*- coding:utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

#mnistデータを格納したオブジェクトを呼び出す
mnist = input_data.read_data_sets("data/", one_hot=True)

with open('logs/meta.tsv', 'w') as f:
    corrects = np.nonzero(mnist.test.labels)[1]
    f.write('Index\tLabel\n')
    for index, label in enumerate(corrects):
    	f.write("%d\t%d\n" % (index, label))


