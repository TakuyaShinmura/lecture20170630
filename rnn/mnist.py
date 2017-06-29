#-*- coding:utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


#mnistデータを格納したオブジェクトを呼び出す
mnist = input_data.read_data_sets("data/", one_hot=True)

"""モデル構築開始"""
#入力データ整形
num_seq = 28
num_input = 28

x = tf.placeholder(tf.float32, [None, 784])
input = tf.reshape(x, [-1, num_seq, num_input])


#ユニット数128個のLSTMセル
#三段に積む
stacked_cells = []
for i in range(3):
    stacked_cells.append(tf.nn.rnn_cell.LSTMCell(num_units=128))
cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_cells)

outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=input, dtype=tf.float32, time_major=False)
outputs = tf.transpose(outputs,[1, 0, 2])

last_output = tf.gather(outputs, int(outputs.get_shape()[0]) -1)

w = tf.Variable(tf.truncated_normal([128,10], stddev=0.1))
b = tf.Variable(tf.zeros([10]))

out = tf.nn.softmax(tf.matmul(last_output, w ) + b)

#正解データの型を定義
y = tf.placeholder(tf.float32, [None, 10])
#誤差関数（クロスエントロピー）
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(out), reduction_indices=[1]))

#訓練
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#評価
correct = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #テストデータをロード
    test_images = mnist.test.images
    test_labels = mnist.test.labels

    for step in range(1000):
        train_images, train_labels = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x:train_images ,y:train_labels})

        #10階ごとに精度を検証
        if step % 100 == 0:
            acc_val = sess.run( accuracy, feed_dict={x:test_images, y:test_labels})
            print('Step %d: accuracy = %.2f' % (step, acc_val))


