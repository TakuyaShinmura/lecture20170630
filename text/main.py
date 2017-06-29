#-*- coding:utf-8 -*-

import tensorflow as tf

text_list = ['text0','text1','text2','text3','text4','text5']

test_string = tf.convert_to_tensor('Test String.')

outer_summary = tf.summary.text('out',test_string)

with tf.name_scope('scope_test') as scope:

	text_ph = tf.placeholder(tf.string,name='input')
	inner_summary = tf.summary.text('in', text_ph)

summary_op = tf.summary.merge_all()

with tf.Session() as sess:
	summary_writer = tf.summary.FileWriter(logdir='logs', graph=sess.graph)
	for i in range(5):
		step = i+1
		result = sess.run(summary_op,feed_dict={text_ph:text_list[i]})
		summary_writer.add_summary(result, global_step=step)



