# -*- coding:utf-8 -*-

import tensorflow as  tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'data/', "data set directory.")
tf.app.flags.DEFINE_string('log_dir','logs/', "tensorboard log directory.")
tf.app.flags.DEFINE_string('model_dir', 'models/', "trained model directory.")
tf.app.flags.DEFINE_float('learning_rate', 0.5, "learning rate.")
tf.app.flags.DEFINE_integer('max_step', 1000, "max step to train.")
tf.app.flags.DEFINE_integer('batch_size', 50, "batch size.")

def main(argv):
    #データセット作成
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784], name='input')

    #入力画像を確認
    images = tf.reshape(x,[-1,28,28,1])
    tf.summary.image("input_data", images, 10)

    #推論
    with tf.name_scope('inference'):
        with tf.name_scope('hidden'):
            w_1 = tf.Variable(tf.truncated_normal([784, 64], stddev=0.1), name='w')
            b_1 = tf.Variable(tf.zeros([64]), name='b')
            h_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1, name='feature')

        with tf.name_scope('out'):
            w_2 = tf.Variable(tf.truncated_normal([64, 10], stddev=0.1), name="w")
            b_2 = tf.Variable(tf.zeros([10]), name="b")
            out = tf.nn.softmax(tf.matmul(h_1, w_2) + b_2, name='out')

            tf.summary.histogram('w', w_2)

    y = tf.placeholder(tf.float32, [None, 10], name='correct')
    
    #誤差計算
    with tf.name_scope('loss'):
        each_loss = tf.reduce_mean(tf.square(y-out), axis=[1])
        loss = tf.reduce_mean(each_loss)
        tf.summary.histogram('each_loss', each_loss)
        tf.summary.scalar("loss", loss)

    #訓練
    with tf.name_scope('train'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss, global_step=global_step)

    #精度評価
    with tf.name_scope('evaluate'):
        correct = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy', accuracy)


    #初期化
    init =tf.global_variables_initializer()   
    
    #全てのログをマージ
    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=3)

    
    with tf.Session() as sess:

        #ログのwriterを定義
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        #初期化or訓練済みモデル読み込み
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt_state:
            last_model = ckpt_state.model_checkpoint_path
            saver.restore(sess,last_model)
            print("model was loaded:", last_model)  
        else:
            sess.run(init)
            print("initialized.")

        #テストデータをロード
        test_images = mnist.test.images
        test_labels = mnist.test.labels

        last_step = sess.run(global_step)
        
        for i in range(FLAGS.max_step):
            step = last_step + i + 1
            train_images, train_labels = mnist.train.next_batch(FLAGS.batch_size)
            sess.run(train_step, feed_dict={x:train_images ,y:train_labels})

            #一定ステップごとに精度評価
            if step % 100 == 0:
                run_opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_meta = tf.RunMetadata()
                summary_str = sess.run(summary_op, feed_dict={x:test_images, y:test_labels}, options=run_opt, run_metadata=run_meta)
                summary_writer.add_summary(summary_str, step)
                summary_writer.add_run_metadata(run_meta, 'step%d'%step)
                saver.save(sess, FLAGS.model_dir+'mymodel', global_step = step)

        

if __name__ == '__main__':
    tf.app.run()    
