#-*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'data/', "data set directory.")
tf.app.flags.DEFINE_string('log_dir','logs/', "tensorboard log directory.")
tf.app.flags.DEFINE_string('model_dir', 'models/', "trained model directory.")
tf.app.flags.DEFINE_string('meta_file', 'mymodel-2000.meta', "meta data to load.")

def main(argv):

	#mnistデータを格納したオブジェクトを呼び出す
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

	with tf.Session() as sess:

		state = tf.train.get_checkpoint_state(FLAGS.model_dir)
		#構築済みのmetaデータをロードしてくる。
		#グラフを定義していないのでdefaultグラフにロードされる
		meta_path = FLAGS.model_dir + FLAGS.meta_file
		saver = tf.train.import_meta_graph(meta_path)
		#最新の値でモデルをロード
		saver.restore(sess, state.model_checkpoint_path)

		#デフォルトグラフにアクセス
		graph = tf.get_default_graph()
		#グラフから必要な要素を取得してくる
		x = graph.get_tensor_by_name('input:0')
		feature = graph.get_tensor_by_name('inference/hidden/feature:0')
		test_images = mnist.test.images

		#feature_valにテストデータを入力した時の実際の値が入る
		feature_val = sess.run(feature, feed_dict={x: test_images})
		#初期値をfeature_valの値にしてあげて初期化して保存
		embed_val = tf.Variable(feature_val, trainable=False, name='embedded')
		sess.run(tf.variables_initializer([embed_val]))
		saver2 = tf.train.Saver([embed_val])
		saver2.save(sess, FLAGS.log_dir+'feature')

		#メタデータと対象のVariable の紐付けを行う
		summary_writer = tf.summary.FileWriter(FLAGS.log_dir)
		config = projector.ProjectorConfig()
		embedding = config.embeddings.add()
		#Variableの名前を登録
		embedding.tensor_name = embed_val.name
		#メタデータファイル名（相対パスにするならTensorBoard起動ディレクトリからの相対パス）
		embedding.metadata_path = "meta.tsv"
		#登録（projector_config.pbtxtというファイルが出来上がる）
		projector.visualize_embeddings(summary_writer, config)


if __name__ == '__main__':
	tf.app.run()

