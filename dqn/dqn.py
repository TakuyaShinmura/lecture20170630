#-*- coding:utf-8 -*-

import tensorflow as tf
import gym

import random
import numpy as np
from collections import deque


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('log_dir','logs/', "tensorboard log directory.")
tf.app.flags.DEFINE_string('model_dir', 'models/', "trained model directory.")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
tf.app.flags.DEFINE_integer('max_step', 1000, "max step to train.")
tf.app.flags.DEFINE_integer('batch_size', 50, "batch size.")
tf.app.flags.DEFINE_integer('num_episode', 1000, "number of episode to play.")
tf.app.flags.DEFINE_bool('train', True, "Training mode.")

ENV_NAME = 'Breakout-v0'
NO_ACTION_STEP = 30
LEARNING_START_ITERATION = 100
DISCOUNT_RATE = 0.95
LEARNING_EVERY_N = 4
UPDATE_EVERY_N = 30
SAVE_EVERY_N = 100
WIDTH = 160
HEIGHT = 210
GAMMA = 0.9
REPLAY_SIZE = 1000
EPSILON = 0.95


def inference(input, scope_name, num_action):

	# 後々scopeから重みを取得してくるのでvariable_scope
	with tf.variable_scope(scope_name):
		with tf.variable_scope('conv1'):
			conv1 = tf.layers.conv2d(input, 32, [15,15],[2,2], padding='same', activation=tf.nn.relu)
		with tf.variable_scope('conv2'):
			conv2 = tf.layers.conv2d(conv1, 64, [15,15],[2,2], padding='same', activation=tf.nn.relu)
		with tf.variable_scope('fc'):
			shape = tf.shape(conv2)
			flat = tf.reshape(conv2, shape=[-1, 40*53*64])
			fc = tf.layers.dense(flat, 256,  activation=tf.nn.relu)
		with tf.variable_scope('out'):
			out = tf.layers.dense(fc, num_action)

	return out

# AtariGameは偶数フレームと奇数フレームで映るオブジェクトが違うので前処理
# グレースケール化
def preprocess(obs, last_obs):
	processed_obs = np.maximum(obs, last_obs)
	processed_obs = processed_obs.mean(axis=2, keepdims=True)
	processed_obs = processed_obs/256.0
	return processed_obs

def main(argv):

	#環境構築
	env = gym.make(ENV_NAME)
	#この環境で取りうるアクション
	num_action = env.action_space.n

	#状態s_t
	x_ph = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, 1])
	#状態s_t+1
	x2_ph = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, 1])
	#報酬r
	r_ph = tf.placeholder(tf.float32,[None,])
	#行動a
	a_ph = tf.placeholder(tf.int32,[None,])


	#qネットワーク、targetネットワークを構築
	q_net = inference(x_ph, 'q', num_action)
	target_net = inference(x2_ph, 'target', num_action)
	#それぞれの重みを取得してくる
	q_vars =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q')
	target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

	#ターゲット(正解データ)
	q_target = r_ph + GAMMA * tf.reduce_max(target_net, axis=1)
	#予測したq値
	a_one_hot = tf.one_hot(a_ph, depth=num_action, dtype=tf.float32)
	q_val = tf.reduce_sum(q_net * a_one_hot, axis=1)

	with tf.variable_scope('loss'):
		loss = tf.reduce_mean(tf.squared_difference(q_target, q_val))

	with tf.variable_scope('train'):
		global_step = tf.Variable(0, trainable=False, name="global_step")
		#訓練するのはqネットワークだけ
		train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss, var_list=q_vars, global_step=global_step)

	with tf.variable_scope('update'):
		#targetネットワークをアップデート
		update_op = [target_vars[i].assign(q_vars[i]) for i in range(len(q_vars))]



	init =tf.global_variables_initializer()
	saver = tf.train.Saver()

	#行動履歴を保存しておく
	replay_memory = deque([], maxlen=REPLAY_SIZE)

	#replay_memoryからランダムに(s_t,a,r,s_t+1, done)を取得してくる
	def sample_memories(batch_size):
		ids = np.random.permutation(len(replay_memory))[:batch_size]
		# state, action, reward, next_state, done
		cols = [[],[],[],[],[]]

		for id in ids:
			memory = replay_memory[id]

			for col, value in zip(cols, memory):

				col.append(value)
		cols = [np.array(col) for col in cols]
		return cols[0], cols[1], cols[2], cols[3], cols[4].reshape(-1,1)


	with tf.Session() as sess:

		ckpt_state = tf.train.get_checkpoint_state(FLAGS.model_dir)
		if ckpt_state:
			last_model = ckpt_state.model_checkpoint_path
			saver.restore(sess,last_model)
			print("model was loaded:", last_model)
		else:
			sess.run(init)
			print("initialized.")

		if FLAGS.train:
			iteration = 0
			for _ in range(FLAGS.num_episode):

				#初期状態の環境を取得
				obs = env.reset()
				last_obs = obs
				#最初の何ステップかは何もしない(初期状態をランダムにするため)
				for _ in range(random.randint(0, NO_ACTION_STEP)):
					last_obs = obs
					obs, _, _, _ = env.step(0)

				state = preprocess(obs, last_obs)
				done = False
				while not done:
					iteration += 1
					last_obs = obs
					#εグリーディー法（簡易版）
					action = env.action_space.sample()
					if np.random.uniform() < EPSILON:
						#qネットワークに状態を入れて次の行動を選択
						action_value = sess.run(q_net, feed_dict={x_ph:[state]})
						action = np.argmax(action_value)
					#行動することで新しい状態, 報酬, ゲームが終了したかどうかがわかる
					obs, reward, done, _ = env.step(action)
					next_state = preprocess(obs, last_obs)
					#行動履歴を保存
					replay_memory.append((state, action, reward, next_state, 1.0 - done))
					state = next_state

					if iteration > LEARNING_START_ITERATION and iteration % LEARNING_EVERY_N == 0:
						#学習に使う行動履歴を取得してくる
						x, a, r, x2, cont = sample_memories(FLAGS.batch_size)
						#訓練実行
						sess.run(train_op, feed_dict={x_ph:x, r_ph:r, a_ph:a, x2_ph:x2})
						step = global_step.eval() + 1
						if step % UPDATE_EVERY_N == 0:
							#一定時間たったらターゲット側をアップデート
							sess.run(update_op)
							print('update at %d step' % step)
						if step % SAVE_EVERY_N == 0:
							saver.save(sess, FLAGS.model_dir+my_model, global_step=step)
							print('saved as %d step' % step)
		else:
			print(env.action_space.n)
			for _ in range(5):
				obs = env.reset()
				state = preprocess(obs, obs)

				total_reward = 0
				for i in range(100):

					last_obs = obs
					#訓練時とは異なり、実際に画面に表示
					env.render()
					action_value = sess.run(q_net, feed_dict={x_ph:[state]})

					action = np.argmax(action_value)
					obs, reward, done, _ = env.step(action)
					total_reward += reward
					next_state = preprocess(obs, last_obs)
					state = next_state
					if done:
						break
				print("episode finishedafter %d timesteps" % (i+1))
				print("total reward %d"%total_reward)
			print("finish")





if __name__ == '__main__':
	tf.app.run()