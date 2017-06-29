# -*- coding: utf-8 -*-

import os
import time
import tensorflow as tf
import numpy as np
import pickle
import scipy.misc
from scipy.misc import imresize
from six.moves import xrange

class DCGAN(object):
    def __init__(self, sess, input_height=32, input_width=32, batch_size=128,
        sample_num=64, output_height=32, output_width=32, z_dim=100, c_dim=3,
        learning_rate=0.0002, beta1=0.5, input_dir=None, checkpoint_dir=None, sample_dir=None):

        self.sess = sess                    # tensorflowのセッション
        self.batch_size = batch_size        # バッチサイズ
        self.sample_num = sample_num        # テスト出力画像数
        self.input_height = input_height    # 入力画像のサイズ(32×32)
        self.input_width = input_width      # 入力画像のサイズ(32×32)
        self.output_height = output_height  # 出力画像のサイズ(32×32)
        self.output_width = output_width    # 出力画像のサイズ(32×32)
        self.c_dim = c_dim                  # 入力画像のチャネル数（RGB）
        self.z_dim = z_dim                  # 入力ノイズ（一様分布）の要素数
        self.learning_rate = learning_rate  # 学習率
        self.beta1 = beta1                  # 学習時のモーメンタム
        self.input_dir = input_dir          # 入力画像の格納ディレクトリ
        self.checkpoint_dir = checkpoint_dir# チェックポイント格納先

        # モデル構築
        self.build_model()

    def build_model(self):
        # 学習画像用の変数
        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.input_height, self.input_width, self.c_dim], name='real_images')
        # 学習用入力ノイズの変数
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        # サンプル画像用入力ノイズの変数
        self.z_sample = tf.placeholder(tf.float32, [self.sample_num, self.z_dim], name='z')

        # 生成モデル
        self.G = self.generator(self.z)
        # 識別モデル
        self.D_t, self.D_logits_t = self.discriminator(self.inputs)
        self.D_g, self.D_logits_g = self.discriminator(self.G, reuse=True)
        # サンプル画像の出力
        self.sampler = self.generator(self.z_sample, reuse=True, is_training=False)

        # 誤差関数
        ## ↓Discriminatorは、D(G())をすべて0にし D(I)をすべて1にするのが理想
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_t, labels=tf.ones_like(self.D_logits_t)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_g, labels=tf.zeros_like(self.D_logits_g)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        ## ↓Generatorは、D(G())がすべて1になるのが理想
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_g, labels=tf.ones_like(self.D_logits_g)))

        # 訓練時に更新する変数取得
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # discriminatorの学習
        self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                  .minimize(self.d_loss, var_list=d_vars, global_step=self.global_step)
        # generatorの学習
        self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                  .minimize(self.g_loss, var_list=g_vars)

        # tensorboard出力
        z_hist = tf.summary.histogram("z", self.z)
        d_t_hist = tf.summary.histogram("d_train", self.D_t)
        d_g_hist = tf.summary.histogram("d_generated", self.D_g)
        g_img = tf.summary.image("G", self.G)
        g_loss_scalar = tf.summary.scalar("g_loss", self.g_loss)
        d_loss_scalar = tf.summary.scalar("d_loss", self.d_loss)
        d_loss_real_scalar = tf.summary.scalar("d_loss_real", self.d_loss_real)
        d_loss_fake_scalar = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.d_sum = tf.summary.merge([z_hist, d_t_hist, d_loss_real_scalar, d_loss_scalar])
        self.g_sum = tf.summary.merge([z_hist, g_img, d_g_hist, d_loss_fake_scalar, g_loss_scalar])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        # 変数の初期化
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()

    def train(self, config):
        # 学習データ（CIFAR10）のロード
        data = self.load_cifar10()
        # テスト用の入力ノイズ、画像（６４画像）の準備
        sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))

        # チェックポイントファイルのチェック（存在すればロード）
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Reading checkpoint from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")

        start_time = time.time()
        batch_idxs = len(data) // self.batch_size
        last_step = self.sess.run(self.global_step)
        last_epoch = last_step // batch_idxs
        counter = last_step + 1
        for epoch in xrange(config.epoch):
            for idx in xrange(0, batch_idxs):
                # 1ステップ分の学習データ、入力ノイズ取得
                batch_images = data[idx*self.batch_size : (idx+1)*self.batch_size]
                batch_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim)).astype(np.float32)

                # discriminatorの学習・更新
                _, summary_str = self.sess.run([self.d_optim, self.d_sum], feed_dict={self.inputs: batch_images, self.z: batch_z,})
                self.writer.add_summary(summary_str, counter)

                # generatorの学習・更新
                for i in range(2):
                    _, summary_str = self.sess.run([self.g_optim, self.g_sum], feed_dict={self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                errG = self.g_loss.eval({self.z: batch_z})

                format_str = ('Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f')
                print(format_str % (epoch+last_epoch+1, idx+1, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))
                counter += 1

            # 1epoch毎にサンプル画像生成
            samples = self.sess.run(self.sampler, feed_dict={self.z_sample: sample_z})
            self.save_images(samples, [8, 8],
                './{}/train_{:02d}.png'.format(config.sample_dir, epoch+last_epoch+1))
            print("[Generato Sample Images]")

            # モデルの保存
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'DCGAN.model'), global_step=epoch+last_epoch+1)

    def conv2d(self, name, inpt, shape, strides):
        with tf.variable_scope(name) as scope:
            # 畳み込み
            w = tf.get_variable('weights', shape=shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
            b = tf.get_variable('biases', shape=[shape[-1]], dtype=tf.float32, initializer=tf.zeros_initializer())
            conv = tf.nn.conv2d(inpt, w, strides, padding='SAME')
            output = tf.nn.bias_add(conv, b)
        return output

    def deconv2d(self, name, inpt, o_shape, strides):
        with tf.variable_scope(name) as scope:
            # 逆畳み込み
            w = tf.get_variable('weights', shape=[5, 5, o_shape[-1], inpt.get_shape()[-1]], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
            b = tf.get_variable('biases', shape=[o_shape[-1]], dtype=tf.float32, initializer=tf.zeros_initializer())
            deconv = tf.nn.conv2d_transpose(inpt, w, o_shape, strides, padding='SAME')
            output = tf.nn.bias_add(deconv, b)
        return output

    def linear(self, name, inpt, out_size):
        shape = inpt.get_shape().as_list()
        with tf.variable_scope(name) as scope:
            w = tf.get_variable('weights', shape=[shape[1], out_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
            b = tf.get_variable('biases', shape=[out_size], dtype=tf.float32, initializer=tf.zeros_initializer())
            output = tf.nn.bias_add(tf.matmul(inpt, w), b)
        return output

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            # 畳み込み層1（フィルター数：64、フィルターサイズ：5、ストライド2、Batch Normalizationなし）
            conv1 = self.conv2d('d_conv1', image, [5, 5, 3, 64], [1, 2, 2, 1])
            conv1 = tf.maximum(conv1, 0.2 * conv1)

            # 畳み込み層2（フィルター数：128、フィルターサイズ：5、ストライド2、Batch Normalizationあり）
            conv2 = self.conv2d('d_conv2', conv1, [5, 5, 64, 128], [1, 2, 2, 1])
            conv2 = tf.contrib.layers.batch_norm(
                conv2, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=True, scope='d_conv2_bn')
            conv2 = tf.maximum(conv2, 0.2 * conv2)

            # 畳み込み層3（フィルター数：256、フィルターサイズ：5、ストライド2、Batch Normalizationあり）
            conv3 = self.conv2d('d_conv3', conv2, [5, 5, 128, 256], [1, 2, 2, 1])
            conv3 = tf.contrib.layers.batch_norm(
                conv3, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=True, scope='d_conv3_bn')
            conv3 = tf.maximum(conv3, 0.2 * conv3)

            # 畳み込み層4（フィルター数：512、フィルターサイズ：5、ストライド2、Batch Normalizationあり）
            conv4 = self.conv2d('d_conv4', conv3, [5, 5, 256, 512], [1, 2, 2, 1])
            conv4 = tf.contrib.layers.batch_norm(
                conv4, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=True, scope='d_conv4_bn')
            conv4 = tf.maximum(conv4, 0.2 * conv4)

            # 出力層
            avg_pool = tf.reduce_mean(conv4, [1, 2])
            output = self.linear('d_output', avg_pool, 1)

        return tf.nn.sigmoid(output), output

    def generator(self, z, reuse=False, is_training=True):
        with tf.variable_scope("generator", reuse=reuse):
            z_shape = z.get_shape().as_list()

            # 入力ノイズの投射、リシェイプ（2×2×512）、Batch Normalizationあり
            img_size = int(self.output_width / 16)
            proj = self.linear('g_proj', z, img_size * img_size * 512)
            proj = tf.reshape(proj, [-1, img_size, img_size, 512])
            proj = tf.contrib.layers.batch_norm(
                proj, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training, scope='g_proj_bn')
            proj = tf.nn.relu(proj)

            # 逆畳み込み層１（フィルター数：256、フィルターサイズ：5、ストライド2、Batch Normalizationあり）
            img_size *= 2
            deconv1 = self.deconv2d('g_conv1', proj, [z_shape[0], img_size, img_size, 256], [1, 2, 2, 1])
            deconv1 = tf.contrib.layers.batch_norm(
                deconv1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training, scope='g_conv1_bn')
            deconv1 = tf.nn.relu(deconv1)

            # 逆畳み込み層２（フィルター数：128、フィルターサイズ：5、ストライド2、Batch Normalizationあり）
            img_size *= 2
            deconv2 = self.deconv2d('g_conv2', deconv1, [z_shape[0], img_size, img_size, 128], [1, 2, 2, 1])
            deconv2 = tf.contrib.layers.batch_norm(
                deconv2, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training, scope='g_conv2_bn')
            deconv2 = tf.nn.relu(deconv2)

            # 逆畳み込み層３（フィルター数：64、フィルターサイズ：5、ストライド2、Batch Normalizationあり）
            img_size *= 2
            deconv3 = self.deconv2d('g_conv3', deconv2, [z_shape[0], img_size, img_size, 64], [1, 2, 2, 1])
            deconv3 = tf.contrib.layers.batch_norm(
                deconv3, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training, scope='g_conv3_bn')
            deconv3 = tf.nn.relu(deconv3)

            # 逆畳み込み層４（フィルター数：3、フィルターサイズ：5、ストライド2、Batch Normalizationなし）
            img_size *= 2
            deconv4 = self.deconv2d('g_conv4', deconv3, [z_shape[0], img_size, img_size, 3], [1, 2, 2, 1])

        return tf.nn.tanh(deconv4)

    def load_cifar10(self):
        data = np.empty((0,32*32*3))
        for i in range(1,6):
            fname = os.path.join(self.input_dir, "%s%d" % ("data_batch_", i))
            with open(fname, 'rb') as f:
                cifar_dict = pickle.load(f, encoding='latin-1')
            data = np.vstack((data, cifar_dict['data']))

        # オリジナル画像は(chanel:3, row:32, column:32)のフラット形式
        # リシェイプ後、(row:32, column:32, chanel:3)に次元を入れ替える
        data = data.reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1)

        return np.array(data) / 127.5 - 1.

    def save_images(self, images, size, image_path):
        # 0~1の範囲に変換
        inv_imgs = (images + 1.) / 2.
        # 画像のマージ
        h, w, c = inv_imgs.shape[1], inv_imgs.shape[2], inv_imgs.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, inv_img in enumerate(inv_imgs):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = inv_img
        img = np.squeeze(img)
        return scipy.misc.imsave(image_path, img)
