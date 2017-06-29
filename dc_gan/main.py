# -*- coding: utf-8 -*-

import os
import scipy.misc
import numpy as np
import pprint
import tensorflow as tf
from model import DCGAN


flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("input_height", 32, "The size of image to use (The image of cifar 10 resized to 32 × 32). [32]")
flags.DEFINE_integer("input_width", 32, "The size of image to use (The image of cifar 10 resized to 32 × 32). [32]")
flags.DEFINE_integer("output_height", 32, "The size of the output images to produce [32]")
flags.DEFINE_integer("output_width", 32, "The size of the output images to produce. [32]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("z_dim", 100, "Dimension of sample noise. [100]")
flags.DEFINE_string("input_dir", "cifar10_data", "Directory name of input image [cifar10_data]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
    # パラメータ表示
    pp.pprint(flags.FLAGS.__flags)

    # ディレクトリがなければ作成
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # メモリ節約のため、最初に必要なGPUメモリをすべて確保するのではなく、
    # 必要になった時点で確保するようにallow_growthをTrueに設定する。
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True
    with tf.Session(config=run_config) as sess:
        dcgan = DCGAN(
            sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            c_dim=FLAGS.c_dim,
            z_dim=FLAGS.z_dim,
            learning_rate=FLAGS.learning_rate,
            beta1=FLAGS.beta1,
            input_dir=FLAGS.input_dir,
            checkpoint_dir=FLAGS.checkpoint_dir,
            sample_dir=FLAGS.sample_dir)

        # 学習開始
        dcgan.train(FLAGS)


if __name__ == '__main__':
  tf.app.run()
