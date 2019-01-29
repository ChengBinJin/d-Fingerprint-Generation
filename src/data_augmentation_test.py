# ---------------------------------------------------------
# Tensorflow dFingerprint Generation MR-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------

import os
import sys
import cv2
import numpy as np
import tensorflow as tf

import utils as utils
from utility import reader

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_float('gpu_memory_fraction', 1., 'GPU memory fraction to use.')
tf.flags.DEFINE_string('dataset', 'BMP_320x280', 'dataset name, default: BMP_320x280')
tf.flags.DEFINE_integer('iters', 10, 'number of iterations, default: 200000')


def main(_):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    run_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    run_config.gpu_options.allow_growth = True
    sess = tf.Session(config=run_config)

    data_path = '../../Data/BMP_320x280/tfrecord/train.tfrecords'
    data_reader = reader.Reader(data_path, name='data', image_size=(320, 280, 3), batch_size=1)
    x_imgs_, y_imgs_, x_imgs_ori_, y_imgs_ori_, img_name = data_reader.feed()
    sess.run(tf.global_variables_initializer())

    # threads for tfrecord
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    iter_time = 0
    try:
        while iter_time < FLAGS.iters:
            x_imgs, y_imgs, x_imgs_ori, y_imgs_ori = sess.run([x_imgs_, y_imgs_, x_imgs_ori_, y_imgs_ori_])
            _, h, w, c = x_imgs.shape
            canvas = np.zeros((h, 4 * w, c), dtype=np.uint8)

            x_imgs = utils.inverse_transform(x_imgs)
            y_imgs = utils.inverse_transform(y_imgs)
            x_imgs_ori = utils.inverse_transform(x_imgs_ori)
            y_imgs_ori = utils.inverse_transform(y_imgs_ori)

            y_imgs = utils.draw_minutiae(x_imgs, y_imgs)

            canvas[:, 0:w, :] = x_imgs[0]
            canvas[:, w:2*w, :] = y_imgs
            canvas[:, 2*w:3*w, :] = x_imgs_ori[0]
            canvas[:, 3*w:, :] = y_imgs_ori[0]

            iter_time += 1

            cv2.imshow('show', canvas[:, :, ::-1])
            if cv2.waitKey(0) == 27:
                sys.exit('Esc clicked!')

    except KeyboardInterrupt:
        coord.request_stop()
    except Exception as e:
        coord.request_stop(e)
    finally:
        # when done, ask the threads to stop
        coord.request_stop()
        coord.join(threads)



if __name__ == '__main__':
    tf.app.run()
