# ---------------------------------------------------------
# Tensorflow dFingerprint Generation MR-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import cv2
import time
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime

# noinspection PyPep8Naming
import tensorflow_utils as tf_utils
from dataset import Dataset
from mrgan import MRGAN

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


class Solver(object):
    def __init__(self, flags):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=flags.gpu_memory_fraction)
        run_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.iter_time = 0
        self._make_folders()
        self._init_logger()

        self.dataset = Dataset(self.flags.dataset, self.flags, log_path=self.log_out_dir)
        self.model = MRGAN(self.sess, self.flags, self.dataset.image_size, self.dataset(self.flags.is_train),
                           log_path=self.log_out_dir)

        # self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        tf_utils.show_all_variables()

    def _make_folders(self):
        if self.flags.is_train:  # train stage
            if self.flags.load_model is None:
                cur_time = datetime.now().strftime("%Y%m%d-%H%M")
                self.model_out_dir = "{}/model/{}".format(self.flags.dataset, cur_time)
                if not os.path.isdir(self.model_out_dir):
                    os.makedirs(self.model_out_dir)
            else:
                cur_time = self.flags.load_model
                self.model_out_dir = "{}/model/{}".format(self.flags.dataset, self.flags.load_model)

            self.sample_out_dir = "{}/sample/{}".format(self.flags.dataset, cur_time)
            if not os.path.isdir(self.sample_out_dir):
                os.makedirs(self.sample_out_dir)

            self.log_out_dir = "{}/logs/{}".format(self.flags.dataset, cur_time)
            self.train_writer = tf.summary.FileWriter("{}/logs/{}".format(self.flags.dataset, cur_time),
                                                      graph_def=self.sess.graph_def)

        elif not self.flags.is_train:  # test stage
            self.model_out_dir = "{}/model/{}".format(self.flags.dataset, self.flags.load_model)
            self.test_out_dir = "{}/test/{}".format(self.flags.dataset, self.flags.load_model)
            self.log_out_dir = "{}/logs/{}".format(self.flags.dataset, self.flags.load_model)

            if not os.path.isdir(self.test_out_dir):
                os.makedirs(self.test_out_dir)

    def _init_logger(self):
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
        # file handler
        file_handler = logging.FileHandler(os.path.join(self.log_out_dir, 'solver.log'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        # stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        # add handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        if self.flags.is_train:
            logger.info('gpu_index: {}'.format(self.flags.gpu_index))
            logger.info('is_train: {}'.format(self.flags.is_train))

            logger.info('batch_size: {}'.format(self.flags.batch_size))
            logger.info('dataset: {}'.format(self.flags.dataset))
            logger.info('L1_lambda: {}'.format(self.flags.L1_lambda))
            logger.info('learning_rate: {}'.format(self.flags.learning_rate))
            logger.info('beta1: {}'.format(self.flags.beta1))

            logger.info('iters: {}'.format(self.flags.iters))
            logger.info('print_freq: {}'.format(self.flags.print_freq))
            logger.info('save_freq: {}'.format(self.flags.save_freq))
            logger.info('sample_freq: {}'.format(self.flags.sample_freq))
            logger.info('load_model: {}'.format(self.flags.load_model))

    def train(self):
        # load initialized checkpoint that provided
        if self.flags.load_model is not None:
            if self.load_model():
                logger.info(' [*] Load SUCCESS!\n')
            else:
                logger.info(' [!] Load Failed...\n')

        # threads for tfrecord
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        try:
            while self.iter_time < self.flags.iters:
                x_imgs, y_imgs, x_imgs_ori, y_imgs_ori = self.model.test()
                print('x_imgs shape: {}'.format(x_imgs.shape))
                _, h, w, c = x_imgs.shape
                canvas = np.zeros((h, 4*w, c), dtype=np.uint8)

                for idx, img in enumerate([x_imgs, y_imgs, x_imgs_ori, y_imgs_ori]):
                    print('img shape: {}'.format(img.shape))
                    img = 255 * (img + 1.) / 2.
                    canvas[:, idx*w:(idx+1)*w, :] = img

                cv2.imshow('show', canvas[:, :, ::-1])
                cv2.waitKey(0)

            #     # samppling images and save them
            #     self.sample(self.iter_time)
            #
            #     # train_step
            #     loss, summary = self.model.train_step()
            #     self.model.print_info(loss, self.iter_time)
            #     self.train_writer.add_summary(summary, self.iter_time)
            #     self.train_writer.flush()
            #
            #     # save model
            #     self.save_model(self.iter_time)
            #     self.iter_time += 1
            #
            # self.save_model(self.flags.iters)

        except KeyboardInterrupt:
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # when done, ask the threads to stop
            coord.request_stop()
            coord.join(threads)

    def test(self):
        if self.load_model():
            logger.info(' [*] Load SUCCESS!')
        else:
            logger.info(' [!] Load Failed...')

        # threads for tfrecord
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        iter_time = 0
        total_time = 0.
        try:
            while iter_time < self.dataset.num_tests:
                print(iter_time)

                tic = time.time()
                imgs, img_names = self.model.test_step()
                total_time += time.time() - tic

                self.model.plots_test(imgs, img_names, self.test_out_dir)
                iter_time += 1

            logger.info('Avg. PT: {:.2f} msec.'.format(total_time / self.dataset.num_tests * 1000.))

        except KeyboardInterrupt:
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        except tf.errors.OutOfRangeError:
            coord.request_stop()
        finally:
            # when done, ask the threads to stop
            coord.request_stop()
            coord.join(threads)

    def sample(self, iter_time):
        if np.mod(iter_time, self.flags.sample_freq) == 0:
            imgs = self.model.sample_imgs()
            self.model.plots(imgs, self.iter_time, self.dataset.image_size, self.sample_out_dir)

    def save_model(self, iter_time):
        if np.mod(iter_time + 1, self.flags.save_freq) == 0:
            model_name = 'model'
            self.saver.save(self.sess, os.path.join(self.model_out_dir, model_name), global_step=self.iter_time)
            logger.info(' [*] Model saved! Iter: {}'.format(iter_time))

    def load_model(self):
        logger.info(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(self.model_out_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_out_dir, ckpt_name))

            meta_graph_path = ckpt.model_checkpoint_path + '.meta'
            self.iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

            logger.info(' [*] Load iter_time: {}'.format(self.iter_time))
            return True
        else:
            return False
