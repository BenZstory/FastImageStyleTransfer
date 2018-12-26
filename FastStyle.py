import os
# from ops import *
from utils import *
import shutil
# from .utils import utils
# from .utils import ImageData, load_image, check_folder
from ops import gram_matrix, transform, content_recon_loss, style_recon_loss, total_variation_loss
import tensorflow as tf
import numpy as np
from glob import glob
from vgg16 import Vgg16
import time
from datetime import datetime

# CONTENT_LAYER = 'relu4_2'
# STYLE_LAYERS = ('conv1_1', 'conv2_1', 'conv3_1', 'relu4_1', 'relu5_1')

STYLE_LAYERS4    = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']
STYLE_LAYERS5    = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

CONTENT_LAYERS4  = ['conv4_2']
CONTENT_LAYERS5  = ['conv5_2']

class FastStyle(object):
    def __init__(self, sess, args, style_img_path=""):
        self.model_name = 'FastStyle'
        self.sess = sess
        self.args_dict = vars(args)

        self.train_log_root = args.train_log_root
        self.model_dir = args.model_dir
        self.result_dir = args.result_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.log_dir = args.log_dir

        self.img_size = args.img_size
        self.img_h = args.img_size
        self.img_w = args.img_size
        self.img_ch = args.img_ch

        self.init_lr = args.lr
        self.epoch = args.epoch
        self.iteration = args.iteration
        self.batch_size = args.batch_size
        self.save_freq = args.save_freq
        self.print_freq = args.print_freq
        self.print_step = args.print_step

        self.style_w = args.style_w
        self.content_w = args.content_w
        self.tv_w = args.tv_w
        self.style_net = args.style_net
        self.content_net = args.content_net

        self.from_checkpoint = args.from_checkpoint

        self.train_data_dir = args.train_data_dir
        self.train_dataset = glob('./{}/*.*'.format(self.train_data_dir))
        self.evaluate_data_dir = args.evaluate_data_dir
        self.evaluate_dataset = glob('./{}/*.*'.format(self.evaluate_data_dir))
        self.dataset_num = max(len(self.train_dataset), len(self.evaluate_data_dir))

        if style_img_path == "":
            self.style_image_path = args.style_path
        else:
            self.style_image_path = style_img_path
        self.vgg_path = args.vgg_path
        self.evaluate_data_dir = args.evaluate_data_dir

        if self.iteration == 0:
            self.iteration = int(self.dataset_num / self.batch_size)


    def build_model(self):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ style target part, style target image -> style target grams """
        print("self.style_image_path = ", self.style_image_path)

        if self.style_net == 4:
            STYLE_LAYERS = STYLE_LAYERS4
        else:
            STYLE_LAYERS = STYLE_LAYERS5

        if self.content_net == 4:
            CONTENT_LAYERS = CONTENT_LAYERS4
        else:
            CONTENT_LAYERS = CONTENT_LAYERS5

        self.style_image = get_image(self.style_image_path).astype(np.float32)
        style_batch = np.expand_dims(self.style_image, 0)
        styles = tf.constant(style_batch, dtype=tf.float32, shape=style_batch.shape)

        print("self.style_image.shape = ", self.style_image.shape)
        style_net = Vgg16(self.vgg_path)
        style_net.build(styles)

        self.style_target_grams = {}
        for layer in STYLE_LAYERS:
            style = style_net.layer_dict[layer]
            self.style_target_grams[layer] = gram_matrix(style)

        img_loader = ImageData(load_size=self.img_size, channels=self.img_ch)
        """ content target part, content target images -> vgg16 relu3_3 feature"""
        content = tf.data.Dataset.from_tensor_slices(self.train_dataset)
        # TODO what if the dataset is so big that tf can't shuffle it?
        content = content.shuffle(self.dataset_num, reshuffle_each_iteration=True).repeat()
        content = content.map(img_loader.image_processing).batch(self.batch_size)
        content_iterator = content.make_one_shot_iterator()
        self.content = content_iterator.get_next()

        content_net = Vgg16(self.vgg_path)
        content_net.build(self.content)

        self.content_targets_features = {}
        for layer in CONTENT_LAYERS:
            self.content_targets_features[layer] = content_net.layer_dict[layer]


        """ transform part, input image -> stylized image """
        # TODO which value scale should we use, [-1, 1] or [0, 1]? now later one.
        content_input = (self.content + 1) / 2.0
        self.stylized = transform(content_input)

        # our transform output scale is [0, 255], yet vgg16 input scale is [-1, 1], convert now
        stylized_norm = tf.cast(self.stylized, tf.float32) / 127.5 - 1
        vgg_net = Vgg16(self.vgg_path)
        vgg_net.build(stylized_norm)

        """ all vgg nets are built, now calculate losses """
        """ content loss """
        self.content_loss = 0
        for layer in CONTENT_LAYERS:
            y_content = vgg_net.layer_dict[layer]
            self.content_loss += content_recon_loss(self.content_targets_features[layer], y_content)

        """" style loss """
        self.style_loss = 0
        for layer in STYLE_LAYERS:
            y_style = vgg_net.layer_dict[layer]
            gram = gram_matrix(y_style)
            self.style_loss += style_recon_loss(self.style_target_grams[layer], gram)
        self.style_loss = self.style_loss / self.batch_size

        """" total variation loss """
        self.tv_loss = total_variation_loss(self.stylized, self.batch_size)

        self.total_loss = self.content_w * self.content_loss + \
                          self.style_w * self.style_loss + \
                          self.tv_w * self.tv_loss

        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)

        self.summ_content_loss  = tf.summary.scalar("content_loss", self.content_loss)
        self.summ_style_loss    = tf.summary.scalar("style_loss", self.style_loss)
        self.summ_tv_loss       = tf.summary.scalar("tv_loss", self.tv_loss)

        self.summ_content_loss_with_w = tf.summary.scalar("content_loss_with_w", self.content_loss * self.content_w)
        self.summ_style_loss_with_w   = tf.summary.scalar("style_loss_with_w", self.style_loss * self.style_w)
        self.summ_tv_loss_with_w      = tf.summary.scalar("tv_loss_with_w", self.tv_loss * self.tv_w)

        self.summ_total_loss    = tf.summary.scalar("total_loss", self.total_loss)
        self.summ_losses        = tf.summary.merge([self.summ_content_loss, self.summ_style_loss, self.summ_tv_loss,
                                                    self.summ_content_loss_with_w, self.summ_style_loss_with_w,
                                                    self.summ_tv_loss_with_w, self.summ_total_loss])

        self.eval_content = tf.placeholder(tf.float32, [1, self.img_h, self.img_w, self.img_ch], name='eval_image')
        self.eval_stylized = transform(self.content, reuse=True)


    def check_and_mkdirs(self):
        # check and make folders
        if self.model_dir == "":
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.train_log_root == "":
                self.model_dir = os.path.join("train_log", self.model_name + "_" + self.style_img_tag + "_" + current_time)
            else:
                self.model_dir = os.path.join(self.train_log_root, self.model_name + "_" + self.style_img_tag + "_" + current_time)
        check_folder(self.model_dir)

        if self.checkpoint_dir == "":
            self.checkpoint_dir = self.model_dir
        elif '/' not in self.checkpoint_dir:
            self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        check_folder(self.checkpoint_dir)

        if self.log_dir == "":
            self.log_dir = os.path.join(self.model_dir, "log")
        elif '/' not in self.log_dir:
            self.log_dir = os.path.join(self.log_dir, self.model_dir)
        check_folder(self.log_dir)

        if self.sample_dir == "":
            self.sample_dir = os.path.join(self.model_dir, "samples")
        elif '/' not in self.sample_dir:
            self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(os.path.join(self.sample_dir, "imgs"))

    def train(self):
        print("Training on style image : " + self.style_image_path)
        self.style_img_tag = str(os.path.basename(self.style_image_path).split('.')[0])
        self.check_and_mkdirs()
        self.total_sample_path = os.path.join(os.path.join(self.sample_dir, "_total_samples.html"))
        self.write_args_to_html()

        shutil.copy(self.style_image_path, os.path.join(self.model_dir, os.path.basename(self.style_image_path)))

        # init
        self.sess.run(tf.global_variables_initializer())

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.from_checkpoint)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # # load style
        # style_image = load_image(self.style_image_path)
        # self.sess.run(self.style_target_grams, feed_dict={self.style_image: style_image})

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            lr = self.init_lr * pow(0.5, epoch)

            for idx in range(start_batch_id, self.iteration):
                _, total_loss, stylized_images, summary_str = self.sess.run([
                    self.optimize, self.total_loss, self.stylized, self.summ_losses],
                    feed_dict={self.lr : lr})

                self.writer.add_summary(summary_str, counter)
                counter += 1
                print("Epoch: [%2d] [%6d/%6d] time: %4.4f total_loss: %.8f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, total_loss))

                ##  Save samples and write to html ##
                if np.mod(counter + self.print_step, self.print_freq) < self.print_step:
                    image_save_epoch = (counter + self.print_step) // self.print_freq
                    step_mod = np.mod(counter + self.print_step, self.print_freq)
                    html_name = "samples_" + str((counter + self.print_step) // self.print_freq) + '.html'
                    if np.mod(counter + self.print_step, self.print_freq) == 0:
                        with open(self.total_sample_path, 'a') as t_html:
                            t_html.write("<hr style=\"border-bottom: 3px solid red\" />\r\n<h3> Samples_of_" +
                                         str((counter + self.print_step) // self.print_freq) + " </h3>")

                    for j in range(0, self.batch_size):
                        img_id = step_mod * self.batch_size + j
                        save_one_img(stylized_images[j], './{}/imgs/stylized_{:02d}_{:06d}_{:02d}.jpg'.format(
                            self.sample_dir, epoch, idx + 1, img_id))
                    self.write_to_html(os.path.join(self.sample_dir, html_name), epoch, idx + 1, img_id)

                ##  Save checkpoint  ##
                if np.mod(idx + 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
        self.save(self.checkpoint_dir, counter)
        return self.checkpoint_dir

    def write_args_to_html(self):
        body = ""
        for k, v in self.args_dict.items():
            if str(v) == "":
                body = body + "--" + str(k) + " \"\"" + " \\<br>"
            else:
                body = body + "--" + str(k) + " " + str(v) + " \\<br>"
        with open(self.total_sample_path, 'a') as t_html:
            t_html.write("python3 main.py \\<br>")
            t_html.write(body)

    def write_to_html(self, html_path, epoch, idx, img_id):
        names = ['stylized']

        body = ""
        for name in names:
            image_name = '{}_{:02d}_{:06d}_{:02d}.jpg'.format(name, epoch, idx, img_id)
            body = body + str("<img src=\"" + os.path.join('imgs', image_name) + "\">")
        body = body + str("<br>")

        with open(html_path, 'a') as v_html:
            v_html.write(body)
        with open(self.total_sample_path, 'a') as t_html:
            t_html.write(body)

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def build_evaluate_model(self):
        self.eval_content = tf.placeholder(tf.float32, [1, self.img_h, self.img_w, self.img_ch], name='eval_image')
        content_input = (self.eval_content + 1) / 2.0
        self.eval_stylized = transform(content_input)


    def evaluate(self, checkpoint_dir = ""):
        evaluate_files = glob('./{}/*.*'.format(self.evaluate_data_dir))
        if len(evaluate_files) <= 0:
            print("Cant find evaluate files in " + self.evaluate_data_dir)
            return

        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        if checkpoint_dir != "":
            # checkpoint_dir assigned when call evaluate, probably called after training
            could_load, checkpoint_counter = self.load(checkpoint_dir)
        else:
            could_load, checkpoint_counter = self.load(self.from_checkpoint)
        assert could_load

        if self.model_dir == "":
            if os.path.isdir(self.from_checkpoint):
                self.model_dir = self.from_checkpoint
            else:
                self.model_dir = os.path.pardir(self.from_checkpoint)
        check_folder(self.model_dir)

        if self.result_dir == "":
            self.result_dir = os.path.join(self.model_dir, "result")
        else:
            self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)
        print("Evaluate and save result to ", self.result_dir)

        for file_path in evaluate_files:
            sample_image = np.asarray(load_image_np(file_path, size_h=self.img_h, size_w=self.img_w))
            result_path = os.path.join(self.result_dir, os.path.basename(file_path))

            stylized_img = self.sess.run(self.eval_stylized, feed_dict={self.eval_content: sample_image})
            save_one_img(stylized_img[0], result_path)
