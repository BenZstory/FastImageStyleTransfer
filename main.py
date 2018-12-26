from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from FastStyle import FastStyle
import argparse
from utils import *
import tensorflow as tf

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of Fast Style Transfer"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or evaluate')
    parser.add_argument('--style_path', type=str, help='The file path of style image for training.')
    parser.add_argument('--train_data_dir', type=str, help='The Directory of training images.')

    parser.add_argument('--epoch', type=int, default=2, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=1000000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=4, help='The batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=1000, help='The number of ckpt_save_freq')
    parser.add_argument('--print_step', type=int, default=1, help='The number of steps that gonna print in one time.')

    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate')
    parser.add_argument('--style_w', type=float, default=100, help='weight of adversarial loss')
    parser.add_argument('--content_w', type=float, default=7.5, help='weight of adversarial loss')
    parser.add_argument('--tv_w', type=float, default=200, help='weight of adversarial loss')

    parser.add_argument('--style_net', type=int, default=5, help='The type of style layers set to use, 4 or 5')
    parser.add_argument('--content_net', type=int, default=4, help='The type of content layer to use, 4 or 5')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--model_dir', type=str, default='',
                        help='Directory to indicate and contain the training model')
    parser.add_argument('--train_log_root', type=str, default='',
                        help='Directory of train log root')
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='Directory to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='',
                        help='Directory to save training logs')
    parser.add_argument('--sample_dir', type=str, default='',
                        help='Directory name to save the samples on training')

    parser.add_argument('--vgg_path', type=str, default='',
                        help='Path to vgg16.npy')
    parser.add_argument('--evaluate_data_dir', type=str, default='test_icons',
                        help='Directory name when stored the icons to evaluate')

    parser.add_argument('--from_checkpoint', type=str, default='',
                        help='Directory to load the checkpoints')

    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    if args.phase == 'train':
        if os.path.isdir(args.style_path):
            filenames = os.listdir(args.style_path)
            for filename in filenames:
                if os.path.splitext(filename)[1] in ('.png', '.jpg'):
                    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                        model = FastStyle(sess, args, os.path.join(args.style_path, filename))
                        model.build_model()
                        ckpt_dir = model.train()
                        model.evaluate(ckpt_dir)
                        sess.close()
                    tf.reset_default_graph()
                    print(" [*] Training of style-{} finished!\r\n".format(filename))
        else:
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                model = FastStyle(sess, args)
                model.build_model()
                ckpt_dir = model.train()
                model.evaluate(ckpt_dir)
                sess.close()
        print(" [*] Training finished!")

    if args.phase == 'evaluate':
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            model = FastStyle(sess, args)
            model.build_evaluate_model()
            model.evaluate()
            print(" [*] Test finished!")

    if args.phase == 'grid_search':
        assert not os.path.isdir(args.style_path)
        for lr in [0.003, 0.0001, 0.00005]: #[0.01, 0.003, 0.006, 0.0001]:
            for style_w in [5e-4, 1e-4, 3e-5, 5e-5]: #[1e-4, 5e-5, 1e-5, 1e-6]:
                for tv_w in [0]:
                    for style_net in [5]: #[4, 5]:
                        for content_net in [4]: #[4, 5]:
                            args.lr = lr
                            args.style_w = style_w
                            args.tv_w = tv_w
                            args.style_net = style_net
                            args.content_net = content_net
                            grid_name = 'lr_' + str(lr) + '_' + \
                                        'stylew_' + str(style_w) + '_' + \
                                        'tvw_' + str(tv_w) + '_' + \
                                        'stylenet_' + str(style_net) + '_' + \
                                        'contentnet_' + str(content_net)
                            args.model_dir = os.path.join(args.train_log_root, 'FastStyle_GridSearch_1228', grid_name)
                            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                                print('ready to train')
                                model = FastStyle(sess, args)
                                model.build_model()
                                ckpt_dir = model.train()
                                model.evaluate(ckpt_dir)
                                sess.close()
                            tf.reset_default_graph()
        print(" [*] GridSearch finished!")

    if args.phase == 'grid_search1':
        assert not os.path.isdir(args.style_path)
        for lr in [0.003, 0.0001]: #[0.01, 0.003, 0.006, 0.0001]:
            for style_w in [5e-06, 1e-06, 5e-07, 1e-07, 1e-08]: #[1e-4, 5e-5, 1e-5, 1e-6]:
                for tv_w in [0]:
                    for style_net in [4]: #[4, 5]:
                        for content_net in [5]: #[4, 5]:
                            args.lr = lr
                            args.style_w = style_w
                            args.tv_w = tv_w
                            args.style_net = style_net
                            args.content_net = content_net
                            grid_name = 'lr_' + str(lr) + '_' + \
                                        'stylew_' + str(style_w) + '_' + \
                                        'tvw_' + str(tv_w) + '_' + \
                                        'stylenet_' + str(style_net) + '_' + \
                                        'contentnet_' + str(content_net)
                            args.model_dir = os.path.join(args.train_log_root, 'FastStyle_GridSearch_1228', grid_name)
                            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                                print('ready to train')
                                model = FastStyle(sess, args)
                                model.build_model()
                                ckpt_dir = model.train()
                                model.evaluate(ckpt_dir)
                                sess.close()
                            tf.reset_default_graph()
        print(" [*] GridSearch finished!")

if __name__ == '__main__':
    main()