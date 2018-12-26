import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
import functools

WEIGHTS_INIT_STDEV = .1
weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None


def transform(image, scope='transform', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        conv1 = conv(image, 32, 9, 1, scope='conv1')
        conv2 = conv(conv1, 64, 3, 2, scope='conv2')
        conv3 = conv(conv2, 128, 3, 2, scope='conv3')
        resid1 = resblock(conv3, 3, scope='resblock_1')
        resid2 = resblock(resid1, 3, scope='resblock_2')
        resid3 = resblock(resid2, 3, scope='resblock_3')
        resid4 = resblock(resid3, 3, scope='resblock_4')
        resid5 = resblock(resid4, 3, scope='resblock_5')
        conv_t1 = conv_tranpose(resid5, 64, 3, 2, scope='convt_1')
        conv_t2 = conv_tranpose(conv_t1, 32, 3, 2, scope='convt_2')
        conv_t3 = conv(conv_t2, 3, 9, 1, relu=False, scope='conv_3')
        preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    return preds


def conv(net, channels, filter_size, strides, relu=True, scope='conv_0'):
    with tf.variable_scope(scope):
        weights_init = conv_init_vars(net, channels, filter_size)
        strides_shape = [1, strides, strides, 1]
        net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
        net = instance_norm(net)
        if relu:
            net = tf.nn.relu(net)

    return net


def conv_tranpose(net, channels, filter_size, strides, scope='convt_0'):
    with tf.variable_scope(scope):
        # weights_init = conv_init_vars(net, channels, filter_size, transpose=True)
        #
        # batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
        # new_rows, new_cols = int(rows * strides), int(cols * strides)
        # # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])
        #
        # new_shape = [batch_size, new_rows, new_cols, channels]
        # tf_shape = tf.stack(new_shape)
        # strides_shape = [1,strides,strides,1]

        # net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
        net = tf.layers.conv2d_transpose(inputs=net, filters=channels,
                                       kernel_size=filter_size, kernel_initializer=weight_init,
                                       kernel_regularizer=weight_regularizer,
                                       strides=strides, padding='SAME', use_bias=True)

        net = instance_norm(net)
        net = tf.nn.relu(net)

    return net


def resblock(x_in, filter_size=3, scope='resblock_0'):
    with tf.variable_scope(scope):
        res1 = conv(x_in, 128, filter_size, 1, scope='res1')
        res2 = conv(res1, 128, filter_size, 1, relu=False, scope='res2')
        x = x_in + res2
    return x


# def instance_norm(net, train=True):
#     batch, rows, cols, channels = [i.value for i in net.get_shape()]
#     var_shape = [channels]
#     mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
#     shift = tf.Variable(tf.zeros(var_shape))
#     scale = tf.Variable(tf.ones(var_shape))
#     epsilon = 1e-3
#     normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
#     return scale * normalized + shift

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)


def conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    return weights_init


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def L2_loss(x, y):
    # tf.nn.l2_loss = sum(t ** 2) / 2
    # return tf.reduce_mean(tf.square(x-y))
    return tf.reduce_mean(tf.nn.l2_loss(x - y))


def L1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))


def norm_loss(x, y):
    return tf.sqrt(tf.reduce_sum(tf.square(x - y)))


def squared_fro(x, y):
    return tf.reduce_sum(tf.square(x - y))


"""
calculate gram matrix of a tensor with shape mode (batch_size, height, width, channel)
"""
def gram_matrix(x):
    assert isinstance(x, tf.Tensor)
    b, h, w, ch = x.get_shape().as_list()
    print("gram_matrix input shapes ", b, h, w, ch)
    features = tf.reshape(x, [-1, h*w, ch])
    # if h * w < ch:
    #     gram = tf.matmul(features, features, transpose_b=True) / tf.constant(ch * w * h, tf.float32)
    # else:
    gram = tf.matmul(features, features, transpose_a=True) / tf.constant(ch * w * h, tf.float32)
    print("gram_matrix shapes ", gram.get_shape())
    return gram


def content_recon_loss(y, y_hat, loss_type='euc'):
    tensor_size = get_tensor_size(y)
    print("content_recon_loss - tensor_size = ", tensor_size)
    if loss_type == 'euc':
        return L2_loss(y, y_hat)*2  # which defined in paper-Perceptual Loss ...
    elif loss_type == 'L2':
        return L2_loss(y, y_hat)
    elif loss_type == 'L1':
        return L1_loss(y, y_hat)
    else:
        return L2_loss(y, y_hat) * 2


def style_recon_loss(gram, gram_hat, loss_type='L2'):
    gram_size = get_tensor_size(gram)
    print("style_recon_loss - gram_size = ", gram_size)
    if loss_type == 'L2':
        return L2_loss(gram, gram_hat)
    if loss_type == 'squared_fro':
        return squared_fro(gram, gram_hat)  # which defined in paper-Perceptual Loss ...
    else:
        return squared_fro(gram, gram_hat)


def total_variation_loss(x, batch_size):
    b, h, w, ch = x.get_shape().as_list()
    tv_h = L2_loss(x[:, 1:, :, :], x[:, :h - 1, :, :])
    tv_w = L2_loss(x[:, :, 1:, :], x[:, :, :w - 1, :])
    return 2 * (tv_h + tv_w) / batch_size


# def gram_loss(y, y_pred):

# def gram_matrix(v):
#     assert isinstance(v, tf.Tensor)
#     v.get_shape().assert_has_rank(4)
#
#     dim = v.get_shape().as_list()
#     v = tf.reshape(v, [dim[1] * dim[2], dim[3]])
#     if dim[1] * dim[2] < dim[3]:
#         return tf.matmul(v, v, transpose_b=True)/get_tensor_size(v)
#     else:
#         return tf.matmul(v, v, transpose_a=True)/get_tensor_size(v)




"""
calculate shape[0] * shape[1] * ... * shape[n]
"""
def get_tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)