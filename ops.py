import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
import functools

WEIGHTS_INIT_STDEV = .1
weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None


def transform(image, scope='transform', reuse=False, tanh_constant=150.0):
    with tf.variable_scope(scope, reuse=reuse):
        channel = 32
        x = conv_block(image, channel, 9, 1, norm='in', activation='relu', scope='conv1')

        for i in range(2):
            x = conv_block(x, channel*2, 3, 2, norm='in', activation='relu', scope='conv_down_' + str(i))
            channel = channel * 2

        for i in range(5):
            x = resblock(x, 128, scope='resblock_' + str(i))

        for i in range(2):
            x = deconv_block(x, channel//2, 3, 2, norm='in', activation='relu', scope='deconv_' + str(i))
            channel = channel // 2

        channel = 3
        x = conv(x, channels=channel, kernel=9, stride=1, pad=4, pad_type='reflect', use_bias=True, scope='logit')
        x = tf.nn.tanh(x) * tanh_constant + 255./2

    return x


##################################################################################
# Layers
##################################################################################
import math
def conv_block(x, channels, kernel=3, stride=2, use_bias=True,
               norm='bn', activation='relu', scope='conv_block'):
    with tf.variable_scope(scope):
        pad = int(math.ceil((kernel - stride) / 2))
        x = conv(x, channels, kernel, stride, pad=pad, use_bias=use_bias, sn=(norm == 'sn'))

        # norm
        if norm == 'bn':
            x = batch_norm(x)
        elif norm == 'in':
            x = instance_norm(x)
        elif norm == 'gn':
            x = group_norm(x)

        # activation
        if activation == 'relu':
            x = relu(x)
        elif activation == 'lrelu':
            x = lrelu(x)
        elif activation == 'tanh':
            x = tanh(x)
        elif activation == 'sigmoid':
            x = sigmoid(x)

        return x


def deconv_block(x, channels, kernel=3, stride=2, use_bias=True,
               norm='bn', activation='relu', scope='deconv_block'):
    with tf.variable_scope(scope):
        x = deconv(x, channels, kernel, stride, use_bias=use_bias, sn=(norm == 'sn'))

        # norm
        if norm == 'bn':
            x = batch_norm(x)
        elif norm == 'in':
            x = instance_norm(x)
        elif norm == 'gn':
            x = group_norm(x)

        # activation
        if activation == 'relu':
            x = relu(x)
        elif activation == 'lrelu':
            x = lrelu(x)
        elif activation == 'tanh':
            x = tanh(x)
        elif activation == 'sigmoid':
            x = sigmoid(x)

        return x


def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if (kernel - stride) % 2 == 0 :
            pad_top = pad
            pad_bottom = pad
            pad_left = pad
            pad_right = pad

        else :
            pad_top = pad
            pad_bottom = kernel - stride - pad_top
            pad_left = pad
            pad_right = kernel - stride - pad_left

        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x


def deconv(x, channels, kernel=4, stride=2, use_bias=True, sn=False, scope='deconv_0'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()
        output_shape = [x_shape[0], x_shape[1]*stride, x_shape[2]*stride, channels]
        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init, regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')

            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                                           strides=stride, padding='SAME', use_bias=use_bias)

        return x


##################################################################################
# Residual-block
##################################################################################
def resblock(x_init, channels, use_bias=True, scope='resblock_0'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)

        return x + x_init


##################################################################################
# Activation function
##################################################################################
def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)


def sigmoid(x) :
    return tf.sigmoid(x)


##################################################################################
# Normalization function
##################################################################################
def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)


def layer_norm(x, scope='layer_norm') :
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)


def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def group_norm(x, groups=32):
    return tf_contrib.layers.group_norm(x, groups=groups, epsilon=1e-06)


##################################################################################
# Loss function
##################################################################################
def L2_loss(x, y):
    # tf.nn.l2_loss = sum(t ** 2) / 2
    # return tf.reduce_sum(tf.square(x-y))
    return 2 * tf.nn.l2_loss(x-y)


def MSE(x, y):
    return tf.reduce_mean(tf.nn.l2_loss(x - y))


def L1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))

"""
calculate gram matrix of a tensor with shape mode (batch_size, height, width, channel)
"""
def gram_matrix(x):
    assert isinstance(x, tf.Tensor)
    b, h, w, ch = x.get_shape().as_list()
    print("gram_matrix input shapes ", b, h, w, ch)
    features = tf.reshape(x, [-1, h*w, ch])
    gram = tf.matmul(features, features, transpose_a=True) / tf.constant(ch * w * h, tf.float32)
    print("gram_matrix shapes ", gram.get_shape())
    return gram


def content_recon_loss(y, y_hat, loss_type='MSE'):
    tensor_size = get_tensor_size(y)
    print("content_recon_loss - tensor_size = ", tensor_size)
    if loss_type == 'MSE':
        return MSE(y, y_hat)
    elif loss_type == 'L2':
        return L2_loss(y, y_hat)
    elif loss_type == 'L1':
        return L1_loss(y, y_hat)
    else:
        return MSE(y, y_hat)


def style_recon_loss(gram, gram_hat, loss_type='MSE'):
    gram_size = get_tensor_size(gram)
    print("style_recon_loss - gram_size = ", gram_size)
    if loss_type == 'L2':
        return L2_loss(gram, gram_hat)
    elif loss_type == 'MSE':
        return MSE(gram, gram_hat)  # which defined in paper-Perceptual Loss ...
    elif loss_type == 'L1':
        return L1_loss(gram, gram_hat)
    else:
        return MSE(gram, gram_hat)


def total_variation_loss(x, batch_size):
    return tf.reduce_sum(tf.image.total_variation(x)) / batch_size


"""
calculate shape[0] * shape[1] * ... * shape[n]
"""
def get_tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)