# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

leaky_alpha = 0.1
xavier_initializer = tf.initializers.glorot_uniform()
def conv_block_mobilet(x, filters, stride, out_channel, net_type, is_training, name='', relu=True):
    """
    :param x: input :nhwc
    :param filters: list [f_w, f_h]
    :param stride: list int
    :param out_channel: int, out_channel
    :param net_type: cnn mobilenet
    :param is_training: used in BN
    :param name: str
    :param relu: boolean
    :return: depwise and pointwise out
    """


    with tf.name_scope('' + name):
        in_channel = x.shape[3].value
        if net_type == 'mobilenetv1':
            with tf.name_scope('depthwise'):
                # depthwise_weight = tf.Variable(tf.truncated_normal([filters[0], filters[1], in_channel, 1], 0, 0.01))
                depthwise_weight = tf.Variable(xavier_initializer([filters[0], filters[1], in_channel, 1]))
                x = tf.nn.depthwise_conv2d(x, depthwise_weight, [1, stride[0], stride[1], 1], 'SAME')
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.relu6(x)

            with tf.name_scope('pointwise'):
                # pointwise_weight = tf.Variable(tf.truncated_normal([1, 1, in_channel, out_channel], 0, 0.01))
                pointwise_weight = tf.Variable(xavier_initializer([1, 1, in_channel, out_channel]))
                x = tf.nn.conv2d(x, pointwise_weight, [1, 1, 1, 1], 'SAME')
                if relu:
                    x = tf.layers.batch_normalization(x, training=is_training)
                    x = tf.nn.relu6(x)
                else:
                    bias = tf.Variable(tf.zeros(shape=out_channel))
                    x += bias

        elif net_type == 'mobilenetv2':
            tmp_channel = out_channel * 3
            with tf.name_scope('expand_pointwise'):
                pointwise_weight = tf.Variable(xavier_initializer([1, 1, in_channel, tmp_channel]))
                x = tf.nn.conv2d(x, pointwise_weight, [1, 1, 1, 1], 'SAME')
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.relu6(x)
            with tf.name_scope('depthwise'):
                depthwise_weight = tf.Variable(xavier_initializer([filters[0], filters[1], tmp_channel, 1]))
                x = tf.nn.depthwise_conv2d(x, depthwise_weight, [1, stride[0], stride[1], 1], 'SAME')
            with tf.name_scope('project_pointwise'):
                pointwise_weight = tf.Variable(xavier_initializer([1, 1, tmp_channel, out_channel]))
                x = tf.nn.conv2d(x, pointwise_weight, [1, 1, 1, 1], 'SAME')
                if relu:
                    x = tf.layers.batch_normalization(x, training=is_training)
                else:
                    bias = tf.Variable(tf.zeros(shape=out_channel))
                    x += bias
        else:
            raise Exception('net type is error, please check')
    return x


def residual_mobilenet(x, net_type, is_training, out_channel=1, expand_time=1, stride=1):
    if net_type in ['cnn', 'mobilenetv1']:
        out_channel = x.shape[3].value
        shortcut = x
        x = conv_block_mobilet(x, [1, 1], [1, 1], out_channel // 2, net_type='cnn', is_training=is_training)
        x = conv_block_mobilet(x, [3, 3], [1, 1], out_channel, net_type='cnn', is_training=is_training)
        x += shortcut

    elif net_type == 'mobilenetv2':
        shortcut = x
        in_channel = x.shape[3].value
        tmp_channel = in_channel * expand_time
        with tf.name_scope('expand_pointwise'):
            pointwise_weight = tf.Variable(xavier_initializer([1, 1, in_channel, tmp_channel]))
            x = tf.nn.conv2d(x, pointwise_weight, [1, 1, 1, 1], 'SAME')
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.relu6(x)
        with tf.name_scope('depthwise'):
            depthwise_weight = tf.Variable(xavier_initializer([3, 3, tmp_channel, 1]))
            x = tf.nn.depthwise_conv2d(x, depthwise_weight, [1, stride, stride, 1], 'SAME')
        with tf.name_scope('project_pointwise'):
            pointwise_weight = tf.Variable(xavier_initializer([1, 1, tmp_channel, out_channel]))
            x = tf.nn.conv2d(x, pointwise_weight, [1, 1, 1, 1], 'SAME')
            x = tf.layers.batch_normalization(x, training=is_training)
        x += shortcut
    return x

def darknet53_mobilenet_body(inputs, net_type, is_training):

    x = conv_block_mobilet(inputs, [3, 3], [1, 1], 32, net_type, is_training=is_training)

    # down sample
    x = conv_block_mobilet(x, [3, 3], [2, 2], 64, net_type, is_training=is_training)
    for i in range(1):
        x = residual_mobilenet(x, net_type, is_training)

    # down sample
    x = conv_block_mobilet(x, [3, 3], [2, 2], 128, net_type, is_training=is_training)
    for i in range(2):
        x = residual_mobilenet(x, net_type, is_training)

    # down sample
    x = conv_block_mobilet(x, [3, 3], [2, 2], 256, net_type, is_training=is_training)
    for i in range(8):
        x = residual_mobilenet(x, net_type, is_training)
    route_1 = x

    # down sample
    x = conv_block_mobilet(x, [3, 3], [2, 2], 512, net_type, is_training=is_training)
    for i in range(8):
        x = residual_mobilenet(x, net_type, is_training)
    route_2 = x

    # down sample
    x = conv_block_mobilet(x, [3, 3], [2, 2], 1024, net_type, is_training=is_training)
    for i in range(4):
        net = residual_mobilenet(x, net_type, is_training)
    route_3 = net

    return route_1, route_2, route_3

def yolo_block_mobilenet(inputs, filters, net_type, is_training):

    net = conv_block_mobilet(inputs, [1, 1], [1, 1], filters * 1, net_type, is_training=is_training)
    net = conv_block_mobilet(net, [3, 3], [1, 1], filters * 2, net_type, is_training=is_training)
    net = conv_block_mobilet(net, [1, 1], [1, 1], filters * 1, net_type, is_training=is_training)
    net = conv_block_mobilet(net, [3, 3], [1, 1], filters * 2, net_type, is_training=is_training)
    net = conv_block_mobilet(net, [1, 1], [1, 1], filters * 1, net_type, is_training=is_training)
    route = net
    net = conv_block_mobilet(net, [3, 3], [1, 1], filters * 1, net_type, is_training=is_training)

    return route, net
