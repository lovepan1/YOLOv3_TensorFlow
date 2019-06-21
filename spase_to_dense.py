import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow
import os
import numpy as np
from model import yolov3
from utils.misc_utils import parse_anchors, read_class_names
from utils.layer_utils import conv2d, darknet53_body, yolo_block, upsample_layer
import collections
from matplotlib import pyplot as plt
from pruning_model import sparse_yolov3

f = open('node.txt', 'w')
w = open("weights.txt", "w")
w2 = open("spase_we.txt", "w")
def plot_histogram(weights_list: list,
                   image_name: str,
                   include_zeros=True):

    """A function to plot weights distribution"""

    weights = []
    for w in weights_list:
        weights.extend(list(w.ravel()))

    if not include_zeros:
        weights = [w for w in weights if w != 0]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    ax.hist(weights,
            bins=100,
            facecolor='green',
            edgecolor='black',
            alpha=0.7,
            range=(-0.15, 0.15))

    ax.set_title('Weights distribution')
    ax.set_xlabel('Weights values')
    ax.set_ylabel('Number of weights')

    fig.savefig(image_name + '.png')

def mask_for_big_values(weights, pruning_threshold):
    '''
    :param weights: conv layer weights
    :param pruning_threshold:
    :return:
    '''

    small_weights = np.abs(weights) < pruning_threshold
    return np.logical_not(small_weights)

def prune_weights(weights, pruning_threshold):

    small_weights = np.abs(weights) < pruning_threshold
    weights[small_weights] = 0
    values = weights[weights != 0]
    indices = np.transpose(np.nonzero(weights))
    return values, indices

def get_th(weight, pencentage=0.8):
    '''
    :param weight: conv layer weights
    :param pencentage: prue pencentage for layer weight
    :return:
    '''
    flat = np.reshape(weight, [-1])
    flat_list = sorted(map(abs, flat))
    return flat_list[int(len(flat_list) * pencentage)]

def get_weights_from_ckpt(check_point_dir):
    '''
    :param check_point_dir: tensorflow ckpt dir
    :return: None
    '''
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        weight = reader.get_tensor(key)
        yiwei_weight = np.reshape(weight, [-1])
        print(yiwei_weight)

def get_sparse_values_indices(weights):
    '''

    :param weights: conv layer weights
    :return: non zero weights and non zero weights indices
    '''
    values = weights[weights != 0]
    # print('values is ', values)
    # for i in values:
    #     w2.write(str(i) + '\n')
    # print('len values', len(values))
    indices = np.transpose(np.nonzero(weights))
    # print(np.nonzero(weights))
    print("indices is ", indices)
    return values, indices

def calculate_number_of_sparse_parameters(sparse_layers):

    total_count = 0

    for layer in sparse_layers:

        total_count += layer.values.nbytes // 4
        total_count += layer.indices.nbytes // 4
        total_count += layer.dense_shape.nbytes // 4
        total_count += layer.bias.nbytes // 4

    return total_count

class SparseLayer(collections.namedtuple('SparseLayer',
                                         ['values',
                                          'indices',
                                          'dense_shape',
                                          'bias'])):

    """An auxilary class to represent sparse layer"""
    pass

anchor_path = "./data/yolo_anchors.txt"
class_name_path  = "./data/my_data/dianli_class.names"
anchors = parse_anchors(anchor_path)
num_class = len(read_class_names(class_name_path))
tensor_name = "yolov3/yolov3_head/Conv_7/weights:0"

yolo_model = yolov3(num_class, anchors)
model_dir = '/home/pcl/tensorflow1.12/YOLOv3_TensorFlow/checkpoint/'
checkpoint_path = os.path.join(model_dir, "model.ckpt")

# graph = tf.get_default_graph()  # 获得默认的图
# input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
graph = tf.get_default_graph()  # 获得默认的图
input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

# layer_weight_metrics = {}
layer_weights = []
layer_name = []
layer_th = []
layer_mask = []
layer_bias = []
img_size = [412, 412]
layer_weight = []
tf_weights = []
tf_bias = []


with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, img_size[1], img_size[0], 3], name='input_data')
    yolo_model = yolov3(num_class, anchors)
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)
    weight = sess.run(tf.get_default_graph().get_tensor_by_name(tensor_name))
    print(weight.shape)
    values, indices = get_sparse_values_indices(weight)
    for key in var_to_shape_map:
        layer_name.append(key + ":0")
        print(key)
        if "weights" in key:
            layer_weights.append(sess.run(tf.get_default_graph().get_tensor_by_name(key + ":0")))
            tf_weights.append(tf.get_default_graph().get_tensor_by_name(key + ":0"))
            th = get_th(weight)
        if "biases" in key:
            layer_bias.append(sess.run(tf.get_default_graph().get_tensor_by_name(key + ":0")))
            tf_bias.append(tf.get_default_graph().get_tensor_by_name(key + ":0"))
    sparse_layers = []
    for true_weights, true_bias in zip(layer_weights, layer_bias):
        values, indices = get_sparse_values_indices(true_weights)
        shape = np.array(true_weights.shape).astype(np.int64)
        sparse_layers.append(SparseLayer(values=values.astype(np.float32),
                                                       indices=indices.astype(np.int16),
                                                       dense_shape=shape,
                                                       bias=true_bias))

    yolo_sparse_model = sparse_yolov3(num_class, anchors, sparse_layers)



    saver.save(sess ,os.path.join(model_dir, 'model.ckpt'))

