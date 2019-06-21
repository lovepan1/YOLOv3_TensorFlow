import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow
import os
import numpy as np
from model import yolov3
from utils.misc_utils import parse_anchors, read_class_names
from utils.layer_utils import conv2d, darknet53_body, yolo_block, upsample_layer
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

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
    # print(indices)
    return values, indices

def calculate_number_of_sparse_parameters(sparse_layers):

    total_count = 0

    for layer in sparse_layers:

        total_count += layer.values.nbytes // 4
        total_count += layer.indices.nbytes // 4
        total_count += layer.dense_shape.nbytes // 4
        total_count += layer.bias.nbytes // 4

    return total_count

def get_saved_filetrs_num(weight, pencentage=0.8):
    '''
    :param weight: conv layer weights
    :param pencentage: prue pencentage filters for layer weight
    :return: filters num, saved filters id
    '''
    flast = np.sum(weight, (0, 1, 2))
    saved_index = []
    sorted_flast = sorted(flast)
    th = sorted_flast[int(len(sorted_flast) * pencentage)]
    saved_filters_num = int(len(flast) * (1 - pencentage))
    for index, layer in enumerate(flast):
        if flast[index] >= th:
            saved_index.append(index)
    print('saved_index is ',saved_index)
    print('saved_filters_num is ', saved_filters_num)
    if saved_filters_num != len(saved_index):
        print("saved_filters_num != len(saved_index)")
    print(len(saved_index))
    np_saved_index = np.where(flast > th)
    return saved_filters_num, saved_index, np_saved_index

def reset_weights(raw_weights, kmeans_weights):
    shape = raw_weights.shape
    count = 0
    for r in range(shape[0]):
        for n in range(shape[3]):
            for c in range(shape[2]):
                for l in range(shape[1]):
                    #replace the non-zero raw weight with corresponding clustered weight
                    if raw_weights[r,l,c,n]:
                        raw_weights[r,l,c,n] = kmeans_weights[count]
                        count = count + 1
    return raw_weights

anchor_path = "./data/yolo_anchors.txt"
class_name_path  = "./data/my_data/dianli_class.names"
anchors = parse_anchors(anchor_path)
num_class = len(read_class_names(class_name_path))
tensor_name = "yolov3/yolov3_head/Conv_7/weights:0"

yolo_model = yolov3(num_class, anchors)
model_dir = '/home/pcl/tensorflow1.12/YOLOv3_TensorFlow/checkpoint/'
checkpoint_path = os.path.join(model_dir, "pure_model")

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
sparse_layers = []
layer_weight_sum = []
quantized_layer_weight = []

with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, img_size[1], img_size[0], 3], name='input_data')
    yolo_model = yolov3(num_class, anchors)
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)
    ########################################pruning################################################

    for key in var_to_shape_map:
        layer_name.append(key + ":0")
        print(key)
        if "weights" in key:
            layer_weight.append(key + ":0")
            weight = sess.run(tf.get_default_graph().get_tensor_by_name(key + ":0"))
            layer_weights.append(weight)
            yiwei_weights = sess.run(tf.reshape(weight, [-1]))
            values, indices = get_sparse_values_indices(weight)
            print(values)
            print(indices)
            '''
            get each layer mask, but this not use
            '''
            tf_weights.append(tf.get_default_graph().get_tensor_by_name(key + ":0"))
            th = get_th(weight)
            mask = mask_for_big_values(weight, th)
            layer_mask.append(mask)

        if "biases" in key:
            bias = sess.run(tf.get_default_graph().get_tensor_by_name(key + ":0"))
            layer_bias.append(bias)
            tf_bias.append(tf.get_default_graph().get_tensor_by_name(key + ":0"))

    layer_name_weights = dict(zip(layer_weight, layer_weights))
    layer_name_masks = dict(zip(layer_weight, layer_mask))
    # plot_histogram(layer_weights, 'weights_distribution_before_pruning', include_zeros=False)

    for (weight_matrix, tf_weight_matrix) in zip(layer_weights, tf_weights):
        th = get_th(weight_matrix)
        print("th is ", th)
        mask = mask_for_big_values(weight_matrix, th)
        sess.run(tf.assign(tf_weight_matrix, weight_matrix * mask))

    '''write new histogram after pruning model'''
    new_weights = sess.run(tf_weights)
    plot_histogram(new_weights, 'weights_distribution_after_pruning', include_zeros=False)
    saver.save(sess, os.path.join(model_dir, 'prue_model.ckpt'))

####################################################quantized#####################################
    for key in var_to_shape_map:
        layer_name.append(key + ":0")
        print(key)
        if "weights" in key:
            layer_weight.append(key + ":0")
            weight = sess.run(tf.get_default_graph().get_tensor_by_name(key + ":0"))
            layer_weights.append(weight)
            yiwei_weights = sess.run(tf.reshape(weight, [-1]))
            yiwei_weights = np.reshape(weight, [-1])
            # for i, w in enumerate(yiwei_weights):
            #         yibai_yiwei_weights.append(w)
            min_val = sorted(yiwei_weights)[0]
            max_val = sorted(yiwei_weights)[len(yiwei_weights) - 1]
            bits = 8
            linspace = np.linspace(min_val, max_val, num=2 ** bits)
            Kmeans = KMeans(n_clusters=len(linspace), init=linspace.reshape(-1, 1), n_init=1, precompute_distances=True,
                            algorithm="full")
            # perform Kmeans proc
            Kmeans.fit(np.array(yiwei_weights).reshape(-1, 1))
            # obtain Kmeans result
            new_weights = Kmeans.cluster_centers_[Kmeans.labels_].reshape(-1)
            quantized_weight = reset_weights(weight, new_weights)
            quantized_layer_weight.append(quantized_weight)

        if "biases" in key:
            bias = sess.run(tf.get_default_graph().get_tensor_by_name(key + ":0"))
            layer_bias.append(bias)
            tf_bias.append(tf.get_default_graph().get_tensor_by_name(key + ":0"))

    layer_name_weights = dict(zip(layer_weight, layer_weights))
    layer_name_masks = dict(zip(layer_weight, layer_mask))
    # plot_histogram(layer_weights, 'weights_distribution_before_pruning', include_zeros=False)


    for (quantized_weight_matrix, tf_weight_matrix) in zip(quantized_layer_weight, tf_weights):
        sess.run(tf.assign(tf_weight_matrix, quantized_weight_matrix))

    '''write new histogram after pruning model'''
    new_weights = sess.run(tf_weights)
    plot_histogram(new_weights, 'weights_distribution_after_pruning', include_zeros=False)

    saver.save(sess ,os.path.join(model_dir, 'model_quantized.ckpt'))

    # for weights, bias in zip(weight_matrices, biases):