import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow
import os
import numpy as np
from model import yolov3
from utils.misc_utils import parse_anchors, read_class_names
from utils.layer_utils import conv2d, darknet53_body, yolo_block, upsample_layer


anchor_path = "./data/yolo_anchors.txt"
class_name_path  = "./data/my_data/dianli_class.names"
anchors = parse_anchors(anchor_path)
num_class = read_class_names(class_name_path)


yolo_model = yolov3(num_class, anchors)
model_dir = '/home/pcl/tensorflow1.12/YOLOv3_TensorFlow/checkpoint/'
checkpoint_path = os.path.join(model_dir, "model.ckpt")
f = open('node.txt', 'w')
w = open("weights.txt", "w")
w2 = open("spase_we.txt", "w")
tensor_name = "yolov3/yolov3_head/Conv_7/weights:0"
node_name = "yolov3/yolov3_head/Conv_7/weights"
# pb =yolov3/darknet53_body/Conv_43/weights
# checkpoint_path = os.path.join(model_dir, "resnet50_csv_18.pb")
inputs = tf.placeholder(tf.float32, [1,412, 412, 3], name='input_data')
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
graph = tf.get_default_graph()  # 获得默认的图
input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
saver = tf.train.import_meta_graph(checkpoint_path + '.meta', clear_devices=True)


def get_th(weight, pencentage=0.8):
    flat = np.reshape(weight, [-1])
    flat_list = sorted(map(abs, flat))
    return flat_list[int(len(flat_list) * pencentage)]


def prune(weights, th):
    '''
    :param weights: weight Variable
    :param th: float value, weight under th will be pruned
    :return: sparse_weight
    '''
    shape = weights.shape
    # weight_arr = sess.run(weights)
    under_threshold = abs(weights) < th
    weights[under_threshold] = 0
    # tmp = weights
    # for i in range(len(shape) - 1):
    #     tmp = tmp[-1]
    # if tmp[-1] == 0:
    #     tmp[-1] == 0.01
    count = np.sum(under_threshold)
    print(count)
    return weights, ~under_threshold


def change_weights(orignal_weight, prune_weight, node_name):
    '''
    because this weight only one layer ,so use weight not weights
    '''
    sess.run(tf.assign(orignal_weight, prune_weight))
    return orignal_weight

def cvt_dim(layer_weight):
    shape = layer_weight.shape
    dim_2_list = np.zeros(shape=(shape[0], shape[1] * shape[2] * shape[3]))
    all_ele = layer_weight.flatten('F')
    count = 0
    for i in range(dim_2_list.shape[1]):
        for j in range(dim_2_list.shape[0]):
            dim_2_list[j, i] = all_ele[count]
            count = count + 1

    return dim_2_list

from scipy.sparse import csc_matrix,csr_matrix

with tf.Session() as sess:
    # saver = tf.train.Saver()
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    saver.restore(sess, checkpoint_path)
    weight = sess.run(tf.get_default_graph().get_tensor_by_name(tensor_name))
    yiwei_weight = np.reshape(weight, [-1])
    dim2_layer_weights = cvt_dim(weight)
    shape = dim2_layer_weights.shape
    # # use csc/csr mat to represent sparse weights
    if shape[0] < shape[1]:  # h<w. i.e. rows are small, the JA matrix will be short
        sparse_mat = csr_matrix(dim2_layer_weights)
    else:
        sparse_mat = csc_matrix(dim2_layer_weights)
    print(len(weight))
    # print(yiwei_weight)
    yiwei_sparse_weight = np.reshape(sparse_mat.data, [-1])
    print(yiwei_sparse_weight)
    for i in yiwei_weight:
        w.write(str(i) + '\n')
    for j in yiwei_sparse_weight:
        w2.write(str(j) + '\n')

    # for key in var_to_shape_map:
    #     print(str(key) + '\n')
    #     # print(sess.run(tf.get_default_graph().get_tensor_by_name(tensor_name)))
    #     weight = sess.run(tf.get_default_graph().get_tensor_by_name(tensor_name))
        # sess.run(tf.assign(yolo_model.forward(), a))
        # print("yolo_model_tensor", darknet[1])



    # print("orignal weights is ",weight)
    # shape = weight.shape
    # # print("orignal weights shape is",shape)
    # yiwei_weight = sess.run(tf.reshape(weight, [-1]))
    # # yiwei_weight = np.reshape(weight, [-1])
    # print("yiwei_weights is ", yiwei_weight)
    # # # # print("yiwei_weights is ", sess.run(yiwei_weights))
    # # yiwei_shape = yiwei_weight.shape
    # # # print("len yiwei_weights shape is ", len(yiwei_weight))
    # # # print("orignal yiwei_weights shape is ", yiwei_shape)
    # for i in range(int(len(yiwei_weight) / 100)):
    #     w.write(str(yiwei_weight[i]) + ' ,')
    # # under_threshold = abs() < th
    # th = get_th(weight)
    # print("th is ", th)
    # prune_weight, prune_counts = prune(weight, th)
    # prune_yiwei_weights = np.reshape(prune_weight, [-1])
    # for i in range(int(len(prune_yiwei_weights) / 100)):
    #     w2.write(str(prune_yiwei_weights[i]) + ' ,')
    # # print(prune_counts)
    # # sess.run(tf.assign(reader.get_tensor(node_name), prune_weight))
    # # change_weights(reader.get_tensor(node_name), prune_weight, node_name)
    # # for i in range(int(len(yiwei_weight) / 100)):
    # #     w2.write(str(weight[i]) + ' ,')
    # sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name(tensor_name), prune_weight))
    # print("final weight is ",weight)
