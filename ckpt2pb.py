import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow
import os
import numpy as np
from utils.misc_utils import parse_anchors, read_class_names
from model import yolov3
from utils.layer_utils import conv2d, darknet53_body, yolo_block, upsample_layer
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
# def freeze_graph(input_checkpoint, output_graph):
#     '''
#     :param input_checkpoint:
#     :param output_graph: PB模型保存路径
#     :return:
#     '''
#     # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
#     # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
#
#     # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
#     output_node_names = "yolov3/yolov3_head/feature_map_1,yolov3/yolov3_head/feature_map_2,yolov3/yolov3_head/feature_map_3"
#     saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
#     graph = tf.get_default_graph()  # 获得默认的图
#     input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
#
#     with tf.Session() as sess:
#         saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
#         output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
#             sess=sess,
#             input_graph_def=input_graph_def,  # 等于:sess.graph_def
#             output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开
#
#         with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
#             f.write(output_graph_def.SerializeToString())  # 序列化输出
#         print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点
#
#         # for op in graph.get_operations():
#         #     print(op.name, op.values())
# # 输入ckpt模型路径
# input_checkpoint='/home/pcl/tensorflow1.12/YOLOv3_TensorFlow/checkpoint/best_model.ckpt'
# # 输出pb模型的路径
# out_pb_path="./frozen_model.pb"
# # 调用freeze_graph将ckpt转为pb
# freeze_graph(input_checkpoint,out_pb_path)








# model_dir = '/home/pcl/tensorflow1.12/YOLOv3_TensorFlow/prun_checkpoint/'
# checkpoint_path = os.path.join(model_dir, "prun_include_res_model")
# f = open('node.txt', 'w')
# node_name = "yolov3/yolov3_head/Conv_7/weights"
# # checkpoint_path = os.path.join(model_dir, "resnet50_csv_18.pb")
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# # saver_to_restore = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
#     # saver_to_restore.restore(sess, checkpoint_path)
#     for key in var_to_shape_map:
#         f.write(str(key) + '\n')
#         # print(sess.run(key))
#         print("tensor_name: ", key)
#         print("tensor shape", reader.get_tensor(key).shape)
#         # if node_name in key:
#         # if 'moving_mean' in key :
#         #     print(reader.get_tensor(key))
#         # if 'bias' in key:
#             # print("bias_shape :", reader.get_tensor(key).shape)
#         f.write(str(reader.get_tensor(key)) + '\n' + str(reader.get_tensor(key).shape) + '\n')
#         # print("tensor_shape :", reader.get_tensor(key).shape)
#         # else:
#         #     print(reader.get_tensor(key))


'''
firet step: i will del 0.8 pencatage params  ,specially, i just make 0.2 pencatages params to 0, not del this params or del filters
'''
model_dir = '/home/pcl/tensorflow1.12/YOLOv3_TensorFlow/checkpoint/'
checkpoint_path = os.path.join(model_dir, "best_model_Epoch_2_step_2024.0_mAP_0.1784_loss_30.0785_lr_0.0001")
f = open('node.txt', 'w')
w = open("weights.txt", "w")
w2 = open("spase_we.txt", "w")
node_name = "yolov3/yolov3_head/Conv_7/weights"
tensor_name = "yolov3/yolov3_head/Conv_7/weights:0"
anchor_path = "./data/yolo_anchors.txt"
class_name_path  = "./data/my_data/dianli_class.names"
anchors = parse_anchors(anchor_path)
num_class = len(read_class_names(class_name_path))
tensor_name = "yolov3/yolov3_head/Conv_7/weights:0"
img_size = [412, 412]
# checkpoint_path = os.path.join(model_dir, "resnet50_csv_18.pb")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
graph = tf.get_default_graph()  # 获得默认的图
input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
# saver_to_restore = tf.train.Saver()
with tf.Session() as sess:
    # input_data = tf.placeholder(tf.float32, [1, img_size[1], img_size[0], 3], name='input_data')
    # yolo_model = yolov3(num_class, anchors)
    # sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    # with tf.variable_scope('yolov3'):
    #     pred_feature_maps = yolo_model.forward(input_data, False)
    # saver = tf.train.Saver()
    # saver.restore(sess, checkpoint_path)
    #
    # # sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    # # saver = tf.train.Saver()
    # # saver.restore(sess, checkpoint_path)
    # # saver_to_restore.restore(sess, checkpoint_path)
    tf_weights = []
    layer_weights = []
    pruning_layer = []
    # tensor_weight = sess.run(tf.get_default_graph().get_tensor_by_name(tensor_name))
    # _, _, _, nb_channels = tensor_weight.shape
    # print(nb_channels)
    for layer_name in var_to_shape_map:
        print(layer_name)
        if "darknet53_body" in layer_name and "weights" in layer_name:
            print(layer_name)
            pruning_layer.append(layer_name)
            # tf_weights.append(tf.get_default_graph().get_tensor_by_name(layer_name + ":0"))
            # layer_weights.append(sess.run(tf.get_default_graph().get_tensor_by_name(layer_name + ":0")))
            # print('channel')
    # weight = reader.get_tensor(node_name)
    # print("orignal weights is ",weight)
    # shape = weight.shape
    # print("orignal weights shape is",shape)
    # # yiwei_weights = tf.reshape(weights, [-1])
    # yiwei_weight = np.reshape(weight, [-1])
    # print("yiwei_weights is ", yiwei_weight)
    # # print("yiwei_weights is ", sess.run(yiwei_weights))
    # yiwei_shape = yiwei_weight.shape
    # print("len yiwei_weights shape is ", len(yiwei_weight))
    # print("orignal yiwei_weights shape is ", yiwei_shape)
#     for i in range(int(len(yiwei_weight) / 100)):
#         w.write(str(yiwei_weight[i]) + ' ,')
#     # under_threshold = abs(weights) < th
#     th = get_th(weight)
#     print("th is ", th)
#
#     prune_weight, prune_counts = prune(weight, th)
#     prune_yiwei_weights = np.reshape(prune_weight, [-1])
#     for i in range(int(len(prune_yiwei_weights) / 100)):
#         w2.write(str(prune_yiwei_weights[i]) + ' ,')
#     print(prune_counts)
#     change_weights(reader.get_tensor(node_name), prune_weight, node_name)
#     for i in range(int(len(yiwei_weight) / 100)):
#         w2.write(str(weight[i]) + ' ,')



# weights = reader.get_tensor(key)
#             xishuhua_list = get_th(weights)
#             yiwei_weights = tf.reshape(weights, [-1])
#             print(yiwei_weights)
#             yiwei_xishuhua_list = tf.reshape(xishuhua_list, [-1])
#             print(yiwei_xishuhua_list)
# import tensorflow as tf
# import os
#
# model_dir = '/home/pcl/tensorflow1.12/YOLOv3_TensorFlow/'
# model_name = 'yolo_darknet53_get_result.pb'
#
# def create_graph():
#     with tf.gfile.FastGFile(os.path.join(
#             model_dir, model_name), 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         tf.import_graph_def(graph_def, name='')
#
# create_graph()
# tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
# for tensor_name in tensor_name_list:
#     print(tensor_name,'\n')

# import tensorflow as tf
# import os
#
# model_dir = '/home/pcl/tensorflow1.12/YOLOv3_TensorFlow/'
# model_name = 'yolo_myself.pb'
#
# def create_graph():
#     with tf.gfile.FastGFile(os.path.join(
#             model_dir, model_name), 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         tf.import_graph_def(graph_def, name='')
#
# create_graph()
# tf_tensor = []
# tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
# for tensor_name in tensor_name_list:
#     print(tensor_name,'\n')
#     # print(tf.get_default_graph().get_tensor_by_name(tensor_name+ ":0"))
#     tf_tensor.append(tf.get_default_graph().get_tensor_by_name(tensor_name+ ":0"))
# print(tf_tensor)
    # def get_th(weight, pencentage=0.8):
    #     flat = np.reshape(weight, [-1])
    #     flat_list = sorted(map(abs, flat))
    #     return flat_list[int(len(flat_list) * pencentage)]
    #
    #
    # def prune(weights, th):
    #     '''
    #     :param weights: weight Variable
    #     :param th: float value, weight under th will be pruned
    #     :return: sparse_weight
    #     '''
    #     shape = weights.shape
    #     # weight_arr = sess.run(weights)
    #     under_threshold = abs(weights) < th
    #     weights[under_threshold] = 0
    #     # tmp = weights
    #     # for i in range(len(shape) - 1):
    #     #     tmp = tmp[-1]
    #     # if tmp[-1] == 0:
    #     #     tmp[-1] == 0.01
    #     count = np.sum(under_threshold)
    #     print(count)
    #     return weights, ~under_threshold
    #
    # def change_weights(orignal_weight, prune_weight, node_name):
    #     '''
    #     because this weight only one layer ,so use weight not weights
    #     '''
    #     sess.run(tf.assign(orignal_weight, prune_weight))
    #     return orignal_weight
