import tensorflow as tf
from tensorflow.python.framework import graph_util
import os
import time
import numpy as np
import tensorflow as tf
import cv2
from utils.nms_utils import gpu_nms
from utils.misc_utils import parse_anchors, read_class_names
import tensorflow.contrib.lite as lite
from utils.plot_utils import get_color_table, plot_one_box
from model import yolov3

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
def show_image(img_ori, boxes_,scores_,classes,num_class,labels_,width_ori,height_ori,new_size):
    # rescale the coordinates to the original image
    color_table = get_color_table(num_class)
    boxes_[:, 0] *= (width_ori / float(new_size[0]))
    boxes_[:, 2] *= (width_ori / float(new_size[0]))
    boxes_[:, 1] *= (height_ori / float(new_size[1]))
    boxes_[:, 3] *= (height_ori / float(new_size[1]))

    print("box coords:")
    print(boxes_)
    print('*' * 30)
    print("scores:")
    print(scores_)
    print('*' * 30)
    print("labels:")
    print(labels_)
    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        scores=scores_[i]
        plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[labels_[i]]+':'+str(scores)[:6], color=color_table[labels_[i]])
        # cv2.imshow('Detection result', img_ori)
    cv2.imwrite('detection_result_orignal_pb.jpg', img_ori)
        # cv2.waitKey(30)
    print("the image test procee")

def show_image_get_result(img_ori, boxes_,scores_,classes,num_class,labels_,width_ori,height_ori,new_size):
    # rescale the coordinates to the original image
    color_table = get_color_table(num_class)
    boxes_[:, 0] *= (width_ori / float(new_size[0]))
    boxes_[:, 2] *= (width_ori / float(new_size[0]))
    boxes_[:, 1] *= (height_ori / float(new_size[1]))
    boxes_[:, 3] *= (height_ori / float(new_size[1]))

    print("box coords:")
    print(boxes_)
    print('*' * 30)
    print("scores:")
    print(scores_)
    print('*' * 30)
    print("labels:")
    print(labels_)
    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        scores=scores_[i]
        plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[labels_[i]]+':'+str(scores)[:6], color=color_table[labels_[i]])
        # cv2.imshow('Detection result', img_ori)
    cv2.imwrite('detection_result_result_pb.jpg', img_ori)
        # cv2.waitKey(30)
    print("the image test procee")

def read_pb_return_tensors(pb_file, return_elements):
    print(pb_file)
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())
        return_elements = tf.import_graph_def(frozen_graph_def, return_elements=return_elements)
        print("return_elements is ", return_elements)
        input_tensor, output_tensors = return_elements[0], return_elements[1:]
    return input_tensor, output_tensors

def freeze(sess, output_file, output_node_names):
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        output_node_names,
    )
    with tf.gfile.GFile(output_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("=> {} ops written to {}.".format(len(output_graph_def.node), output_file))

def freeze_graph():
    # input_checkpoint = 'checkpoint/yolov3.ckpt-99'
    # cpu_pb_path='checkpoint/yolov3.pb'
    ckpt_path = "/home/pcl/tensorflow1.12/YOLOv3_TensorFlow/checkpoint/best_model_Epoch_109_step_8139.0_mAP_0.0883_loss_20.0605_lr_0.0004243221"
    pb_path= "./data/my_data/yolov3_01.pb"
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = ["yolov3/yolov3_head/feature_map_1",
    "yolov3/yolov3_head/feature_map_2",
    'yolov3/yolov3_head/feature_map_3']
    saver = tf.train.import_meta_graph(ckpt_path + '.meta', clear_devices=True)
    with tf.Session() as sess:
        saver.restore(sess, ckpt_path) # 恢复图并得到数据
        for op in sess.graph.get_operations():
            print(op.name, op.values())
        freeze(sess, pb_path, output_node_names)

def get_feature_map(input_image,pb_path):
# 定义一个计算图graph，获得feature_map
    fea = open('feature_map.txt', 'w')
    input_node_names = ["input_data:0"]
#     input_node_names = ["Iterator:0"]
    output_node_names = ["yolov3/yolov3_head/feature_map_1:0",
    "yolov3/yolov3_head/feature_map_2:0",
    'yolov3/yolov3_head/feature_map_3:0']
    graph=tf.Graph()
    with graph.as_default():
        input_tensor, output_tensors = read_pb_return_tensors(pb_path,input_node_names+output_node_names)
        with tf.Session(graph=graph) as sess:
            #feature_map= sess.run(output_tensors, feed_dict={input_tensor: [True, input_image]})
            feature_map = sess.run(output_tensors, feed_dict={input_tensor:input_image})
            for feature in feature_map:
                print("orignal featuremap shape is ", feature.shape)

                yiwei_orignal_featuremap = sess.run(tf.reshape(feature, [-1]))
                for i in yiwei_orignal_featuremap:
                    fea.write(str(i) + '\n')
    fea.close()
    print('orignal featuremap is ', feature_map)
    return feature_map

def get_yolov3_result(input_image,pb_path, img_ori, classes, num_classes, ):
# 定义一个计算图graph，获得feature_map
    height_ori, width_ori = img_ori.shape[:2]
    img_size = [416, 416]
    input_node_names = ["input_data:0"]
    output_node_names = ["yolov3/yolov3_head/detect_bbox:0", "yolov3/yolov3_head/detect_scores:0", "yolov3/yolov3_head/detect_labels:0"]
    graph=tf.Graph()
    with graph.as_default():
        input_tensor, output_tensors = read_pb_return_tensors(pb_path,input_node_names+output_node_names)
    with tf.Session(graph=graph) as sess:
        boxes, scores, labels  = sess.run(output_tensors, feed_dict={input_tensor:input_image})
    print(boxes)
    print(scores)
    print(labels)
    # show_image(img_ori, boxes, scores, classes, num_class, labels, width_ori, height_ori, img_size)
    return boxes, scores, labels

def pb_test_tf(img_ori,feature_map,classes,num_class,anchors,score_threshold,iou_threshold):
    '''
    :param img_ori:
    :param feature_map:
    :param classes:
    :param num_class:
    :param anchors:
    :param score_threshold:
    :param iou_threshold:
    :return:
    '''
    height_ori, width_ori = img_ori.shape[:2]
    img_size = [416, 416]
    # 定义一个计算图graph，获得输出结果：
    graph = tf.Graph()
    with graph.as_default():
        feature_map_1, feature_map_2, feature_map_3 = feature_map
        feature_map_1 = tf.constant(feature_map_1, dtype=tf.float32)
        feature_map_2 = tf.constant(feature_map_2, dtype=tf.float32)
        feature_map_3 = tf.constant(feature_map_3, dtype=tf.float32)
        tf_img_size = tf.constant(value=img_size, dtype=tf.int32)
        print("img_size:{}".format(img_size))

        # model = yolov3FeatureMap.tf_yolov3FeatureMap(num_classes=num_classes, anchors=anchors)
        yolo_model = yolov3(num_class, anchors)

    with tf.Session(graph=graph) as sess:
        pred_boxes, pred_confs, pred_probs = yolo_model.predict2(feature_map, tf_img_size)
        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=30, score_thresh=score_threshold,
                                        nms_thresh=iou_threshold)

        # saver = tf.train.Saver()
        # saver.restore(sess, ckpt_path)

        boxes_, scores_, labels_ = sess.run([boxes, scores, labels])
        print('orignal pb outputs result is ', boxes_, scores_, labels_)

        show_image(img_ori, boxes_, scores_, classes, num_class, labels_, width_ori, height_ori, img_size)

def freeze_graph_test():
    image_path = './4.jpg'
    anchor_path = "./data/yolo_anchors.txt"
    input_size = [416, 416]
    class_name_path = "./data/my_data/dianli_class.names"
    pb_path= "/home/pcl/tensorflow1.12/YOLOv3_TensorFlow/yolo_myself_darknet53.pb"
    score_threshold = 0.5
    iou_threshold = 0.5

    anchors = parse_anchors(anchor_path)
    classes = read_class_names(class_name_path)
    num_class = len(classes)

    # 读取图像数据
    img_ori = cv2.imread(image_path)
    img_resized = cv2.resize(img_ori, tuple(input_size))
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_resized = np.asarray(img_resized, np.float32)
    img_resized = img_resized[np.newaxis, :] / 255.

    feature_map = get_feature_map(img_resized, pb_path)
    pb_test_tf(img_ori, feature_map, classes, num_class, anchors, score_threshold, iou_threshold)


def freeze_graph_test_get_result_pb():
    image_path = './4.jpg'
    anchor_path = "./data/yolo_anchors.txt"
    input_size = [416, 416]
    class_name_path = "./data/my_data/dianli_class.names"
    pb_path= "/home/pcl/tensorflow1.12/YOLOv3_TensorFlow/yolo_darknet53_get_result.pb"
    score_threshold = 0.5
    iou_threshold = 0.5

    anchors = parse_anchors(anchor_path)
    classes = read_class_names(class_name_path)
    num_class = len(classes)

    # 读取图像数据
    img_ori = cv2.imread(image_path)
    img_resized = cv2.resize(img_ori, tuple(input_size))
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_resized = np.asarray(img_resized, np.float32)
    img_resized = img_resized[np.newaxis, :] / 255.
    height_ori, width_ori = img_ori.shape[:2]
    img_size = [416, 416]
    boxes, scores, labels  = get_yolov3_result(img_resized, pb_path, anchors, score_threshold, iou_threshold)
    print("get result pb is", boxes, scores, labels)
    show_image_get_result(img_ori, boxes, scores, classes, num_class, labels, width_ori, height_ori, img_size)
    # pb_test_tf_get_result(img_ori, boxes, scores, labels, classes, num_class, anchors, score_threshold, iou_threshold)
# # if __name__ == 'main':
# freeze_graph()
# print("freeze_graph done...")
'''
use the orignal pb model to inference img
'''
#########################################################################################
freeze_graph_test()
#########################################################################################
'''
use predict pb model to inference img
'''
##########################################################################################
freeze_graph_test_get_result_pb()
##########################################################################################
print("freeze_graph_test done...")
# convert_tflite()
# print("convert_tflite done...")