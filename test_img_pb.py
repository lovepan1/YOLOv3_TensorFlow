# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import os
from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
import datetime
from model import yolov3
# from pruning_model import parse_darknet53_body
from pruning_model import sparse_yolov3

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--input_image_dir", type=str,default = "./img_dir",
                    help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--class_name_path", type=str, default="./data/my_data/dianli_class.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default='/home/pcl/tensorflow1.12/YOLOv3_TensorFlow/checkpoint/kmeans_prune_restore_model_all.ckpt',
                    help="The path of the weights to restore.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

with tf.Session() as sess:
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    '''
    frozen feature map pb model for yolov3, but pb model exclude post process, nms process, because of huawei a200 required post process must added in pb model,so i create another pb model
    '''
    ########################################################################################
    # yolo_model = yolov3(args.num_class, args.anchors)
    # with tf.variable_scope('yolov3'):
    #     pred_feature_maps = yolo_model.forward(input_data, False)
    # pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    #
    # pred_scores = pred_confs * pred_probs
    #
    # boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=30, score_thresh=0.4, nms_thresh=0.5)
    #
    # saver = tf.train.Saver()
    # saver.restore(sess, args.restore_path)
    # constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["yolov3/yolov3_head/feature_map_1", "yolov3/yolov3_head/feature_map_2", "yolov3/yolov3_head/feature_map_3"])
    # with tf.gfile.FastGFile("./yolo_myself_darknet53.pb", mode='wb') as f:
    #     f.write(constant_graph.SerializeToString())
    #########################################################################################
    '''
    darknet53 ckpt file -> darknet53 pb file 
    '''
    # yolo_model = yolov3(args.num_class, args.anchors)
    # with tf.variable_scope('yolov3'):
    #     boxes, scores, labels = yolo_model.forward_get_result(input_data, False)
    # saver = tf.train.Saver()
    # saver.restore(sess, args.restore_path)
    # constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["yolov3/yolov3_head/detect_bbox", "yolov3/yolov3_head/detect_scores", "yolov3/yolov3_head/detect_labels"])
    # output_tensors = ["yolov3/yolov3_head/detect_bbox:0", "yolov3/yolov3_head/detect_scores:0", "yolov3/yolov3_head/detect_labels:0"]
    # with tf.gfile.FastGFile("./yolo_darknet53_get_result.pb", mode='wb') as f:
    #     f.write(constant_graph.SerializeToString())
    ##########################################################################################
    '''
    prun model exlcude res ckpt file -> prun model exlcude res pb file 
    '''
    # yolo_model = sparse_yolov3(args.num_class, args.anchors)
    # with tf.variable_scope('yolov3'):
    #     pred_feature_maps = yolo_model.forward(input_data, False)
    #     pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    #     pred_scores = pred_confs * pred_probs
    #     boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=30, score_thresh=0.4, nms_thresh=0.5)
    # saver = tf.train.Saver()
    # saver.restore(sess, args.restore_path)
    # constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["yolov3/yolov3_head/feature_map_1", "yolov3/yolov3_head/feature_map_2", "yolov3/yolov3_head/feature_map_3"])
    # with tf.gfile.FastGFile("./yolo_myself_prun.pb", mode='wb') as f:
    #     f.write(constant_graph.SerializeToString())
    ############################################################################################
    '''
    prun model inlcude res ckpt file -> prun model inlcude res pb file 
    '''
    # yolo_model = sparse_yolov3(args.num_class, args.anchors)
    # with tf.variable_scope('yolov3'):
    #     pred_feature_maps = yolo_model.forward_include_res(input_data, False)
    #     pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    #     pred_scores = pred_confs * pred_probs
    #     boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=30, score_thresh=0.4, nms_thresh=0.5)
    # saver = tf.train.Saver()
    # saver.restore(sess, args.restore_path)
    # constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["yolov3/yolov3_head/feature_map_1", "yolov3/yolov3_head/feature_map_2", "yolov3/yolov3_head/feature_map_3"])
    # with tf.gfile.FastGFile("./yolo_myself_prun_include_res.pb", mode='wb') as f:
    #     f.write(constant_graph.SerializeToString())
    #########################################################################################
    '''
    prune channels and filters yolov3 ckpt file ->  prun model inlcude res pb file 
    '''
    # prune_check_point_path = os.path.join(prune_model._checkpoint_dir, 'prue_channel_model.ckpt')
    # input_data = tf.placeholder(tf.float32, [1, prune_model._img_size[1], prune_model._img_size[0], 3],
    #                             name='input_data')
    yolo_prun_model = sparse_yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps_fintune = yolo_prun_model.forward_include_res_with_prune_factor(input_data,
                                                                                          prune_factor=0.8,
                                                                                          is_training=True)
        pred_boxes, pred_confs, pred_probs = yolo_prun_model.predict(pred_feature_maps_fintune)
        pred_scores = pred_confs * pred_probs
        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=30, score_thresh=0.4, nms_thresh=0.5)
    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)
    constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["yolov3/yolov3_head/feature_map_1", "yolov3/yolov3_head/feature_map_2", "yolov3/yolov3_head/feature_map_3"])
    with tf.gfile.FastGFile("./prune_filters_channels_yolo_get_vector.pb", mode='wb') as f:
        f.write(constant_graph.SerializeToString())
    img_list = os.listdir(args.input_image_dir)
    print(args.input_image_dir)
    for i in img_list:
        print(i)
        img_dir = os.path.join(args.input_image_dir, i)
        # img = cv2.imread(img_dir)
        # img_ori_list
        img_ori = cv2.imread(img_dir)
        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(args.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.
        starttime = datetime.datetime.now()
        feature_maps = sess.run([pred_feature_maps_fintune], feed_dict={input_data: img})
        # boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
        # print("boxes_, scores_, labels is ", boxes_, scores_, labels_)
        # boxes_extract_pb, scores_extract_pb, lables_extract_pb = sess.run(output_tensors, feed_dict={input_data: img})
        # print("boxes_extract_pb, scores_extract_pb, lables_extract_pb", boxes_extract_pb, scores_extract_pb, lables_extract_pb)
        endtime = datetime.datetime.now()
        print("sess cost time is ", endtime - starttime)
        # rescale the coordinates to the original image
        # boxes_[:, 0] *= (width_ori/float(args.new_size[0]))
        # boxes_[:, 2] *= (width_ori/float(args.new_size[0]))
        # boxes_[:, 1] *= (height_ori/float(args.new_size[1]))
        # boxes_[:, 3] *= (height_ori/float(args.new_size[1]))
        #
        # print("box coords:")
        # print(boxes_)
        # print('*' * 30)
        # print("scores:")
        # print(scores_)
        # print('*' * 30)
        # print("labels:")
        # print(labels_)
        # for i in range(len(boxes_)):
        #     x0, y0, x1, y1 = boxes_[i]
        #     plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]], color=color_table[labels_[i]])
    # cv2.imshow('Detection result', img_ori)
    cv2.imwrite('detection_result.jpg', img_ori)
    cv2.waitKey(0)
