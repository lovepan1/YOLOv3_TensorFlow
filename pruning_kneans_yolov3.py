import abc
import re
import tempfile
import traceback
import os
from typing import Tuple, Callable, Union, List, Optional
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import models, layers
from sklearn import cluster, metrics
from utils.misc_utils import parse_anchors, read_class_names
from model import yolov3
from pruning_model import sparse_yolov3

# from ridurre import base_filter_pruning
anchor_path = "./data/yolo_anchors.txt"
class_name_path  = "./data/my_data/dianli_class.names"
anchors = parse_anchors(anchor_path)
num_class = len(read_class_names(class_name_path))

class BasePruning:
    _FUZZ_EPSILON = 1e-5

    def __init__(self,
                 pruning_factor: float,
                 # model_orignal,
                 # model_finetune,
                 # model_finetune_fn: Optional[Callable[[models.Model, int, int], None]],
                 nb_finetune_epochs: int,
                 nb_trained_for_epochs: int,
                 maximum_prune_iterations: int,
                 maximum_pruning_percent: float,
                 checkpoint_dir: str,
                 ):

        self._pruning_factor = pruning_factor
        self._tmp_model_file_name = tempfile.NamedTemporaryFile().name

        # self._model_compile_fn = model_compile_fn
        # self._model_finetune_fn = model_finetune_fn

        self._nb_finetune_epochs = nb_finetune_epochs
        self._current_nb_of_epochs = nb_trained_for_epochs
        self._maximum_prune_iterations = maximum_prune_iterations
        self._maximum_pruning_percent = maximum_pruning_percent

        self._channel_number_bins = None
        self._pruning_factors_for_channel_bins = None

        self._original_number_of_filters = -1

        self._checkpoint_dir = checkpoint_dir
        # self._model_orignal = model_orignal
        # self._model_finetune = model_finetune

        # TODO: select a subset of layers to prune
        self._prunable_layers_regex = ".*"
        self._restore_part_first = ['yolov3/darknet53_body']
        self._restore_part_second = ['yolov3/darknet53_body','yolov3/yolov3_head']
        self._update_part = ['yolov3/yolov3_head']
        self._img_size = [412, 412]

    def run_pruning(self, prune_factor = 0.8,
                    custom_objects_inside_model: dict = None) -> Tuple[models.Model, int]:
        # self._original_number_of_filters = self._count_number_of_filters(model)

        pruning_iteration = 0
        # with tf.Session() as sess:
        while True:
            if prune_factor is not None:
                self._pruning_factor = prune_factor

                # Pruning step
                print("Running filter pruning {0}".format(pruning_iteration))
                model, pruning_dict = self._prune_first_stage()


                # Computing statistics
                # nb_of_pruned_filters = sum(pruning_dict.values())
                # if nb_of_pruned_filters == 0:
                #     print("Number of pruned filters == 0, so pruning is stopped")
                #     break
                # print("Number of pruned filters at this step: {0}".format(nb_of_pruned_filters))
                # pruning_percent = self._compute_pruning_percent(model)
                # print("Network is pruned from the original state, by {0} %".format(pruning_percent * 100))
                #
                # Finetune step
                # self._save_after_pruning(model)
                # self._clean_up_after_pruning(model)
                # model = self._load_back_saved_model(custom_objects_inside_model)
                # self._model_compile_fn(model)
                # if self._model_finetune_fn is not None:
                #     self._model_finetune_fn(model, self._current_nb_of_epochs,
                #                             self._current_nb_of_epochs + self._nb_finetune_epochs)
                # self._current_nb_of_epochs += self._nb_finetune_epochs

                # Stopping conditions
                # if nb_of_pruned_filters < 1:
                #     print("No filters were pruned. Pruning is stopped.")
                #     break
                # if self._maximum_pruning_percent is not None:
                #     if pruning_percent > self._maximum_pruning_percent:
                #         print(
                #             "Network pruning (currently {0} %) reached the maximum based on your definition ({1} %)".format(
                #                 pruning_percent * 100, self._maximum_pruning_percent * 100))
                #         break
                pruning_iteration += 1
                #
                if self._maximum_prune_iterations is not None:
                    if pruning_iteration > self._maximum_prune_iterations:
                        break

            print("Pruning stopped.")
            # return model, self._current_nb_of_epochs
            return model, pruning_dict

    def _prune_first_stage(self):
        # surgeon = kerassurgeon.Surgeon(model, copy=True)
        pruning_dict = dict()
        pruning_layer = []
        tf_weights = []
        layer_weights = []
        checkpoint_path = os.path.join(self._checkpoint_dir, "best_model_Epoch_2_step_2024.0_mAP_0.1784_loss_30.0785_lr_0.0001")
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        graph = tf.get_default_graph()  # 获得默认的图
        input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
        # with tf.Session(graph = graph) as sess:
        #     input_data = tf.placeholder(tf.float32, [1, self._img_size[1], self._img_size[0], 3], name='input_data')
        #     yolo_model = yolov3(num_class, anchors)
        #     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        #     with tf.variable_scope('yolov3'):
        #         pred_feature_maps = yolo_model.forward(input_data, is_training=True)
        #     saver = tf.train.Saver()
        #     saver.restore(sess, checkpoint_path)
        #     itera = 0
        #     for layer_name in var_to_shape_map:
        #         if "darknet53_body" in layer_name and "weights" in layer_name:
        #             print('current layer is ', layer_name)
        #             pruning_layer.append(layer_name)
        #             tf_layer_weight = tf.get_default_graph().get_tensor_by_name(layer_name + ":0")
        #             layer_weight = sess.run(tf.get_default_graph().get_tensor_by_name(layer_name + ":0"))
        #             tf_weights.append(tf.get_default_graph().get_tensor_by_name(layer_name + ":0"))
        #             layer_weights.append(layer_weight)
        #             pruning_factor = self._pruning_factor
        #             filter_indices_to_prune, filter_indices_to_prune_input = self.run_pruning_for_conv2d_layer(pruning_factor, layer_weight)
        #
        #             print('filter_indices_to_prune is ', filter_indices_to_prune)
        #             print('filter_indices_to_prune_input is ', filter_indices_to_prune_input)
        #             ######prune output weight######
        #             W, H, N, nb_channels = layer_weight.shape
        #             print("layer_weight.shape is ", layer_weight.shape)
        #             layer_weight_reshaped = sess.run(tf.reshape(layer_weight.transpose(3, 0, 1, 2), (nb_channels, -1)))
        #             # layer_weight_reshaped = layer_weight.reshape(nb_channels, -1)
        #             prun_weight_reshape = np.delete(layer_weight_reshaped, filter_indices_to_prune, axis=0)
        #             prun_channel, _ = prun_weight_reshape.shape
        #             print('prun_channel is ', prun_channel)
        #             print('calc prune channel is ', nb_channels - len(filter_indices_to_prune))
        #             prune_weight = prun_weight_reshape.reshape(W, H, N, nb_channels - len(filter_indices_to_prune))
        #             print("prun weight shape is", prune_weight.shape)
        #             sess.run(tf.assign(tf_layer_weight, prune_weight, validate_shape=False))
        #             pruning_dict[layer_name] = len(filter_indices_to_prune)
        #
        #             #######prune BN params########
        #             bn_params = ['BatchNorm/gamma', 'BatchNorm/beta', 'BatchNorm/moving_variance',
        #                          'BatchNorm/moving_mean']
        #             bn_layer_name = []
        #             for i in bn_params:
        #                 bn_params_str = layer_name.replace('weights', i)
        #                 bn_layer_name.append(bn_params_str)
        #             for bn_layer in bn_layer_name:
        #                 tf_bn_param = tf.get_default_graph().get_tensor_by_name(bn_layer + ":0")
        #                 layer_bn_param = sess.run(tf_bn_param)
        #                 bn_channel = layer_bn_param.shape
        #                 bn_filter_prune = filter_indices_to_prune
        #                 prune_bn_param = np.delete(layer_bn_param, bn_filter_prune, axis=0)
        #                 sess.run(tf.assign(tf_bn_param, prune_bn_param, validate_shape=False))
        #                 print('current layer is ', bn_layer)
        #                 print("bn param.shape is ", layer_bn_param.shape)
        #                 print("layer_bn_param.shape is ", bn_channel)
        #                 print('prune bn param shape is ', prune_bn_param.shape)
        #             ###############################
        #             try:
        #                 next_layer_number = int(layer_name.split('/')[-2].split('_')[-1]) + 1
        #             except:
        #                 next_layer_number = 1
        #             next_layer_name = 'yolov3/darknet53_body/Conv_' + str(next_layer_number) + '/weights'
        #             if next_layer_number != 52:
        #                 print("the next layer is ", next_layer_name)
        #                 ######prune input filter weight######
        #                 # if 'yolov3/darknet53_body/Conv/weights' not in layer_name:  ### cannot prune the first conv
        #                 next_layer_weight = sess.run(tf.get_default_graph().get_tensor_by_name(next_layer_name + ":0"))
        #                 tf_next_layer = tf.get_default_graph().get_tensor_by_name(next_layer_name + ":0")
        #                 W, H, input_channels, nb_channels_2 = next_layer_weight.shape
        #                 print('next layer shape is ', next_layer_weight.shape)
        #                 next_layer_weight_reshaped = sess.run(tf.reshape(next_layer_weight.transpose(2, 0, 1, 3), (input_channels, -1)))
        #                 # layer_weight_reshaped = layer_weight.reshape(nb_channels, -1)
        #                 prun_weight_reshape_input = np.delete(next_layer_weight_reshaped, filter_indices_to_prune, axis=0)
        #                 prun_channel_input, _ = prun_weight_reshape_input.shape
        #                 print('prun_channel input is ', prun_channel_input)
        #                 print('calc prune channel input is ', input_channels - len(filter_indices_to_prune))
        #                 prune_weight_input = prun_weight_reshape_input.reshape(W, H, input_channels - len(filter_indices_to_prune), nb_channels_2)
        #                 print("prune_weight_input shape is", prune_weight_input.shape)
        #                 sess.run(tf.assign(tf_next_layer, prune_weight_input, validate_shape=False))
        #                 pruning_dict[layer_name] = len(filter_indices_to_prune) + len(filter_indices_to_prune)
        #
        #     saver.save(sess, os.path.join(self._checkpoint_dir, 'prue_channel_model.ckpt'))
        yolo_prune_model = self._reconstruction_model()
        return yolo_prune_model, pruning_dict

    def _prune_second_stage(self):
        # surgeon = kerassurgeon.Surgeon(model, copy=True)
        pruning_dict = dict()
        pruning_layer = []
        tf_weights = []
        layer_weights = []
        checkpoint_path = os.path.join(self._checkpoint_dir, "best_model_Epoch_2_step_2024.0_mAP_0.1784_loss_30.0785_lr_0.0001")
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        graph = tf.get_default_graph()  # 获得默认的图
        input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
        with tf.Session(graph = graph) as sess:
            input_data = tf.placeholder(tf.float32, [1, self._img_size[1], self._img_size[0], 3], name='input_data')
            yolo_model = yolov3(num_class, anchors)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            with tf.variable_scope('yolov3'):
                pred_feature_maps = yolo_model.forward(input_data, is_training=True)
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            itera = 0
            for layer_name in var_to_shape_map:
                if "darknet53_body" in layer_name and "weights" in layer_name:
                    print('current layer is ', layer_name)
                    pruning_layer.append(layer_name)
                    tf_layer_weight = tf.get_default_graph().get_tensor_by_name(layer_name + ":0")
                    layer_weight = sess.run(tf.get_default_graph().get_tensor_by_name(layer_name + ":0"))
                    tf_weights.append(tf.get_default_graph().get_tensor_by_name(layer_name + ":0"))
                    layer_weights.append(layer_weight)
                    pruning_factor = self._pruning_factor
                    filter_indices_to_prune, filter_indices_to_prune_input = self.run_pruning_for_conv2d_layer(pruning_factor, layer_weight)

                    # if 'yolov3/darknet53_body/Conv_25' in layer_name or 'yolov3/darknet53_body/Conv_42' in layer_name:
                    #     continue
                    # if 'yolov3/darknet53_body/Conv_24' not in layer_name and 'yolov3/darknet53_body/Conv_41' not in layer_name: ####conv24\conv41 just prune filter, cannot prune channel
                    print('filter_indices_to_prune is ', filter_indices_to_prune)
                    print('filter_indices_to_prune_input is ', filter_indices_to_prune_input)
                    ######prune output weight######
                    W, H, N, nb_channels = layer_weight.shape
                    print("layer_weight.shape is ", layer_weight.shape)
                    layer_weight_reshaped = sess.run(tf.reshape(layer_weight.transpose(3, 0, 1, 2), (nb_channels, -1)))
                    # layer_weight_reshaped = layer_weight.reshape(nb_channels, -1)
                    prun_weight_reshape = np.delete(layer_weight_reshaped, filter_indices_to_prune, axis=0)
                    prun_channel, _ = prun_weight_reshape.shape
                    print('prun_channel is ', prun_channel)
                    print('calc prune channel is ', nb_channels - len(filter_indices_to_prune))
                    prune_weight = prun_weight_reshape.reshape(W, H, N, nb_channels - len(filter_indices_to_prune))
                    print("prun weight shape is", prune_weight.shape)
                    # sess.run(tf.assign(tf_layer_weight, prune_weight, validate_shape=False))
                    pruning_dict[layer_name] = len(filter_indices_to_prune)

                    #######prune BN params########
                    bn_params = ['BatchNorm/gamma', 'BatchNorm/beta', 'BatchNorm/moving_variance',
                                 'BatchNorm/moving_mean']
                    bn_layer_name = []
                    for i in bn_params:
                        bn_params_str = layer_name.replace('weights', i)
                        bn_layer_name.append(bn_params_str)
                    for bn_layer in bn_layer_name:
                        tf_bn_param = tf.get_default_graph().get_tensor_by_name(bn_layer + ":0")
                        layer_bn_param = sess.run(tf_bn_param)
                        bn_channel = layer_bn_param.shape
                        bn_filter_prune = filter_indices_to_prune
                        prune_bn_param = np.delete(layer_bn_param, bn_filter_prune, axis=0)
                        # sess.run(tf.assign(tf_bn_param, prune_bn_param, validate_shape=False))
                        print('current layer is ', bn_layer)
                        print("bn param.shape is ", layer_bn_param.shape)
                        print("layer_bn_param.shape is ", bn_channel)
                        print('prune bn param shape is ', prune_bn_param.shape)
                    ###############################

                    ######prune input filter weight######
                    if 'yolov3/darknet53_body/Conv/weights' not in layer_name:  ### cannot prune the first conv
                        layer_weight = sess.run(tf.get_default_graph().get_tensor_by_name(layer_name + ":0"))
                        W, H, input_channels, nb_channels_2 = layer_weight.shape
                        print('first prune output shape is ', layer_weight.shape)
                        layer_weight_reshaped_input = sess.run(tf.reshape(layer_weight.transpose(2, 0, 1, 3), (input_channels, -1)))
                        # layer_weight_reshaped = layer_weight.reshape(nb_channels, -1)
                        prun_weight_reshape_input = np.delete(layer_weight_reshaped_input, filter_indices_to_prune_input, axis=0)
                        prun_channel_input, _ = prun_weight_reshape_input.shape
                        print('prun_channel input is ', prun_channel_input)
                        print('calc prune channel input is ', input_channels - len(filter_indices_to_prune_input))
                        prune_weight_input = prun_weight_reshape_input.reshape(W, H, input_channels - len(filter_indices_to_prune_input), nb_channels_2)
                        print("prune_weight_input shape is", prune_weight_input.shape)
                        sess.run(tf.assign(tf_layer_weight, prune_weight_input, validate_shape=False))
                        pruning_dict[layer_name] = len(filter_indices_to_prune) + len(filter_indices_to_prune_input)

            saver.save(sess, os.path.join(self._checkpoint_dir, 'prue_channel_model.ckpt'))
            # yolo_prune_model = self._reconstruction_model()
        return yolo_model, pruning_dict
    # def _prune(self):
    #     # surgeon = kerassurgeon.Surgeon(model, copy=True)
    #     pruning_dict = dict()
    #     pruning_layer = []
    #     tf_weights = []
    #     layer_weights = []
    #     img_size = [412, 412]
    #
    #     checkpoint_path = os.path.join(self._checkpoint_dir, "best_model_Epoch_2_step_2024.0_mAP_0.1784_loss_30.0785_lr_0.0001")
    #     reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    #     var_to_shape_map = reader.get_variable_to_shape_map()
    #     graph = tf.get_default_graph()  # 获得默认的图
    #     input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
    #     with tf.Session(graph = graph) as sess:
    #         input_data = tf.placeholder(tf.float32, [1, self._img_size[1], self._img_size[0], 3], name='input_data')
    #         yolo_model = yolov3(num_class, anchors)
    #         sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    #         with tf.variable_scope('yolov3'):
    #             pred_feature_maps = yolo_model.forward(input_data, is_training=True)
    #         saver = tf.train.Saver()
    #         saver.restore(sess, checkpoint_path)
    #         itera = 0
    #         for layer_name in var_to_shape_map:
    #             if "darknet53_body" in layer_name and "weights" in layer_name:
    #                 print('current layer is ', layer_name)
    #                 pruning_layer.append(layer_name)
    #                 tf_layer_weight = tf.get_default_graph().get_tensor_by_name(layer_name + ":0")
    #                 layer_weight = sess.run(tf.get_default_graph().get_tensor_by_name(layer_name + ":0"))
    #                 tf_weights.append(tf.get_default_graph().get_tensor_by_name(layer_name + ":0"))
    #                 layer_weights.append(layer_weight)
    #                 pruning_factor = self._pruning_factor
    #                 filter_indices_to_prune, filter_indices_to_prune_input = self.run_pruning_for_conv2d_layer(pruning_factor, layer_weight)
    #
    #                 # if 'yolov3/darknet53_body/Conv_25' in layer_name or 'yolov3/darknet53_body/Conv_42' in layer_name:
    #                 #     continue
    #                 # if 'yolov3/darknet53_body/Conv_24' not in layer_name and 'yolov3/darknet53_body/Conv_41' not in layer_name: ####conv24\conv41 just prune filter, cannot prune channel
    #                 print('filter_indices_to_prune is ', filter_indices_to_prune)
    #                 print('filter_indices_to_prune_input is ', filter_indices_to_prune_input)
    #                 ######prune output weight######
    #                 W, H, N, nb_channels = layer_weight.shape
    #                 print("layer_weight.shape is ", layer_weight.shape)
    #                 layer_weight_reshaped = sess.run(tf.reshape(layer_weight.transpose(3, 0, 1, 2), (nb_channels, -1)))
    #                 # layer_weight_reshaped = layer_weight.reshape(nb_channels, -1)
    #                 prun_weight_reshape = np.delete(layer_weight_reshaped, filter_indices_to_prune, axis=0)
    #                 prun_channel, _ = prun_weight_reshape.shape
    #                 print('prun_channel is ', prun_channel)
    #                 print('calc prune channel is ', nb_channels - len(filter_indices_to_prune))
    #                 prune_weight = prun_weight_reshape.reshape(W, H, N, nb_channels - len(filter_indices_to_prune))
    #                 print("prun weight shape is", prune_weight.shape)
    #                 sess.run(tf.assign(tf_layer_weight, prune_weight, validate_shape=False))
    #                 pruning_dict[layer_name] = len(filter_indices_to_prune)
    #
    #                 #######prune BN params########
    #                 bn_params = ['BatchNorm/gamma', 'BatchNorm/beta', 'BatchNorm/moving_variance',
    #                              'BatchNorm/moving_mean']
    #                 bn_layer_name = []
    #                 for i in bn_params:
    #                     bn_params_str = layer_name.replace('weights', i)
    #                     bn_layer_name.append(bn_params_str)
    #                 for bn_layer in bn_layer_name:
    #                     tf_bn_param = tf.get_default_graph().get_tensor_by_name(bn_layer + ":0")
    #                     layer_bn_param = sess.run(tf_bn_param)
    #                     bn_channel = layer_bn_param.shape
    #                     bn_filter_prune = filter_indices_to_prune
    #                     prune_bn_param = np.delete(layer_bn_param, bn_filter_prune, axis=0)
    #                     sess.run(tf.assign(tf_bn_param, prune_bn_param, validate_shape=False))
    #                     print('current layer is ', bn_layer)
    #                     print("bn param.shape is ", layer_bn_param.shape)
    #                     print("layer_bn_param.shape is ", bn_channel)
    #                     print('prune bn param shape is ', prune_bn_param.shape)
    #                 ###############################
    #
    #                 ######prune input filter weight######
    #                 if 'yolov3/darknet53_body/Conv/weights' not in layer_name:  ### cannot prune the first conv
    #                     layer_weight = sess.run(tf.get_default_graph().get_tensor_by_name(layer_name + ":0"))
    #                     W, H, input_channels, nb_channels_2 = layer_weight.shape
    #                     print('first prune output shape is ', layer_weight.shape)
    #                     layer_weight_reshaped_input = sess.run(tf.reshape(layer_weight.transpose(2, 0, 1, 3), (input_channels, -1)))
    #                     # layer_weight_reshaped = layer_weight.reshape(nb_channels, -1)
    #                     prun_weight_reshape_input = np.delete(layer_weight_reshaped_input, filter_indices_to_prune_input, axis=0)
    #                     prun_channel_input, _ = prun_weight_reshape_input.shape
    #                     print('prun_channel input is ', prun_channel_input)
    #                     print('calc prune channel input is ', input_channels - len(filter_indices_to_prune_input))
    #                     prune_weight_input = prun_weight_reshape_input.reshape(W, H, input_channels - len(filter_indices_to_prune_input), nb_channels_2)
    #                     print("prune_weight_input shape is", prune_weight_input.shape)
    #                     sess.run(tf.assign(tf_layer_weight, prune_weight_input, validate_shape=False))
    #                     pruning_dict[layer_name] = len(filter_indices_to_prune) + len(filter_indices_to_prune_input)
    #
    #         saver.save(sess, os.path.join(self._checkpoint_dir, 'prue_channel_model.ckpt'))
    #         # yolo_prune_model = self._reconstruction_model()
    #     return yolo_model, pruning_dict

    def _reconstruction_model(self):
        prun_graph = tf.Graph()
        with tf.Session(graph=prun_graph) as sess:
            prune_check_point_path = os.path.join(self._checkpoint_dir, 'prue_channel_model.ckpt')
            input_data_2 = tf.placeholder(tf.float32, [1, self._img_size[1], self._img_size[0], 3], name='input_data')
            yolo_prun_model = sparse_yolov3(num_class, anchors)
            with tf.variable_scope('yolov3'):
                pred_feature_maps = yolo_prun_model.forward_include_res_with_prune_factor(input_data_2,
                                                                                          prune_factor=self._pruning_factor,
                                                                                          is_training=True)
            print("prune network completed")
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            print("completed initialized")
            saver_to_restore = tf.train.Saver(
                var_list=tf.contrib.framework.get_variables_to_restore(include=self._restore_part_first))
            update_vars = tf.contrib.framework.get_variables_to_restore(include=self._update_part)
            saver_to_restore.restore(sess, prune_check_point_path)
            saver_to_restore.save(sess, os.path.join(self._checkpoint_dir, 'kmeans_prune_restore_model.ckpt'))
            saver_best = tf.train.Saver()
            saver_best.save(sess, os.path.join(self._checkpoint_dir, 'kmeans_prune_restore_model_all.ckpt'))
            return yolo_prun_model

    def run_pruning_for_conv2d_layer(self, pruning_factor: float, layer_weight_mtx):
        _, _, input_channels, nb_channels = layer_weight_mtx.shape

        # Initialize KMeans
        ########################pruning outputs#########################################
        nb_of_clusters, _ = self._calculate_number_of_channels_to_keep(pruning_factor, nb_channels)
        kmeans = cluster.KMeans(nb_of_clusters, "k-means++")

        # Fit with the flattened weight matrix
        # (height, width, input_channels, output_channels) -> (output_channels, flattened features)
        layer_weight_mtx_reshaped = layer_weight_mtx.transpose(3, 0, 1, 2).reshape(nb_channels, -1)
        # Apply some fuzz to the weights, to avoid duplicates
        layer_weight_mtx_reshaped = self._apply_fuzz(layer_weight_mtx_reshaped)
        kmeans.fit(layer_weight_mtx_reshaped)

        # If a cluster has only a single member, then that should not be pruned
        # so that point will always be the closest to the cluster center
        closest_point_to_cluster_center_indices = metrics.pairwise_distances_argmin(kmeans.cluster_centers_,
                                                                                    layer_weight_mtx_reshaped)
        #print('closest_point_to_cluster_center_indices is ', closest_point_to_cluster_center_indices)
        # Compute filter indices which can be pruned
        channel_indices = set(np.arange(len(layer_weight_mtx_reshaped)))
        channel_indices_to_keep = set(closest_point_to_cluster_center_indices)
        channel_indices_to_prune = list(channel_indices.difference(channel_indices_to_keep))
        channel_indices_to_keep = list(channel_indices_to_keep)

        if len(channel_indices_to_keep) > nb_of_clusters:
            print("Number of selected channels for pruning is less than expected")
            diff = len(channel_indices_to_keep) - nb_of_clusters
            print("Randomly adding {0} channels for pruning".format(diff))
            np.random.shuffle(channel_indices_to_keep)
            for i in range(diff):
                channel_indices_to_prune.append(channel_indices_to_keep.pop(i))
        elif len(channel_indices_to_keep) < nb_of_clusters:
            print("Number of selected channels for pruning is greater than expected. Leaving too few channels.")
            diff = nb_of_clusters - len(channel_indices_to_keep)
            print("Discarding {0} pruneable channels".format(diff))
            for i in range(diff):
                channel_indices_to_keep.append(channel_indices_to_prune.pop(i))

        if len(channel_indices_to_keep) != nb_of_clusters:
            raise ValueError(
                "Number of clusters {0} is not equal with the selected "
                "pruneable channels {1}".format(nb_of_clusters, len(channel_indices_to_prune)))
        #####################################################################################

        ############################pruning inputs channels##################################
        input_of_clusters, _ = self._calculate_number_of_channels_to_keep(pruning_factor, input_channels)
        kmeans_input = cluster.KMeans(input_of_clusters, "k-means++")

        # Fit with the flattened weight matrix
        # (height, width, input_channels, output_channels) -> (output_channels, flattened features)
        layer_weight_mtx_reshaped_input = layer_weight_mtx.transpose(2, 0, 1, 3).reshape(input_channels, -1)
        # Apply some fuzz to the weights, to avoid duplicates
        layer_weight_mtx_reshaped_input = self._apply_fuzz(layer_weight_mtx_reshaped_input)
        kmeans_input.fit(layer_weight_mtx_reshaped_input)

        # If a cluster has only a single member, then that should not be pruned
        # so that point will always be the closest to the cluster center
        closest_point_to_cluster_center_indices_input = metrics.pairwise_distances_argmin(kmeans_input.cluster_centers_,
                                                                                    layer_weight_mtx_reshaped_input)
        #print('closest_point_to_cluster_center_indices is ', closest_point_to_cluster_center_indices)
        # Compute filter indices which can be pruned
        channel_indices_input = set(np.arange(len(layer_weight_mtx_reshaped_input)))
        channel_indices_to_keep_input = set(closest_point_to_cluster_center_indices_input)
        channel_indices_to_prune_input = list(channel_indices_input.difference(channel_indices_to_keep_input))
        channel_indices_to_keep_input = list(channel_indices_to_keep_input)

        if len(channel_indices_to_keep_input) > input_of_clusters:
            print("Number of selected channels for pruning is less than expected")
            diff = len(channel_indices_to_keep_input) - input_of_clusters
            print("Randomly adding {0} channels for pruning".format(diff))
            np.random.shuffle(channel_indices_to_keep_input)
            for i in range(diff):
                channel_indices_to_prune_input.append(channel_indices_to_keep_input.pop(i))
        elif len(channel_indices_to_keep_input) < input_of_clusters:
            print("Number of selected channels for pruning is greater than expected. Leaving too few channels.")
            diff = input_of_clusters - len(channel_indices_to_keep_input)
            print("Discarding {0} pruneable channels".format(diff))
            for i in range(diff):
                channel_indices_to_keep_input.append(channel_indices_to_prune_input.pop(i))

        if len(channel_indices_to_keep_input) != input_of_clusters:
            raise ValueError(
                "Number of clusters {0} is not equal with the selected "
                "pruneable channels {1}".format(input_of_clusters, len(channel_indices_to_prune_input)))


        return channel_indices_to_prune, channel_indices_to_prune_input

    def _apply_fuzz(self, x):
        for i in range(len(x)):
            self.apply_fuzz_to_vector(x[i])
        return x

    def apply_fuzz_to_vector(self, x):
        # Prepare the vector element indices
        indices = np.arange(0, len(x), dtype=int)
        np.random.shuffle(indices)
        # Select the indices to be modified (always modify only N-1 values)
        nb_of_values_to_modify = np.random.randint(0, len(x) - 1)
        modify_indices = indices[:nb_of_values_to_modify]
        # Modify the selected elements of the vector
        x[modify_indices] += self._FUZZ_EPSILON

    @staticmethod
    def _epsilon(self):
        return BasePruning._FUZZ_EPSILON

    def _calculate_number_of_channels_to_keep(self, keep_factor: float, nb_of_channels: int):
        # This is the number of channels we would like to keep
        # new_nb_of_channels = int(np.ceil(nb_of_channels * keep_factor))
        new_nb_of_channels = int(np.floor(nb_of_channels * keep_factor))
        if new_nb_of_channels > nb_of_channels:
            # This happens when (factor > 1)
            new_nb_of_channels = nb_of_channels
        elif new_nb_of_channels < 1:
            # This happens when (factor <= 0)
            new_nb_of_channels = 1

        # Number of channels which will be removed
        nb_channels_to_remove = nb_of_channels - new_nb_of_channels

        return new_nb_of_channels, nb_channels_to_remove


    # def define_prune_bins(self, channel_number_bins: Union[List[int], np.ndarray],
    #                       pruning_factors_for_bins: Union[List[float], np.ndarray]):
    #     if (len(channel_number_bins) - 1) != len(pruning_factors_for_bins):
    #         raise ValueError("While defining pruning bins, channel numbers list "
    #                          "should contain 1 more items than the pruning factor list")

        self._channel_number_bins = np.asarray(channel_number_bins).astype(int)
        self._pruning_factors_for_channel_bins = np.asarray(pruning_factors_for_bins).astype(float)

    # def _get_pruning_factor_based_on_prune_bins(self, nb_channels: int) -> float:
    #     for i, pruning_factor in enumerate(self._pruning_factors_for_channel_bins):
    #         min_channel_number = self._channel_number_bins[i]
    #         max_channel_number = self._channel_number_bins[i + 1]
    #         if min_channel_number <= nb_channels < max_channel_number:
    #             return self._pruning_factors_for_channel_bins[i]
    #     # If we did not found any match we will return with the default pruning factor value
    #     print("No entry was found for a layer with channel number {0}, "
    #           "so returning pruning factor {1}".format(nb_channels, self._pruning_factor))
    #     return self._pruning_factor



    # @staticmethod
    # def _count_number_of_filters(model: models.Model) -> int:
    #     nb_of_filters = 0
    #     for layer in model.layers:
    #         if layer.__class__.__name__ == "Conv2D":
    #             layer_weight_mtx = layer.get_weights()[0]
    #             _, _, _, channels = layer_weight_mtx.shape
    #             nb_of_filters += channels
    #     return nb_of_filters
    #
    # def _compute_pruning_percent(self, model: models.Model) -> float:
    #     nb_filters = self._count_number_of_filters(model)
    #     left_filters_percent = 1.0 - (nb_filters / self._original_number_of_filters)
    #     return left_filters_percent

    def _save_after_pruning(self, model: models.Model):
        model.save(self._tmp_model_file_name, overwrite=True, include_optimizer=True)

    @staticmethod
    def _clean_up_after_pruning(model: models.Model):
        del model
        K.clear_session()
        tf.reset_default_graph()

    def _load_back_saved_model(self, custom_objects: dict) -> models.Model:
        model = models.load_model(self._tmp_model_file_name, custom_objects=custom_objects)
        return model





