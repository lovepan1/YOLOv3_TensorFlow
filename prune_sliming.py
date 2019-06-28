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
from model_sliming import sliming_yolov3

# from ridurre import base_filter_pruning
anchor_path = "./data/yolo_anchors.txt"
class_name_path  = "./data/my_data/dianli_class.names"
anchors = parse_anchors(anchor_path)
num_class = len(read_class_names(class_name_path))

class SlimPruning:
    _FUZZ_EPSILON = 1e-5

    def __init__(self,
                 pruning_factor: float,
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

        # TODO: select a subset of layers to prune
        self._prunable_layers_regex = ".*"
        self._restore_part_first = ['yolov3/darknet53_body']
        self._restore_part_second = ['yolov3/darknet53_body','yolov3/yolov3_head']
        self._update_part = ['yolov3/yolov3_head']
        self._img_size = [412, 412]

    def run_pruning(self, prune_factor = 0.8,
                    custom_objects_inside_model: dict = None) -> Tuple[models.Model, int]:

        pruning_iteration = 0
        # with tf.Session() as sess:
        while True:
            if prune_factor is not None:
                self._pruning_factor = prune_factor

                # Pruning step
                print("Running filter pruning {0}".format(pruning_iteration))
                weight_dict = self._prune_first_stage()
                sliming_yolo_model = self._reconstruction_model()
                pruning_iteration += 1
                if self._maximum_prune_iterations is not None:
                    if pruning_iteration > self._maximum_prune_iterations:
                        break

            print("Pruning stopped.")
            # return model, self._current_nb_of_epochs
            return sliming_yolo_model, weight_dict

    def _reconstruction_model(self, weight_dict):
        with tf.Session() as sess:
            input_data_2 = tf.placeholder(tf.float32, [1, self._img_size[1], self._img_size[0], 3], name='input_data')
            sliming_yolo_model = sliming_yolov3(num_class, anchors)
            with tf.variable_scope('yolov3'):
                pred_feature_maps = sliming_yolo_model.forward_include_res_with_prune_factor(input_data_2,
                                                                                          prune_factor=self._pruning_factor,
                                                                                          is_training=True)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            for layer_name, weight in weight_dict.items():
                sess.run(tf.assign(layer_name, weight, validate_shape=True))
            print("reconstruction network completed")
            print("completed initialized")
            saver_to_restore = tf.train.Saver(
                var_list=tf.contrib.framework.get_variables_to_restore(include=self._restore_part_second))
            update_vars = tf.contrib.framework.get_variables_to_restore(include=self._update_part)
            # saver_to_restore.restore(sess, prune_check_point_path)
            # saver_to_restore.save(sess, os.path.join(self._checkpoint_dir, 'sliming_prune_model.ckpt'))
            saver_best = tf.train.Saver()
            saver_best.save(sess, os.path.join(self._checkpoint_dir, 'sliming_prune_model.ckpt'))
            return sliming_yolo_model




    def _run_pruning_for_conv2d_layer(self, pruning_factor: float, gamma_layer_weight_mtx):
        _nb_channels = gamma_layer_weight_mtx.shape
        abs_weight = list(map(abs, gamma_layer_weight_mtx))
        sorted_weight = np.sort(abs_weight)
        pruning_threshold = sorted_weight[  np.ceil(  len(abs_weight) * pruning_factor ) ]
        bool_weight_indice = np.logical_not(abs_weight < pruning_threshold)
        save_indices = []
        for i, j in enumerate(bool_weight_indice):
            if j == True:
                save_indices.append[i]
        # Compute filter indices which can be pruned
        channel_indices = set(np.arange(len(gamma_layer_weight_mtx)))
        channel_indices_to_keep = set(save_indices)
        channel_indices_to_prune = list(channel_indices.difference(channel_indices_to_keep))
        channel_indices_to_keep = list(channel_indices_to_keep)

        nb_of_clusters = len(save_indices)
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
        return channel_indices_to_prune

    def _prune_first_stage(self):
        tf_weights_value = []
        layer_name_weights_dict = dict()
        checkpoint_path = os.path.join(self._checkpoint_dir, "best_model_Epoch_2_step_2024.0_mAP_0.1784_loss_30.0785_lr_0.0001")
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for layer_name in var_to_shape_map:
            layer_name_weights_dict[layer_name + ':0'] = reader.get_tensor(layer_name)
        pruning_factor = self._pruning_factor
        for layer_name, layer_weight in layer_name_weights_dict.items():
            if 'weights' in layer_name:
                ###### prune currnet channel ##############
                ####g get bn layer#################
                current_layer_name = layer_name
                current_bn_gamma_layer = current_layer_name.replace('weights', 'BatchNorm/gamma')
                current_gamma_layer_weight = layer_name_weights_dict[current_bn_gamma_layer]
                filter_indices_to_prune = self._run_pruning_for_conv2d_layer(pruning_factor, current_gamma_layer_weight)
                print('filter_indices_to_prune is ', filter_indices_to_prune)
                W, H, N, nb_channels = layer_weight.shape
                print("layer_weight.shape is ", layer_weight.shape)
                prune_weight = np.delete(layer_weight, filter_indices_to_prune, axis=-1)
                _, _, _, prun_channel = prune_weight.shape
                print('prun_channel is ', prun_channel)
                print('calc prune channel is ', nb_channels - len(filter_indices_to_prune))
                print("prun weight shape is", prune_weight.shape)
                layer_name_weights_dict[layer_name] = prune_weight
                ###########################################

                #######prune currne layer BN params########
                bn_params = ['BatchNorm/gamma', 'BatchNorm/beta', 'BatchNorm/moving_variance',
                             'BatchNorm/moving_mean']
                bn_layer_name = []
                for i in bn_params:
                    bn_params_str = layer_name.replace('weights', i)
                    bn_layer_name.append(bn_params_str)
                for bn_layer in bn_layer_name:
                    bn_param = layer_name_weights_dict[bn_layer]
                    bn_filter_prune = filter_indices_to_prune
                    prune_bn_param = np.delete(bn_param, bn_filter_prune, axis=0)
                    layer_name_weights_dict[bn_layer] = prune_bn_param
                    print('current layer is ', bn_layer)
                    print("bn param.shape is ", bn_param.shape)
                    print('prune bn param shape is ', layer_name_weights_dict[bn_layer].shape)
                ###########################################

                ##### prune next input channels #############
                try:
                    next_layer_number = int(layer_name.split('/')[-2].split('_')[-1]) + 1
                except:
                    next_layer_number = 1
                if 'darknet' in layer_name:
                    next_layer_name = 'yolov3/darknet53_body/Conv_' + str(next_layer_number) + '/weights:0'
                elif 'yolo_head' in layer_name:
                    next_layer_name = 'yolov3/yolo_head/Conv_'+ str(next_layer_number) + '/weights:0'

                if next_layer_number != 52:
                    print("the next layer is ", next_layer_name)
                    ######prune input filter weight######
                    # if 'yolov3/darknet53_body/Conv/weights' not in layer_name:  ### cannot prune the first conv
                    next_layer_weight = layer_name_weights_dict[next_layer_name]
                    W, H, input_channels, nb_channels_2 = next_layer_weight.shape
                    prun_input_next_layer_weight = np.delete(next_layer_weight, filter_indices_to_prune, axis=-2)
                    layer_name_weights_dict[next_layer_name] = prun_input_next_layer_weight
                    print('next layer is ', next_layer_name)
                    print('orignal input shape is ', (W, H, input_channels, nb_channels_2 ))
                    print('prun_input shape is ', layer_name_weights_dict[next_layer_name].shape)
                    print('calc prune channel input is ', input_channels - len(filter_indices_to_prune))
        np.save('weights.npy', layer_name_weights_dict)
        return layer_name_weights_dict

    # def _reconstruction_model(self):
    #     prun_graph = tf.Graph()
    #     with tf.Session(graph=prun_graph) as sess:
    #         prune_check_point_path = os.path.join(self._checkpoint_dir, 'prue_channel_model.ckpt')
    #         input_data_2 = tf.placeholder(tf.float32, [1, self._img_size[1], self._img_size[0], 3], name='input_data')
    #         yolo_prun_model = sparse_yolov3(num_class, anchors)
    #         with tf.variable_scope('yolov3'):
    #             pred_feature_maps = yolo_prun_model.forward_include_res_with_prune_factor(input_data_2,
    #                                                                                       prune_factor=self._pruning_factor,
    #                                                                                       is_training=True)
    #         print("prune network completed")
    #         sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    #         print("completed initialized")
    #         saver_to_restore = tf.train.Saver(
    #             var_list=tf.contrib.framework.get_variables_to_restore(include=self._restore_part_first))
    #         update_vars = tf.contrib.framework.get_variables_to_restore(include=self._update_part)
    #         saver_to_restore.restore(sess, prune_check_point_path)
    #         saver_to_restore.save(sess, os.path.join(self._checkpoint_dir, 'kmeans_prune_restore_model.ckpt'))
    #         saver_best = tf.train.Saver()
    #         saver_best.save(sess, os.path.join(self._checkpoint_dir, 'kmeans_prune_restore_model_all.ckpt'))
    #         return yolo_prun_model



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





