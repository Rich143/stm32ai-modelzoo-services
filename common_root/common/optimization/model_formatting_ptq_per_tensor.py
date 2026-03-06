# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import tensorflow as tf
from keras.models import Model
import keras
import numpy as np
import re

from bn_weights_folding import bw_bn_folding

CLE_NEUTRAL_LAYERS_NAMES = ["ReLU", "PreLU", "Dropout", "ZeroPadding2D"]
RELU6_SAT_UP = 6.0


def is_neutral_layer(layer):
    """
        returns True if layer is among so called 'neutral layers' list from equalization point of view or if layer is a
        ReLU or ReLU6
        Args:
            layer: the Keras layer we want to analyse

        Returns: a boolean indicating if the layer is considered as 'neutral' from CLE point of view.
    """

    if layer.__class__.__name__ in CLE_NEUTRAL_LAYERS_NAMES:
        return True
    elif layer.__class__.__name__ == "Activation":
        if layer.get_config()["activation"] == "relu" or layer.get_config()['activation'] == "relu6":
            return True
        else:
            return False
    else:
        return False


def is_relu6(layer):
    """
        returns True if layer is ReLU6
        Args:
            layer: the Keras layer we want to analyse

        Returns: a boolean indicating if the layer is a relu6
    """

    if layer.__class__.__name__ == "Activation":
        if layer.get_config()["activation"] == "relu6":
            return True
    elif layer.__class__.__name__ == "ReLU":
        if 'max_value' in layer.get_config():
            if layer.get_config()['max_value'] is None:
                return False
            elif int(layer.get_config()['max_value']) == 6:
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def bn_parameters(model):
    """
        returns a dictionary with Batch Norm parameters each time a BN immediately follows a DW. To be called before
        folding of course. It will be used for bias absorption later on
        Args:
            model: the Keras model before folding

        Returns: a dictionary with Batch Norm parameters
    """

    bn_parameters_dict = {}

    for i, layer in enumerate(model.layers):
        if layer.__class__.__name__ == "DepthwiseConv2D":
            nodes = layer.outbound_nodes
            list_node_first = [n.layer for n in nodes]
            one_node = len(list_node_first) == 1
            # one_node controls that DW and BN are sequential
            # otherwise algo undefined
            if one_node and list_node_first[0].__class__.__name__ == "BatchNormalization":
                # store name previous DW and gamma, beta
                bn_parameters_dict[layer.name] = [list_node_first[0].get_weights()[0], list_node_first[0].get_weights()[1]]

    return bn_parameters_dict


def high_bias_absorption(model, coupled_index, bn_params_dict, inv_s, n=3):
    """
        implement bias absorption as defined in the Nagel research paper.
        Args:
            model: the Keras model after CLE was executed
            coupled_index: index of couple DW+Conv2d on which was applied CLE
            bn_params_dict: a dictionary with Batch Norm parameters for the original model
            inv_s: inverse of 's' in Nagel's paper. Equalisation coefficient
            n: parameter to approximate Gaussian distribution width

        Returns: a dictionary with Batch Norm parameters

    """

    for k, couple_layer_idx in enumerate(coupled_index):
        if model.layers[couple_layer_idx[0]].name in bn_params_dict:
            gamma = bn_params_dict[model.layers[couple_layer_idx[0]].name][0] * inv_s[k]
            beta = bn_params_dict[model.layers[couple_layer_idx[0]].name][1] * inv_s[k]
            c = tf.nn.relu(beta - n*gamma).numpy()

            # although not described in Nagel's paper, there is a potential issue when too many samples of the
            # activations are above saturation point. In this case, the simplifying assumptions taken by the paper
            # are no longer valid from a math point of view. In this case, we disable bias absorption by setting
            # 'c' to 0 for the corresponding channels
            sat_level = RELU6_SAT_UP * np.array(inv_s[k])
            for q, sat in enumerate(sat_level):
                if beta[q] + n*gamma[q] >= sat:
                    c[q] = 0

            w1 = model.layers[couple_layer_idx[0]].get_weights()[0]
            b1 = model.layers[couple_layer_idx[0]].get_weights()[1]
            new_b1 = b1 - c

            w2 = model.layers[couple_layer_idx[1]].get_weights()[0]
            b2 = model.layers[couple_layer_idx[1]].get_weights()[1]
            # have ch_in first
            w2_tr = np.transpose(w2, (3, 0, 1, 2))
            w2_tr_c = [np.sum(c * channel) for k, channel in enumerate(w2_tr)]
            new_b2 = w2_tr_c + b2

            model.layers[couple_layer_idx[0]].set_weights([w1, new_b1])
            model.layers[couple_layer_idx[1]].set_weights([w2, new_b2])

    return model


def active_number_of_nodes(list_node):
    """
        our BN folding function keeps in the layer list 'ghost' tensors corresponding to BN that was folded.
        To manage this undesired effect, before removing a BN layer from the graph, the folding function 'neutralises'
        it by setting gamma to all ones, beta to all zeros, moving var to all ones, and moving avg to all zeros.
        By detecting these specific values, we can detect a 'ghost' tensor and later on avoid to take it into account in
        graph parsing and reconstruction.

        Args:
            list_node: list of node at a given place in the network graph

        Returns: the number of active nodes in a list after filtering these 'ghost' tensors out.
    """

    tensor_name = ['']
    list_node_filtered = []

    for idx, member in enumerate(list_node):
        if member.__class__.__name__ == 'BatchNormalization':
            # either was not possible to fold or 'ghost'
            moving_avg = np.array(member.moving_mean)
            moving_var = np.array(member.moving_variance)
            if np.all(moving_avg == np.zeros(len(moving_avg))) and np.all(moving_var == np.ones(len(moving_var))):
                # "ghost"
                pass
            else:
                tensor_name.append(member.name)
                list_node_filtered.append(member)
        else:
            if member.name not in tensor_name:
                tensor_name.append(member.name)
                list_node_filtered.append(member)

    return list_node_filtered


def couple_names_and_indexes(model):
    """
           Returns a list of DW/Conv2d couple names when candidate to equalization, and the list of DW/Conv2d
           corresponding indexes. To finish returns the list of ReLU6 layer names when in between DW and Conv2d

           Args:
               model: model after batch norm folding

            Returns: candidate couples for CLE index, names and relu6 layer names
    """

    model_layer_coupled_names = []
    model_layer_coupled_index = []
    relu6_layer_names = []

    for i, layer in enumerate(model.layers):
        if layer.__class__.__name__ == "DepthwiseConv2D":
            nodes = layer.outbound_nodes
            list_node_first = [n.layer for n in nodes]
            list_node_filtered_first = active_number_of_nodes(list_node_first)
            valid_equalization1 = len(list_node_filtered_first) == 1
            # valid_equalization controls that DW and Conv2D or activation are sequential
            # if not, equalization is anyway not clearly specified
            if valid_equalization1 and list_node_filtered_first[0].__class__.__name__ == "Conv2D":
                model_layer_coupled_names.append([layer.name, list_node_filtered_first[0].name])
            elif valid_equalization1 and is_neutral_layer(list_node_filtered_first[0]):
                nodes = list_node_filtered_first[0].outbound_nodes
                list_node_second = [n.layer for n in nodes]
                list_node_filtered_second = active_number_of_nodes(list_node_second)
                valid_equalization2 = len(list_node_filtered_second) == 1
                if valid_equalization2 and list_node_filtered_second[0].__class__.__name__ == "Conv2D":
                    # valid_equalization controls that {DW+activation} and Conv2D are sequential
                    # if not, equalization is anyway not clearly specified
                    model_layer_coupled_names.append([layer.name, list_node_filtered_second[0].name])
                    if is_relu6(list_node_filtered_first[0]):
                        relu6_layer_names.append(list_node_filtered_first[0].name)

    for name_layer in model_layer_coupled_names:
        sub_list = []
        sub_list.append([i for i, layer in enumerate(model.layers) if layer.name == name_layer[0]][0])
        sub_list.append([i for i, layer in enumerate(model.layers) if layer.name == name_layer[1]][0])
        model_layer_coupled_index.append(sub_list)

    return model_layer_coupled_names, model_layer_coupled_index, relu6_layer_names


def choose_tensors_when_multiple_outputs(layer_input_tensor, layer_input_signature):

    layer_input_selection = []
    list_signature_names = []

    if type(layer_input_signature) is list:
        # print('layer in signature first input {}'.format(layer_input_signature[0].name))
        for elem in layer_input_signature:
            if hasattr(elem, 'name'):
                list_signature_names.append(elem.name)
    else:
        list_signature_names = [layer_input_signature.name]

    for elem in layer_input_tensor:
        if type(elem) is tuple:
            for sub_elem in elem:
                if sub_elem.name in list_signature_names:
                    layer_input_selection.append(sub_elem)
        else:
            layer_input_selection.append(elem)

    return layer_input_selection


def insert_layer_in_graph(model, layer_list, insert_layer, inv_scale, insert_layer_name=None, position='replace'):
    """
        Returns a model where some layers (layer_List) have been replaced by a new layer type 'insert_layer' with
        as parameter an element of 'inv_scale'

        Args:
            model: keras model after CLE and bias absorption
            layer_list: list of layer names we want to replace in the graph
            insert_layer: the layer we want to insert in replacement in the graph
            inv_scale: inverse of 's' in Nagel's paper. Equalisation coefficient
            insert_layer_name: name of the layer we insert. Not used at the moment
            position: could be 'replace', 'after', 'before'. Always 'replace' for CLE

        Returns: a keras model with specified layers replaced by new insert_layer
    """

    # early exit
    if not layer_list:
        return model
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer.outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                # condition added Jan16, 2024, because due to duplication of a tensor in outbound_nodes (2 same tensors
                # instead of 1 expected), unexplained, causing issues later on at conversion/evaluation
                # by having this condition we may lose some generality in some corner cases, but should be rare
                if layer.name not in network_dict['input_layers_of'][layer_name]:
                    network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update({model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    count_scale = 0

    for layer in model.layers[1:]:
        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                       for layer_aux in network_dict['input_layers_of'][layer.name]]
        layer_input = choose_tensors_when_multiple_outputs(layer_input, layer.input)

        if len(layer_input) == 1:
            layer_input = layer_input[0]
            nb_inputs_layer = 1
        else:
            nb_inputs_layer = len(layer_input)

        # Insert layer if name matches the regular expression
        if layer.name in layer_list:
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            #debug
            #.print(layer.name + ' outside if ')
            #print(insert_layer.__class__.__name__ + ' outside if ')
            # function if adaptive_clip
            if insert_layer.__class__.__name__ == 'ReLU':
                new_layer = insert_layer()
                new_layer._name = '{}_{}'.format(layer.name, 'modified_to_relu')
                x = new_layer(x)
            elif (insert_layer.__class__.__name__ == 'function' or
                  insert_layer.__class__.__name__ == 'cython_function_or_method'):
                # adaptive clip
                #print(insert_layer.__class__.__name__ + ' inside if ' + '\n')
                x = insert_layer(t=x, invs=inv_scale[count_scale])
            else:
                pass

            count_scale = count_scale + 1

            if position == 'before':
                x = layer(x)
        else:
            if layer.__class__.__name__ == 'TFOpLambda' or layer.__class__.__name__ == 'SlicingOpLambda':
                kwargs = dict(layer.output.node.call_kwargs)
                layer_config_dict = layer.get_config()

                if len(layer.output.node.call_args) < len(layer.output.node.keras_inputs):
                    # 'abnormal case' where inputs number is lower than expected inputs number from graph parsing.
                    # It means that one graph input is treated among the kwargs instead of the args. So the fix is to
                    # pop the kwargs, and use the tensor as an actual input
                    list_key_to_remove = [k for k in kwargs.keys() if hasattr(kwargs[k], 'is_tensor_like') if
                                          kwargs[k].is_tensor_like]
                    if list_key_to_remove:
                        for k in list_key_to_remove:
                            kwargs.pop(k, None)
                        if nb_inputs_layer == 2:
                            x = layer(layer_input[0], layer_input[1], **kwargs)
                        elif nb_inputs_layer == 3:
                            x = layer(layer_input[0], layer_input[1], layer_input[2], **kwargs)
                        else:
                            print("Unsupported Lambda layer {} with {} inputs".format(layer.name, nb_inputs_layer))
                    else:
                        # standard list of inputs tensor
                        x = layer(layer_input, **kwargs)
                elif layer.__class__.__name__ == 'TFOpLambda' and layer_config_dict['function'] == 'tile':
                    kwargs['multiples'][0] = layer_input
                    x = layer(layer.output.node.call_args[0], **kwargs)
                else:
                    # standard list of inputs tensor
                    x = layer(layer_input, **kwargs)
            else:
                x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model at origin, if layer_name
        if layer.name in model.output_names:
            model_outputs.append(x)

    return Model(inputs=model.inputs, outputs=model_outputs)


def cross_layer_equalisation(model, coupled_index):
    """
        Returns a model where couple layers weights are equalized as described in Nagel's paper

        Args:
            model: keras model after folding
            coupled_index: index of all the couples DW/Conv2d eligible to equalisation

        Returns: a model with weights and bias updated by CLE, and the list of inverse equalisation coefficients
    """

    eps = 0.0
    list_inv_s = []

    for couple_layer_idx in coupled_index:
        w1 = model.layers[couple_layer_idx[0]].get_weights()[0]
        b1 = model.layers[couple_layer_idx[0]].get_weights()[1]
        # have ch_out first
        w1_tr = np.transpose(w1, (2, 0, 1, 3))

        w2 = model.layers[couple_layer_idx[1]].get_weights()[0]
        b2 = model.layers[couple_layer_idx[1]].get_weights()[1]
        # have ch_in first
        w2_tr = np.transpose(w2, (2, 0, 1, 3))

        # vector s calculation
        r1 = [np.max(e) - np.min(e) for e in w1_tr]
        r2 = [np.max(e) - np.min(e) for e in w2_tr]
        s = [1/(r2[k] + eps) * np.sqrt(r1[k] * r2[k]) for k in range(len(r1))]

        # 22 Jan 2024: treat the corner case where s(k) == 0 in this case it would be impossible to calculate 1/s(k)
        # In case r1(k) was null we can set s(k) to 1 because there is no need in this case to scale down this channel
        # weights, since in any case they are null
        for idx, e in enumerate(s):
            if e == 0 and r1[idx] == 0:
                s[idx] = 1

        inv_s = [1/(e + eps) for e in s]
        list_inv_s.append(inv_s)

        new_w1_tr = [inv_s[k]*channel for k, channel in enumerate(w1_tr)]
        new_w1 = np.array(np.transpose(new_w1_tr, (1, 2, 0, 3)))
        new_b1 = inv_s * b1

        new_w2_tr = [s[k]*channel for k, channel in enumerate(w2_tr)]
        new_w2 = np.array(np.transpose(new_w2_tr, (1, 2, 0, 3)))

        model.layers[couple_layer_idx[0]].set_weights([new_w1, new_b1])
        model.layers[couple_layer_idx[1]].set_weights([new_w2, b2])

    return model, list_inv_s


def zero_irrelevant_channels(model, min_weights_th, ct_value=0.0):
    """
        Returns a model with weights arbitrarily set to constant value typically 0, if all weights corresponding to a
        given output channel are below 'min_weight_th' in absolute value. Restricted to Conv2d and DW.
        This helps reducing possible bias saturation issue at quantization, when weights channel scale is very small

        Args:
            model: keras model after batch normalisation folding
            min_weights_th: arbitrary threshold under which we consider current weights to be replaced by 'ct_value'
            ct_value: constant value set to the weights when they are < min_weights_th for a given channel. For
            this application ct_value is always set to 0.0

        Returns: the keras model with weights updated

    """

    for layer in model.layers:

        if layer.__class__.__name__ == 'Functional':
            zero_irrelevant_channels(layer, min_weights_th)
        if layer.__class__.__name__ in ("Conv2D", "DepthwiseConv2D"):
            # weights
            bias_exist = len(layer.get_weights()) == 2
            if bias_exist:
                w = layer.get_weights()[0]
                b = layer.get_weights()[1]
            else:
                w = layer.get_weights()[0]
            if layer.__class__.__name__ == "DepthwiseConv2D":
                # have ch_out first
                w = np.transpose(w, (2, 0, 1, 3))
                for i, we in enumerate(w):
                    if np.abs(np.min(we)) < min_weights_th and np.abs(np.max(we)) < min_weights_th:
                        w[i] = ct_value * np.ones((w.shape[1], w.shape[2], w.shape[3]))
                w = np.transpose(w, (1, 2, 0, 3))
                if bias_exist:
                    layer.set_weights([w, b])
                else:
                    layer.set_weights([w])
            elif layer.__class__.__name__ == "Conv2D":
                # have ch_out first
                w = np.transpose(w, (3, 0, 1, 2))
                for i, we in enumerate(w):
                    if np.abs(np.min(we)) < min_weights_th and np.abs(np.max(we)) < min_weights_th:
                        w[i] = ct_value * np.ones((w.shape[1], w.shape[2], w.shape[3]))
                w = np.transpose(w, (1, 2, 3, 0))
                if bias_exist:
                    layer.set_weights([w, b])
                else:
                    layer.set_weights([w])

    return model


def adaptive_clip_per_channel(t=None, invs=None):
    """
        Returns a layer for adaptive channel clipping whose level is given through 'invs'
        Restricted to ReLU6

        Args:
            t: a Keras tensor input of the adpative clip per channel layer
            invs: list of equalisation coefficients as described on Nagel's paper

        Returns:
            A tensorflow layer for adaptive clipping per-channel
    """
    nb_ch_out = int(t.shape.dims[-1])

    ch_sat_level = [RELU6_SAT_UP*k for k in invs]
    scale = (np.max(ch_sat_level) - np.min(ch_sat_level)) / 65535  # 255.0
    ch_sat_level = np.round(ch_sat_level / scale) * scale
    # although not useful from a math point of view since the following clip has clip_min == 0, the addition of this
    # relu before the clip will make the interpreter understand it needs to fuse it with previous layer which helps
    # reducing the dynamic range of the layer output and thus to find a smaller scale and eventually reduce the
    # quantization noise.
    t = tf.nn.relu(t)

    return tf.clip_by_value(t, clip_value_min=np.zeros(nb_ch_out), clip_value_max=ch_sat_level)


def model_formatting_ptq_per_tensor(model_origin):
    """
        Returns a keras model after all the PTQ optimization chain was executed:
            - batch norm folding
            - zeroing irrelevant channels (too weak)
            - cross layer equalisation (CLE)
            - bias absorption
            - insertion of the adaptive clip layers wherever needed

        Args:
            model_origin: the original Keras model

        Returns:
            A Keras model optimized for subsequent per-tensor quantization
    """

    # keep in memory BN parameters for future bias absorption
    bn_params_dict = bn_parameters(model_origin)

    # BN folding
    model_folded = bw_bn_folding(model_origin, dead_channel_th=1e-10)

    # zeroing some channels to avoid bias saturation at quantization
    model_folded = zero_irrelevant_channels(model_folded, min_weights_th=1e-10)

    # extract layer couples names and indexes for equalization
    layer_coupled_names, layer_coupled_index, layer_to_replace_names = couple_names_and_indexes(model_folded)

    # performs Nagel's paper cross-layer equalization on selected couples
    model_cle, list_inv_s = cross_layer_equalisation(model=model_folded, coupled_index=layer_coupled_index)

    # performs bias absorption, which is optional
    model_cle = high_bias_absorption(model=model_cle, coupled_index=layer_coupled_index, inv_s=list_inv_s,
                                     bn_params_dict=bn_params_dict, n=3)

    # insert adaptive channel clipping layers at the right places in the graph
    model_cle = insert_layer_in_graph(model=model_cle, layer_list=layer_to_replace_names,
                                      insert_layer=adaptive_clip_per_channel, inv_scale=list_inv_s,
                                      insert_layer_name=None,
                                      position='replace')
    #model_cle.save(save_path)
    return model_cle
