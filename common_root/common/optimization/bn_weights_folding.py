# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import tensorflow as tf
from keras.models import Model
import numpy as np


def fold_bn_in_weights(weights, bias, gamma, beta, moving_avg, moving_var, epsilon=1e-3):
    """
         Implements equation for Backward BN weights folding.
         Args:
              weights: original weights
              bias: original bias
              gamma: multiplicative trainable parameter of the batch normalisation. Per-channel
              beta: additive trainable parameter of the batch normalisation. Per-channel
              moving_avg: moving average of the layer output. Used for centering the samples distribution after
              batch normalisation
              moving_var: moving variance of the layer output. Used for reducing the samples distribution after batch
              normalisation
              epsilon: a small number to void dividing by 0
         Returns: folded weights and bias
    """

    scaler = gamma / np.sqrt(moving_var + epsilon)
    weights_prime = [scaler[k] * channel for k, channel in enumerate(weights)]
    bias_prime = scaler * (bias - moving_avg) + beta

    return weights_prime, bias_prime


def bw_bn_folding(model, epsilon=1e-3, dead_channel_th=1e-10):
    """
        Search for BN to fold in Backward direction. Neutralise them before removal by setting gamma to all ones, beta
        to all zeros, moving_avg to all zeros and moving_var to all ones

        Args:
            model: input keras model
            epsilon: a small number to avoid dividing dy 0.0
            dead_channel_th: a threshold (very small) on moving avg and var below which channel is considered as dead
            with respect to the weights

        Returns: a keras model, with BN folded in backward direction, BN neutralised and then removed form the graph
    """
    folded_bn_name_list = []

    list_layers = model.layers
    for i, layer in enumerate(model.layers):
        if layer.__class__.__name__ == 'Functional':
            list_layers = model.layers[i].layers
            break

    for i, layer in enumerate(list_layers):
        if layer.__class__.__name__ == "DepthwiseConv2D" or layer.__class__.__name__ == "Conv2D":
            nodes = layer.outbound_nodes
            list_node_first = [n.layer for n in nodes]
            one_node = len(list_node_first) == 1
            # one_node controls that DW and BN are sequential
            # otherwise algo undefined
            if one_node and list_node_first[0].__class__.__name__ == "BatchNormalization":
                # store name previous DW and gamma, beta
                gamma = list_node_first[0].get_weights()[0]
                beta = list_node_first[0].get_weights()[1]
                moving_avg = list_node_first[0].get_weights()[2]
                moving_var = list_node_first[0].get_weights()[3]

                bias_exist = len(layer.get_weights()) == 2
                if bias_exist:
                    w = layer.get_weights()[0]
                    b = layer.get_weights()[1]
                else:
                    w = layer.get_weights()[0]
                    layer.use_bias = True
                    b = layer.bias = layer.add_weight(name=layer.name + '/kernel_1',
                                                      shape=(len(moving_avg),), initializer='zeros')

                if layer.__class__.__name__ == "DepthwiseConv2D":
                    # dead channel feature:
                    # when at the BN level there is a moving avg AND a moving variance very weak, it means the given
                    # channel is dead. Most probably the input channel was already close to zero, if the layer was a DW
                    # in fact this channel is dead w.r.t weight but plays a role with bias and beta. If the bias was
                    # used, most probably moving_avg would not be that weak
                    # however a very small moving_avg results in a great increase of weight dynamics for this channel
                    # which brings nothing in the end. This would degrade per-tensor quantization

                    for k, value in enumerate(moving_var):
                        if moving_var[k] <= dead_channel_th and moving_avg[k] <= dead_channel_th:
                            moving_var[k] = 1.0
                            moving_avg[k] = 0.0
                            gamma[k] = 0.0

                    # have ch_out first
                    w = np.transpose(w, (2, 0, 1, 3))
                    w, b = fold_bn_in_weights(weights=w, bias=b, gamma=gamma, beta=beta, moving_avg=moving_avg,
                                              moving_var=moving_var, epsilon=epsilon)
                    w = np.transpose(w, (1, 2, 0, 3))
                    layer.set_weights([w, b])
                elif layer.__class__.__name__ == "Conv2D":
                    # have ch_out first
                    w = np.transpose(w, (3, 0, 1, 2))
                    w, b = fold_bn_in_weights(weights=w, bias=b, gamma=gamma, beta=beta, moving_avg=moving_avg,
                                              moving_var=moving_var, epsilon=epsilon)
                    w = np.transpose(w, (1, 2, 3, 0))
                    layer.set_weights([w, b])

                # neutralise BN
                list_node_first[0].set_weights(
                    [np.ones(len(gamma)), np.zeros(len(beta)), np.zeros(len(moving_avg)), np.ones(len(moving_var))])
                folded_bn_name_list.append(list_node_first[0].name)

    from model_formatting_ptq_per_tensor import insert_layer_in_graph
    model_folded = insert_layer_in_graph(model, layer_list=folded_bn_name_list, insert_layer=None,
                                         inv_scale=None, insert_layer_name=None, position='replace')

    return model_folded
