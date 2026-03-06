# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/


import tensorflow as tf
import tensorflow.keras.layers as layers
from dataclasses import dataclass
from typing import Optional, List
from enum import StrEnum

from keras_tuner_model_utils import check_model_compute_budget
import keras_tuner as kt

class GlobalPoolingType(StrEnum):
    AVG = 'avg'
    MAX = 'max'

@dataclass
class ConvLayerConfig:
    kernel_size: int = 3
    num_filters: int = 16
    dilation_rate: int = 1
    max_pooling_size: Optional[int] = 2  # Pooling done BEFORE the layer. None → no max pooling

def create_build_model(input_shape: tuple[int, int, int] = (24, 3, 1),
                       num_classes: int = 4,
                       max_maccs: int = 0,
                       max_num_params: int = 0):
    def get_layer_conf_hp(layer_name: str,
                          num_filters: List[int], 
                          kernel_sizes: List[int],
                          dilation_rates: List[int],
                          max_pooling_sizes: Optional[List[int]],
                          hp) -> ConvLayerConfig:
        kernel_size_param = hp.Choice(f'{layer_name} kernel_size', values=kernel_sizes)
        num_filters_param = hp.Choice(f'{layer_name} num_filters', values=num_filters)
        dilation_rate_param = hp.Choice(f'{layer_name} dilation_rate', values=dilation_rates)

        if max_pooling_sizes is None:
            max_pooling_size_param = None
        else:
            max_pooling_size_param = hp.Choice(f'{layer_name} max_pooling_size', values=max_pooling_sizes)

        return ConvLayerConfig(
            kernel_size=kernel_size_param,
            num_filters=num_filters_param,
            dilation_rate=dilation_rate_param,
            max_pooling_size=max_pooling_size_param,
        )


    def build_model(hp):
        num_depthwise_layers = hp.Int('depthwise_layers', min_value=3,
                                      max_value=4, step=1)

        global_pooling_type = GlobalPoolingType(
            hp.Choice('global_pooling_type',
                      values=[p.value for p in GlobalPoolingType],
                      default=GlobalPoolingType.MAX.value))

        # Layer at which x, y and z axis are mixed (0 based layer index)
        axis_mixing_layer = hp.Choice('axis_mixing_layer',
                                      values=[1, 2],
                                      default=1)

        layer_1_conf = get_layer_conf_hp(
            layer_name='layer_1',
            num_filters=[4, 8, 16, 24, 32],
            kernel_sizes=[3, 5, 7],
            dilation_rates=[1, 2],
            max_pooling_sizes=None,
            hp=hp
        )

        layer_2_conf = get_layer_conf_hp(
            layer_name='layer_2',
            num_filters=[4, 8, 16, 24, 32],
            kernel_sizes=[3, 5, 7],
            dilation_rates=[1, 2, 3, 4],
            max_pooling_sizes=[1, 2, 4],
            hp=hp
        )

        layer_3_conf = get_layer_conf_hp(
            layer_name='layer_3',
            num_filters=[8, 16, 24, 32, 40],
            kernel_sizes=[3, 5],
            dilation_rates=[1, 2, 3, 4],
            max_pooling_sizes=[1, 2, 4],
            hp=hp
        )

        layer_4_conf = get_layer_conf_hp(
            layer_name='layer_4',
            num_filters=[16, 24, 32, 40],
            kernel_sizes=[3, 5],
            dilation_rates=[1, 2, 3, 4],
            max_pooling_sizes=[1, 2, 4],
            hp=hp
        )

        layer_5_conf = get_layer_conf_hp(
            layer_name='layer_5',
            num_filters=[16, 24, 32, 40],
            kernel_sizes=[3, 5],
            dilation_rates=[1, 2, 3, 4],
            max_pooling_sizes=[1, 2, 4],
            hp=hp
        )

        layer_confs = [layer_1_conf, layer_2_conf,
                       layer_3_conf, layer_4_conf, layer_5_conf]


        model = get_ddcnn(
            input_shape=input_shape,
            num_classes=num_classes,
            layer_confs=layer_confs,
            num_depthwise_layers=num_depthwise_layers,
            global_pooling_type=global_pooling_type,
            axis_mixing_layer=axis_mixing_layer
        )

        if max_maccs and max_num_params:
            if not check_model_compute_budget(model, max_maccs,
                                          max_num_params):
                print("[INFO] Model exceeds compute budget")
                raise kt.errors.FailedTrialError(
                    "Compute Budget exceeded"
                )

        model.summary()

        return model

    return build_model

def get_pr_auc_metric():
    return tf.keras.metrics.AUC(
            curve="PR",
            multi_label=False,
            name="auc_pr"
        )

def get_f1_macro_metric():
    return tf.keras.metrics.F1Score(
            average="macro",
            name="f1_macro"
        )

def get_effective_kernel_size(
    kernel_size: int,
    dilation_rate: int
):
    return kernel_size + (kernel_size - 1) * (dilation_rate - 1)

def update_and_check_size_conv_to_max_pool(
    current_size: int,
    kernel_size: int,
    dilation_rate: int,
    max_pooling_size: int,
    layer_name: str,
):
    k_effective = get_effective_kernel_size(kernel_size, dilation_rate)

    # Update current size after convolution
    current_size = current_size - k_effective + 1

    # Check if max pool is valid
    if max_pooling_size > current_size:
        raise kt.errors.FailedTrialError(
            f"Model invalid at layer {layer_name}: max_pooling_size > current_size: {max_pooling_size} > {current_size}"
        )

    return current_size

def update_and_check_size_max_pool_to_conv(
    current_size: int,
    max_pooling_size: int,
    kernel_size: int,
    dilation_rate: int,
    layer_name: str,
):
    k_effective = get_effective_kernel_size(kernel_size, dilation_rate)

    # Update current size after max pooling
    current_size //= max_pooling_size

    # Check if convolution is valid
    if k_effective > current_size:
        raise kt.errors.FailedTrialError(
            f"Model invalid at layer {layer_name}: kernel_size (effective) > current_size: {k_effective} > {current_size}"
        )

    return current_size


# Depthwise Dilated CNN
def get_ddcnn(input_shape: tuple[int, int, int] = (24, 3, 1),
              num_classes: int = 4,
              layer_confs: List[ConvLayerConfig] = [ConvLayerConfig()],
              num_depthwise_layers: int = 1,
              global_pooling_type: GlobalPoolingType = GlobalPoolingType.MAX,
              axis_mixing_layer: int = 1,
           ):
    """
    Builds and returns an gmp model for human_activity_recognition.
    Args:
        input_shape (tuple): A dictionary containing the configuration for the model.
        num_classes (int): number of nodes in the output layer
        dropout (float): dropout ratio to be used for dropout layer
        num_conv_layers (int): number of convolutional layers
    Returns:
        - keras.Model object, the gmp model.
    """

    current_size = input_shape[0]

    model = tf.keras.Sequential()

    # Input block
    model.add(
        layers.Input(shape=input_shape)
    )

    ##
    # Layer 0
    ##
    layer_0_conf = layer_confs[0]

    if layer_0_conf.max_pooling_size:
        raise ValueError("Max pooling for layer 0 not supported")

    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(
        layer_0_conf.num_filters,
        kernel_size=(layer_0_conf.kernel_size, 1),
        strides=(1, 1),
        dilation_rate=(layer_0_conf.dilation_rate, 1),
        padding="valid",
        activation="relu"))

    ##
    # Layer 1-4
    ##
    for layer in range(1, num_depthwise_layers + 1, 1):
        cur_layer_conf = layer_confs[layer]
        prev_layer_conf = layer_confs[layer - 1]

        if cur_layer_conf.max_pooling_size is None:
            raise ValueError(f"Max pooling for layer {layer} required")

        current_size = update_and_check_size_conv_to_max_pool(
            current_size=current_size,
            kernel_size=prev_layer_conf.kernel_size,
            dilation_rate=prev_layer_conf.dilation_rate,
            max_pooling_size=cur_layer_conf.max_pooling_size,
            layer_name=f"layer_{layer}",
        )

        if cur_layer_conf.max_pooling_size > 1:
            model.add(layers.MaxPooling2D(pool_size=(cur_layer_conf.max_pooling_size, 1)))

        current_size = update_and_check_size_max_pool_to_conv(
            current_size=current_size,
            max_pooling_size=cur_layer_conf.max_pooling_size,
            kernel_size=cur_layer_conf.kernel_size,
            dilation_rate=cur_layer_conf.dilation_rate,
            layer_name=f"layer_{layer}",
        )

        kernel_height = 1
        if layer == axis_mixing_layer:
            # Set kernel height to 3 to mix x, y, and z axes
            kernel_height = 3

        model.add(layers.BatchNormalization())
        model.add(layers.SeparableConv2D(
            cur_layer_conf.num_filters,
            kernel_size=(cur_layer_conf.kernel_size, kernel_height),
            strides=(1, 1),
            dilation_rate=(cur_layer_conf.dilation_rate, 1),
            padding="valid",
            activation="relu"))

    ##
    # Global Pool + Output Layers
    ##

    # maxpooling
    if global_pooling_type == GlobalPoolingType.AVG:
        model.add(layers.GlobalAveragePooling2D())
    elif global_pooling_type == GlobalPoolingType.MAX:
        model.add(layers.GlobalMaxPooling2D())
    else:
        raise ValueError(f"Invalid global pooling type: {global_pooling_type}")

    model.add(
        layers.Dense(num_classes)
    )

    model.add(
        layers.Activation('softmax')
    )

    f1_metric = get_f1_macro_metric()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_metric])

    return model
