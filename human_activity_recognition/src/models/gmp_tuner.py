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

from keras_tuner_model_utils import check_model_compute_budget
import keras_tuner as kt

@dataclass
class ConvLayerConfig:
    kernel_size: int = 3
    num_filters: int = 16
    max_pooling_size: Optional[int] = 2  # Pooling done BEFORE the layer. None → no max pooling
    optional_layer: Optional[bool] = False

def create_build_model(input_shape: tuple[int] = (24, 3, 1),
                       num_classes: int = 4,
                       max_maccs: int = 0,
                       max_num_params: int = 0):
    def get_layer_conf_hp(layer_name: str,
                          num_filters: List[int], 
                          kernel_sizes: List[int],
                          max_pooling_sizes: Optional[List[int]],
                          optional_layer: bool,
                          hp) -> ConvLayerConfig:
        kernel_size_param = hp.Choice(f'{layer_name} kernel_size', values=kernel_sizes)
        num_filters_param = hp.Choice(f'{layer_name} num_filters', values=num_filters)

        if max_pooling_sizes is None:
            max_pooling_size_param = None
        else:
            max_pooling_size_param = hp.Choice(f'{layer_name} max_pooling_size', values=max_pooling_sizes)

        if optional_layer:
            optional_layer_param = hp.Boolean(f'{layer_name} optional_layer')
        else:
            optional_layer_param = None

        return ConvLayerConfig(
            kernel_size=kernel_size_param,
            num_filters=num_filters_param,
            max_pooling_size=max_pooling_size_param,
            optional_layer=optional_layer_param
        )


    def build_model(hp):

        layer_1_conf = get_layer_conf_hp(
            layer_name='layer_1',
            num_filters=[2, 4, 8, 16, 24],
            kernel_sizes=[3, 5, 7],
            max_pooling_sizes=None,
            optional_layer=False,
            hp=hp
        )

        layer_2_conf = get_layer_conf_hp(
            layer_name='layer_2',
            num_filters=[2, 4, 8, 16, 24],
            kernel_sizes=[3, 5, 7],
            max_pooling_sizes=[1, 2, 4],
            optional_layer=False,
            hp=hp
        )

        layer_3_conf = get_layer_conf_hp(
            layer_name='layer_3',
            num_filters=[2, 4, 8, 16, 24],
            kernel_sizes=[3, 5, 7],
            max_pooling_sizes=[1, 2, 4],
            optional_layer=True,
            hp=hp
        )


        model = get_gmp(
            input_shape=input_shape,
            num_classes=num_classes,
            layer_1_conf=layer_1_conf,
            layer_2_conf=layer_2_conf,
            layer_3_conf=layer_3_conf,
        )

        if max_maccs and max_num_params:
            if not check_model_compute_budget(model, max_maccs,
                                          max_num_params):
                print("[INFO] Model exceeds compute budget")
                raise kt.errors.FailedTrialError(
                    "Compute Budget exceeded"
                )
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

def update_and_check_size_conv_to_max_pool(
    current_size: int,
    kernel_size: int,
    max_pooling_size: int,
    layer_name: str,
):
    # Update current size after convolution
    current_size = current_size - kernel_size + 1

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
    layer_name: str,
):
    # Update current size after max pooling
    current_size //= max_pooling_size

    # Check if convolution is valid
    if kernel_size > current_size:
        raise kt.errors.FailedTrialError(
            f"Model invalid at layer {layer_name}: kernel_size > current_size: {kernel_size} > {current_size}"
        )

    return current_size


def get_gmp(input_shape: tuple[int] = (24, 3, 1),
            num_classes: int = 4,
            layer_1_conf: ConvLayerConfig = ConvLayerConfig(),
            layer_2_conf: ConvLayerConfig = ConvLayerConfig(),
            layer_3_conf: ConvLayerConfig = ConvLayerConfig(),
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

    # Check max pooling sizes
    if layer_1_conf.max_pooling_size:
        raise ValueError("Max pooling for first layer not supported")
    if layer_2_conf.max_pooling_size is None:
        raise ValueError("Max pooling for second layer required")
    if layer_3_conf.max_pooling_size is None:
        raise ValueError("Max pooling for third layer required")

    current_size = input_shape[0]

    model = tf.keras.Sequential()

    # Input block
    model.add(
        layers.Input(shape=input_shape)
    )

    ##
    # Layer 1
    ##
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(layer_1_conf.num_filters,
                            kernel_size=(layer_1_conf.kernel_size, 1),
                            strides=(1, 1),
                            kernel_initializer='glorot_uniform',
                            padding="valid",
                            activation="relu"))

    
    current_size = update_and_check_size_conv_to_max_pool(
        current_size=current_size,
        kernel_size=layer_1_conf.kernel_size,
        max_pooling_size=layer_2_conf.max_pooling_size,
        layer_name="layer_2",
    )

    ##
    # Layer 2
    ##

    if layer_2_conf.max_pooling_size > 1:
        model.add(layers.MaxPooling2D(pool_size=(layer_2_conf.max_pooling_size, 1)))

    current_size = update_and_check_size_max_pool_to_conv(
        current_size=current_size,
        max_pooling_size=layer_2_conf.max_pooling_size,
        kernel_size=layer_2_conf.kernel_size,
        layer_name="layer_2",
    )

    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(layer_2_conf.num_filters,
                            kernel_size=(layer_2_conf.kernel_size, 3),
                            strides=(1, 1),
                            kernel_initializer='glorot_uniform',
                            padding="valid",
                            activation="relu"))

    ##
    # Layer 3
    ##
    if layer_3_conf.optional_layer:
        current_size = update_and_check_size_conv_to_max_pool(
            current_size=current_size,
            kernel_size=layer_2_conf.kernel_size,
            max_pooling_size=layer_3_conf.max_pooling_size,
            layer_name="layer_3",
        )

        if layer_3_conf.max_pooling_size > 1:
            model.add(layers.MaxPooling2D(pool_size=(layer_3_conf.max_pooling_size, 1)))

        current_size = update_and_check_size_max_pool_to_conv(
            current_size=current_size,
            max_pooling_size=layer_3_conf.max_pooling_size,
            kernel_size=layer_3_conf.kernel_size,
            layer_name="layer_3",
        )

        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(layer_3_conf.num_filters,
                                kernel_size=(layer_3_conf.kernel_size, 1),
                                strides=(1, 1),
                                kernel_initializer='glorot_uniform',
                                padding="valid",
                                activation="relu"))

    ##
    # MaxPool + Output Layers
    ##

    # maxpooling
    model.add(
        layers.GlobalMaxPooling2D()
    )

    model.add(
        layers.Dense(num_classes)
    )

    model.add(
        layers.Activation('softmax')
    )

    f1_metric = get_f1_macro_metric()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_metric])

    return model
