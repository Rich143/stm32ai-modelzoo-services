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

import optuna
from optuna.trial import TrialState

@dataclass
class ConvLayerConfig:
    kernel_size_min: int = 3
    kernel_size_max: int = 7

    num_filter_min: int = 4
    num_filter_max: int = 40
    num_filters_step: int = 8

    # Uses log option
    dilation_rate_min: int = 1
    dilation_rate_max: int = 4

@dataclass
class MaxPoolLayerConfig:
    # Max pooling before conv
    max_pooling_enabled: bool = True
    max_pooling_size_min: int = 2
    max_pooling_size_max: int = 4

def get_max_kernel_size(layer_input_shape: tuple[int, int, int],
                        kernel_size_max_configured: int):
    length = layer_input_shape[0]

    return min(length, kernel_size_max_configured)

def get_max_dilation_rate(layer_input_shape: tuple[int, int, int],
                          kernel_size: int,
                          max_dilation_rate_configured: int):
    if kernel_size == 1:
        return 1

    length = layer_input_shape[0]

    max_dilation_rate = (length - 1) // (kernel_size - 1)

    if max_dilation_rate == 0:
        # No dilation
        max_dilation_rate = 1

    return min(max_dilation_rate, max_dilation_rate_configured)

def get_conv_layer_params(layer_input_shape: tuple[int, int, int],
                          trial: optuna.Trial,
                          layer_num: int,
                          conv_layer_config: ConvLayerConfig):
    num_filters = trial.suggest_int(
        f"layer_{layer_num}_num_filters",
        conv_layer_config.num_filter_min,
        conv_layer_config.num_filter_max,
        step=conv_layer_config.num_filters_step
    )

    max_kernel_size = get_max_kernel_size(
        layer_input_shape,
        conv_layer_config.kernel_size_max
    )

    min_kernel_size = min(conv_layer_config.kernel_size_min, max_kernel_size)

    kernel_size = trial.suggest_int(
        f"layer_{layer_num}_kernel_size",
        min_kernel_size,
        max_kernel_size,
        step=2
    )

    max_dilation_rate = get_max_dilation_rate(
        layer_input_shape,
        kernel_size,
        conv_layer_config.dilation_rate_max
    )

    dilation_rate = trial.suggest_int(
        f"layer_{layer_num}_dilation_rate",
        conv_layer_config.dilation_rate_min,
        max_dilation_rate,
        log=True
    )

    return (num_filters, kernel_size, dilation_rate)

def get_max_pooling_size_max(layer_input_shape: tuple[int, int, int],
                             max_pooling_size_max_configured: int):
    length = layer_input_shape[0]

    # Ensure output at least length 5 (so that a conv filter of size 3 can be applied in a useful manner)
    max_val = length // 5

    if max_val == 0:
        # Length < 5 will give output 0, so set to 1 meaning no max pool allowed
        max_val = 1

    return min(max_val, max_pooling_size_max_configured)

def get_max_pooling_params(layer_input_shape: tuple[int, int, int],
                           trial: optuna.Trial,
                           layer_num: int,
                           max_pooling_config: MaxPoolLayerConfig):
    max_pooling_size = get_max_pooling_size_max(
        layer_input_shape,
        max_pooling_config.max_pooling_size_max
    )

    max_pooling_size = trial.suggest_int(f"layer_{layer_num}_max_pooling_size",
                                         max_pooling_config.max_pooling_size_min,
                                         max_pooling_size,
                                         step=2)

    return max_pooling_size

def add_conv_layer(model: tf.keras.Sequential,
                   trial: optuna.Trial,
                   layer_num: int,
                   layer_input_shape: tuple[int, int, int],
                   conv_layer_config: ConvLayerConfig,
                   max_pooling_config: MaxPoolLayerConfig,
                   axis_mixing: bool = False,
                   conv_type: str = "standard"):

    # Max Pool if enabled

    if max_pooling_config.max_pooling_enabled:
        max_pooling_size = get_max_pooling_params(layer_input_shape,
                                                  trial,
                                                  layer_num,
                                                  max_pooling_config)
        if max_pooling_size > 1:
            model.add(layers.MaxPooling2D(pool_size=(max_pooling_size, 1)))

            layer_input_shape = model.output_shape[1:]

    # BatchNorm

    model.add(layers.BatchNormalization())


    # Conv Layer

    (num_filters, kernel_size, dilation_rate) = (
        get_conv_layer_params(
            layer_input_shape,
            trial,
            layer_num,
            conv_layer_config
        )
    )

    kernel_height = 1
    if axis_mixing:
        kernel_height = 3

    if conv_type == "standard":
        model.add(
            layers.Conv2D(
                num_filters,
                kernel_size=(kernel_size, kernel_height),
                dilation_rate=(dilation_rate, 1),
                padding="valid",
                activation="relu"
            )
        )
    elif conv_type == "separable":
        depth_multiplier = trial.suggest_int(
            f"layer_{layer_num}_depth_multiplier",
            1,
            4
        )

        model.add(
            layers.SeparableConv2D(
                num_filters,
                kernel_size=(kernel_size, kernel_height),
                dilation_rate=(dilation_rate, 1),
                depth_multiplier=depth_multiplier,
                padding="valid",
                activation="relu"
            )
        )
    else:
        raise ValueError(f"Invalid conv_type: {conv_type}")

def get_layer_confs():
    conv_1_conf = ConvLayerConfig(
        kernel_size_min=3, kernel_size_max=3,

        dilation_rate_min=1, dilation_rate_max=2,

        num_filter_min=4, num_filter_max=20, num_filters_step=8,
    )
    max_pooling_1_conf = MaxPoolLayerConfig(
        max_pooling_enabled=False,
    )

    conv_2_conf = ConvLayerConfig(
        kernel_size_min=3, kernel_size_max=5,

        dilation_rate_min=1, dilation_rate_max=4,

        num_filter_min=4, num_filter_max=36, num_filters_step=8,
    )
    max_pooling_2_conf = MaxPoolLayerConfig(
        max_pooling_enabled=True,
        max_pooling_size_min=1, max_pooling_size_max=6
    )

    conv_3_conf = ConvLayerConfig(
        kernel_size_min=3, kernel_size_max=5,

        dilation_rate_min=1, dilation_rate_max=4,

        num_filter_min=8, num_filter_max=88, num_filters_step=8,
    )
    max_pooling_3_conf = MaxPoolLayerConfig(
        max_pooling_enabled=True,
        max_pooling_size_min=1, max_pooling_size_max=6
    )

    conv_4_conf = ConvLayerConfig(
        kernel_size_min=3, kernel_size_max=5,

        dilation_rate_min=1, dilation_rate_max=4,

        num_filter_min=16, num_filter_max=88, num_filters_step=8,
    )
    max_pooling_4_conf = MaxPoolLayerConfig(
        max_pooling_enabled=True,
        max_pooling_size_min=1, max_pooling_size_max=6
    )

    conv_5_conf = ConvLayerConfig(
        kernel_size_min=3, kernel_size_max=5,

        dilation_rate_min=1, dilation_rate_max=4,

        num_filter_min=32, num_filter_max=88, num_filters_step=8,
    )
    max_pooling_5_conf = MaxPoolLayerConfig(
        max_pooling_enabled=True,
        max_pooling_size_min=1, max_pooling_size_max=4
    )

    conv_layer_confs = [conv_1_conf, conv_2_conf,
                        conv_3_conf, conv_4_conf, conv_5_conf]

    max_pooling_layer_confs = [max_pooling_1_conf,
                               max_pooling_2_conf, max_pooling_3_conf,
                               max_pooling_4_conf, max_pooling_5_conf]

    return conv_layer_confs, max_pooling_layer_confs

def get_f1_macro_metric():
    return tf.keras.metrics.F1Score(
            average="macro",
            name="f1_macro"
        )

def get_ddcnn_model(trial: optuna.Trial,
              input_shape: tuple[int, int, int] = (48, 3, 1),
              num_classes: int = 4):
    model = tf.keras.Sequential()

    # Input block
    model.add(
        layers.Input(shape=input_shape)
    )

    ##
    # Conv Layers
    ##

    conv_layer_confs, max_pooling_layer_confs = get_layer_confs()

    num_conv_layers = trial.suggest_int(
        "num_conv_layers",
        3,
        5
    )

    axis_mixing_layer = trial.suggest_int(
        "axis_mixing_layer",
        2,
        num_conv_layers
    )

    pooling_type = trial.suggest_categorical(
        "pooling_type",
        ["max", "avg"]
    )

    num_dense_layers = trial.suggest_int(
        "num_hidden_dense_layers",
        0,
        1
    )

    for i in range(num_conv_layers):
        axis_mixing = False
        if i == (axis_mixing_layer - 1):
            axis_mixing = True

        if i == 0:
            layer_input_shape = input_shape
        else:
            layer_input_shape = model.output_shape[1:]

        # First layer is always standard conv since # channels is 1
        if i == 0:
            conv_type = "standard"
        else:
            conv_type = trial.suggest_categorical(
                f"layer_{i}_conv_type",
                ["standard", "separable"])

        add_conv_layer(
            model,
            trial,
            i,
            layer_input_shape,
            conv_layer_confs[i],
            max_pooling_layer_confs[i],
            axis_mixing=axis_mixing,
            conv_type=conv_type
        )


    if pooling_type == "max":
        model.add(layers.GlobalMaxPooling2D())
    elif pooling_type == "avg":
        model.add(layers.GlobalAveragePooling2D())
    else:
        raise ValueError(f"Invalid pooling type: {pooling_type}")

    for i in range(num_dense_layers):
        model.add(
            layers.Dense(
                units=trial.suggest_int(
                    f"layer_{i}_dense_units",
                    8,
                    12,
                    step=2
                ),
                activation='relu'
            )
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
