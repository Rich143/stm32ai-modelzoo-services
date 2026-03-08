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

from omegaconf import DictConfig

from common.training.common_training import get_optimizer

def get_f1_macro_metric():
    return tf.keras.metrics.F1Score(
            average="macro",
            name="f1_macro"
        )

def get_richard_v1(input_shape: tuple[int] = (48, 3, 1),
                   num_classes: int = 4,
                   optimizer_cfg: DictConfig = None):
    """
    Builds and returns an ign model for human_activity_recognition.
    Args:
        input_shape (tuple): A dictionary containing the configuration for the model.
        num_classes (int): number of nodes in the output layer
        dropout (float): dropout ratio to be used for dropout layer
    Returns:
        - keras.Model object, the ign model.
    """
    # function to build the CNN model

    # Input block
    inputs = layers.Input(shape=input_shape)

    # First block
    x = layers.Conv2D(
                12,
                kernel_size=(3, 1),
                padding="valid",
                activation="relu")(inputs)

    # Second block
    x = layers.SeparableConv2D(
                20,
                kernel_size=(3, 3),
                padding="valid",
                activation="relu")(x)

    # Third block
    x = layers.SeparableConv2D(
                32,
                kernel_size=(5, 1),
                padding="valid",
                activation="relu")(x)


    # Global max pooling
    x = layers.GlobalMaxPooling2D()(x)

    # last fully connected layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[outputs], name="richard_v1")

    f1_metric = get_f1_macro_metric()
    optimizer = get_optimizer(optimizer_cfg)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy', f1_metric])

    return model
