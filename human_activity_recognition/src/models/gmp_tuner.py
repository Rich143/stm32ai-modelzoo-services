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
from keras_tuner_model_utils import check_model_compute_budget
import keras_tuner as kt

def create_build_model(input_shape: tuple[int] = (24, 3, 1),
                       num_classes: int = 4,
                       dropout: float = 0.5,
                       num_conv_layers_min: int = 1,
                       num_conv_layers_max: int = 5,
                       max_maccs: int = 0,
                       max_num_params: int = 0):
    def build_model(hp):
        num_conv_layers = hp.Int('num_conv_layers',
                                 min_value=num_conv_layers_min,
                                 max_value=num_conv_layers_max, step=1)

        model = get_gmp(input_shape=input_shape,
                        num_classes=num_classes, dropout=dropout,
                        num_conv_layers=num_conv_layers)

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

def get_gmp(input_shape: tuple[int] = (24, 3, 1),
            num_classes: int = 4,
            dropout: float = 0.5,
            num_conv_layers: int = 2):
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

    model = tf.keras.Sequential()

    # Input block
    model.add(
        layers.Input(shape=input_shape)
    )

    # Conv layers
    # for _ in range(num_conv_layers):
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(16, kernel_size=(5, 1),
                            strides=(1, 1),
                            kernel_initializer='glorot_uniform',
                            padding="valid",
                            activation="relu"))

    model.add(layers.MaxPooling2D(pool_size=(2, 1)))

    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(16, kernel_size=(5, 3),
                            strides=(1, 1),
                            kernel_initializer='glorot_uniform',
                            padding="valid",
                            activation="relu"))
    # maxpooling
    model.add(
        layers.GlobalMaxPooling2D()
    )

    # front
    if dropout:
        model.add(
            layers.Dropout(dropout)
        )

    model.add(
        layers.Dense(num_classes)
    )

    model.add(
        layers.Activation('softmax')
    )

    pr_auc_metric = get_pr_auc_metric()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', pr_auc_metric])

    return model
