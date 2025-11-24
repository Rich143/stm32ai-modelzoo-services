# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from omegaconf import DictConfig
from typing import Tuple, TypeAlias, Any
import sys
import os

from models_utils import get_model_name_and_its_input_shape
from data_loader import load_dataset, load_and_filter_dataset, segment_dataset
from data_loader import segment_presplit_dataset, CallbackList
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_and_filter_dataset_from_config(cfg: DictConfig = None) -> Tuple:
    dataset = load_and_filter_dataset(dataset_name=cfg.dataset.name,
                                      dataset_path=cfg.dataset.training_path,
                                      class_names=cfg.dataset.class_names,
                                      gravity_rot_sup=cfg.preprocessing.gravity_rot_sup,
                                      seed=cfg.dataset.seed)

    return dataset


ds: TypeAlias = tf.data.Dataset[Any]

def segment_presplit_dataset_using_config(train_ds: pd.DataFrame,
                                          val_ds: pd.DataFrame,
                                          test_ds: pd.DataFrame,
                                          cfg: DictConfig,
                                          to_cache: bool = False,
) -> Tuple[ds, ds, ds, CallbackList]:
    return segment_presplit_dataset(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        class_names=cfg.dataset.class_names,
        input_shape=cfg.training.model.input_shape,
        seed=cfg.dataset.seed,
        batch_size=cfg.training.batch_size,
        gaussian_noise=cfg.preprocessing.gaussian_noise,
        gaussian_std=cfg.preprocessing.gaussian_std,
        to_cache=to_cache
    )


def segment_dataset_from_config(dataset: pd.DataFrame,
                                test_dataset: pd.DataFrame | None = None,
                                cfg: DictConfig = None) -> Tuple:
    # Get the model input shape
    if cfg.general.model_path:
        _, input_shape = get_model_name_and_its_input_shape(cfg.general.model_path)
    else:
        # We are running a training using the 'training' section of the config file.
        if cfg.training.model:
            input_shape = cfg.training.model.input_shape
        else:
            raise ValueError('Either `cfg.general.model_path` or `cfg.model` information should be provided.\n'
                             'Check your configuration file.')

    batch_size = cfg.training.batch_size
    train_ds, valid_ds, test_ds = segment_dataset(dataset_name=cfg.dataset.name,
                                                  dataset=dataset,
                                                  test_dataset=test_dataset,
                                                  validation_split=cfg.dataset.validation_split,
                                                  test_split=cfg.dataset.test_split,
                                                  class_names=cfg.dataset.class_names,
                                                  input_shape=input_shape[:2],
                                                  batch_size=batch_size,
                                                  gaussian_noise=cfg.preprocessing.gaussian_noise,
                                                  gaussian_std=cfg.preprocessing.gaussian_std,
                                                  seed=cfg.dataset.seed)

    return train_ds, valid_ds, test_ds

def preprocess(cfg: DictConfig = None) -> Tuple:
    """
    Preprocesses the data based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration object containing the settings.

    Returns:
        Tuple: A tuple containing the following:
            - data_augmentation (object): Data augmentation object.
            - augment (bool): Flag indicating whether data augmentation is enabled.
            - pre_process (object): Preprocessing object.
            - train_ds (object): Training dataset.
            - valid_ds (object): Validation dataset.
    """

    # Get the model input shape
    if cfg.general.model_path:
        _, input_shape = get_model_name_and_its_input_shape(cfg.general.model_path)
    else:
        # We are running a training using the 'training' section of the config file.
        if cfg.training.model:
            input_shape = cfg.training.model.input_shape
        else:
            raise ValueError('Either `cfg.general.model_path` or `cfg.model` information should be provided.\n'
                             'Check your configuration file.')

    batch_size = cfg.training.batch_size
    train_ds, valid_ds, test_ds = load_dataset(
                                                dataset_name=cfg.dataset.name,
                                                training_path=cfg.dataset.training_path,
                                                validation_path=cfg.dataset.validation_path,
                                                test_path=cfg.dataset.test_path,
                                                test_split=cfg.dataset.test_split,
                                                validation_split=cfg.dataset.validation_split,
                                                class_names=cfg.dataset.class_names,
                                                input_shape=input_shape[:2],
                                                gravity_rot_sup=cfg.preprocessing.gravity_rot_sup,
                                                batch_size=batch_size,
                                                seed=cfg.dataset.seed)

    return train_ds, valid_ds, test_ds
