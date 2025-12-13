# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from omegaconf import DictConfig
from typing import Tuple, TypeAlias, Any, List
from data_loader import load_datasets, preprocess_dataset
from data_load_helpers import global_activity_id_to_name

import pandas as pd
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import statistics
from tensorflow.keras.utils import to_categorical
import mlflow

CallbackList: TypeAlias = List[tf.keras.callbacks.Callback]
ds: TypeAlias = tf.data.Dataset[Any]

def load_and_preprocess_dataset(cfg: DictConfig) -> pd.DataFrame:
    dataset = load_dataset_from_config(cfg=cfg)

    dataset = preprocess_dataset_from_config(dataset=dataset, cfg=cfg)

    return dataset

def load_dataset_from_config(cfg: DictConfig) -> pd.DataFrame:
    dataset = load_datasets(dataset_names=cfg.dataset.names,
                            dataset_paths=cfg.dataset.paths)

    return dataset

def preprocess_dataset_from_config(dataset: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    return preprocess_dataset(dataset=dataset,
                       class_names=cfg.dataset.class_names,
                       fs=cfg.preprocessing.sample_rate,
                       mean_group_delay=cfg.preprocessing.mean_group_delay,
                       gravity_rot_sup=cfg.preprocessing.gravity_rot_sup)


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

class NoiseConfig:
    def __init__(self, noise_std: float, seed: int):
        # holds the current epoch number
        self.epoch = tf.Variable(0, dtype=tf.int64, trainable=False)

        self.noise_std = noise_std
        self.seed = seed

class UpdateEpoch(tf.keras.callbacks.Callback):
    def __init__(self, cfg: NoiseConfig):
        super().__init__()
        self.cfg = cfg

    def on_epoch_begin(self, epoch, logs=None):
        # assign epoch to the tf.Variable
        self.cfg.epoch.assign(epoch)

def make_stateless_noise_fn(cfg: NoiseConfig):
    def add_noise(batch_idx, batch):
        x, y = batch

        base_seed = tf.constant([cfg.seed, 0], dtype=tf.int32)

        # seed = fold(global_seed, epoch)
        seed1 = tf.random.experimental.stateless_fold_in(base_seed, cfg.epoch)

        # seed = fold(seed1, batch_idx)
        final_seed = tf.random.experimental.stateless_fold_in(seed1, batch_idx)

        noise = tf.random.stateless_normal(
            shape=tf.shape(x),
            seed=final_seed,
            stddev=cfg.noise_std,
            dtype=x.dtype,
        )
        return x + noise, y

    return add_noise

def get_segment_indices(data_column: List, win_len: int):
    '''
    this function gets the start and end indices for the segments
    args:
        data_column: indices
        win_len: segment length
    yields:
        init: int
        end: int
    '''
    # get segment indices to window the data into overlapping frames
    init = 0
    while init < len(data_column):
        yield int(init), int(init + win_len)
        init = init + win_len


def get_data_segments(dataset: pd.DataFrame,
                      seq_len: int) -> Tuple[np.ndarray, np.ndarray]:

    '''
    This function segments the data into (non)overlaping frames
    args:
        dataset: a dataframe containing 'x', 'y', 'z', and 'activity_label' columns
    returns:
        A Tuple of np.ndarray containing segments and labels
    '''
    data_indices = dataset.index.tolist()
    n_samples = len(data_indices)
    segments = []
    labels = []

    # need the following variable for tqdm to show the progress bar
    num_segments = int(np.floor((n_samples - seq_len) / seq_len)) + 1

    # creating segments until the get_segment_indices keep on yielding the start and end of the segments
    for (init, end) in tqdm(get_segment_indices(data_indices, seq_len),
                            unit=' segments', desc='Segments built ',
                            total=num_segments):

        # check if the nr of remaing samples are enough to create a frame
        if end < n_samples:
            segments.append(np.transpose([dataset['x'].values[init: end],
                            dataset['y'].values[init: end],
                            dataset['z'].values[init: end]]))

            # use the label which occured the most in the frame
            # print(labels, statistics.mode(dataset['activity_label'][init: end]))
            labels.append(statistics.mode(dataset['activity_label'][init: end]))

    # converting the segments from list to numpy array
    segments = np.asarray(segments, dtype=float)
    segments = segments.reshape(segments.shape[0], segments.shape[1],
                                segments.shape[2], 1)
    labels = np.asarray(labels)
    return segments, labels

def segment_and_get_labels(dataset: pd.DataFrame,
                           class_names: List[str],
                           seq_len: int):
    # removing uneeded columns and rearranging remaining columns
    dataset = dataset[['x', 'y', 'z', 'activity_label']]

    segments, labels = get_data_segments(dataset=dataset,
                                         seq_len=seq_len)

    # one-hot encode labels, convert from id to name first so that the order is
    # defined based on the config file
    labels = to_categorical([class_names.index(global_activity_id_to_name(label))
                            for label in labels], num_classes=len(class_names))

    return segments, labels

def build_train_ds(train_x: np.ndarray,
                   train_y: np.ndarray,
                   batch_size: int,
                   seed: int,
                   gaussian_noise: bool = False,
                   gaussian_std: float = 0,
                   to_cache: bool = False
                  ) -> Tuple[
                      tf.data.Dataset,
                      CallbackList
                  ]:
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))

    if to_cache:
        train_ds = train_ds.cache()

    train_ds = (train_ds.repeat()
                .batch(batch_size))
    # train_ds = (train_ds.shuffle(train_x.shape[0],
                                 # reshuffle_each_iteration=True,
                                 # seed=seed)
                # .repeat()
                # .batch(batch_size))

    callbacks = []

    # if gaussian_noise:
        # noise_cfg = NoiseConfig(gaussian_std, seed)

        # callbacks.append(UpdateEpoch(noise_cfg))

        # # Adds in a batch idx to use with the stateless random number generator
        # train_ds = train_ds.enumerate()
        # train_ds = train_ds.map(make_stateless_noise_fn(noise_cfg),
                                # num_parallel_calls=tf.data.AUTOTUNE)
        # mlflow.log_params({"gaussian_noise": True})
        # mlflow.log_params({"gaussian_std": gaussian_std})
    # else:
        # mlflow.log_params({"gaussian_noise": False})
        # mlflow.log_params({"gaussian_std": 0})

    # train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, callbacks

def segment_presplit_dataset(train_ds: pd.DataFrame,
                             val_ds: pd.DataFrame,
                             test_ds: pd.DataFrame,
                             class_names: List[str],
                             input_shape: Tuple,
                             seed: int,
                             batch_size: int,
                             gaussian_noise: bool = False,
                             gaussian_std: float = 0,
                             to_cache: bool = False
                            ) -> Tuple[
                                tf.data.Dataset,
                                tf.data.Dataset,
                                tf.data.Dataset,
                                CallbackList,
                            ]:
    train_segments, train_labels = segment_and_get_labels(train_ds,
                                                          class_names,
                                                          input_shape[0])
    val_segments, val_labels = segment_and_get_labels(val_ds,
                                                      class_names,
                                                      input_shape[0])
    test_segments, test_labels = segment_and_get_labels(test_ds,
                                                        class_names,
                                                        input_shape[0])

    train_x, train_y = train_segments, train_labels
    valid_x, valid_y = val_segments, val_labels
    test_x, test_y = test_segments, test_labels

    print("Dataset stats:")
    train_size = train_x.shape[0]
    valid_size = valid_x.shape[0]
    test_size = test_x.shape[0]

    print(f"Train size: {train_size}")
    print(f"Valid size: {valid_size}")
    print(f"Test size: {test_size}")
    print(f"Classes: {len(class_names)}")

    if batch_size is None:
        batch_size=32

    mlflow.log_param("batch_size_actual", batch_size)
    train_ds, callbacks = build_train_ds(train_x=train_x,
                              train_y=train_y,
                              batch_size=batch_size,
                              seed=seed,
                              gaussian_noise=gaussian_noise,
                              gaussian_std=gaussian_std,
                              to_cache=to_cache)

    valid_ds = tf.data.Dataset.from_tensor_slices((valid_x, valid_y))
    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))

    if to_cache:
        valid_ds = valid_ds.cache()
        test_ds = test_ds.cache()

    valid_ds = valid_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds, test_ds, callbacks
