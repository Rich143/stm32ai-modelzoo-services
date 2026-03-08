import mlflow
import numpy as np
import tensorflow as tf
from datetime import datetime
import uuid

from omegaconf import DictConfig
from preprocessing.preprocess import (load_and_preprocess_dataset,
                        segment_presplit_dataset_using_config,
                        one_hot_encoding_from_activity_ids)
from training.train import train_tuner
from experiments.experiment_utils import (kfold_train_val_test,
                                          get_cv_subjects)

from preprocessing.data_load_helpers import _GLOBAL_ACTIVITY_ID_TO_NAME
import pandas as pd
from typing import Dict, List
from experiments.tuner_utils import (get_class_weights, print_dataset_class_summary)
from training.train_utils import get_split_datasets, segment_datasets

def run_tuner(configs: DictConfig) -> None:

    train_ds, valid_ds, test_ds, class_weights = get_split_datasets(cfg=configs)
    train_ds, valid_ds, test_ds, callbacks = segment_datasets(train_ds=train_ds,
                                                              valid_ds=valid_ds,
                                                              test_ds=test_ds,
                                                              cfg=configs)
    train_tuner(cfg=configs,
                      train_ds=train_ds,
                      valid_ds=valid_ds,
                      test_ds=test_ds,
                      callbacks=callbacks,
                      class_weights=class_weights)
