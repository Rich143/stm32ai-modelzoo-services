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

def run_tuner(configs: DictConfig) -> None:
    dataset = load_and_preprocess_dataset(cfg=configs)

    train_subjects, cv_subjects, test_subjects, excluded_subjects = (
        get_cv_subjects(experiment_cv_config=configs.dataset.train_val_test_cv_split,
                        dataset=dataset))

    seed = configs.dataset.seed

    (datasets, subjects_in_folds) = kfold_train_val_test(dataset=dataset,
                                    test_subjects=test_subjects,
                                    always_train_subjects=train_subjects,
                                    cv_subjects=cv_subjects,
                                    n_splits=2,
                                    random_state=seed,
                                    shuffle=True)

    print_dataset_class_summary(datasets[1][0])
    print_dataset_class_summary(datasets[1][1])
    print_dataset_class_summary(datasets[1][2])

    labels = datasets[1][0]['activity_label'].values

    class_weights = get_class_weights(labels=labels, class_names=configs.dataset.class_names)

    # We just use one split (don't do kfold for tuner)
    train_ds, valid_ds, test_ds, callbacks = (
        segment_presplit_dataset_using_config(train_ds=datasets[0][0],
                                              val_ds=datasets[0][1],
                                              test_ds=datasets[0][2],
                                              cfg=configs))

    train_tuner(cfg=configs,
                      train_ds=train_ds,
                      valid_ds=valid_ds,
                      test_ds=test_ds,
                      callbacks=callbacks,
                      class_weights=class_weights)
