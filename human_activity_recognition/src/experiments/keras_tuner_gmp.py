import mlflow
import numpy as np
import tensorflow as tf
from datetime import datetime
import uuid

from omegaconf import DictConfig
from preprocess import load_and_preprocess_dataset, segment_presplit_dataset_using_config
from train import train_keras_tuner
from experiments.experiment_utils import (kfold_train_val_test,
                                          get_cv_subjects)

def run_gmp_tuner(configs: DictConfig) -> None:
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

    # We just use one split (don't do kfold for keras tuner)
    train_ds, valid_ds, test_ds, callbacks = (
        segment_presplit_dataset_using_config(train_ds=datasets[0][0],
                                              val_ds=datasets[0][1],
                                              test_ds=datasets[0][2],
                                              cfg=configs))

    train_keras_tuner(cfg=configs,
                      train_ds=train_ds,
                      valid_ds=valid_ds,
                      test_ds=test_ds,
                      callbacks=callbacks)
