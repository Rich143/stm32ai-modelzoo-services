import mlflow
import numpy as np
import tensorflow as tf
from datetime import datetime
import uuid

from omegaconf import DictConfig
from preprocess import (load_and_preprocess_dataset,
                        segment_presplit_dataset_using_config,
                        one_hot_encoding_from_activity_ids)
from train import train_keras_tuner
from experiments.experiment_utils import (kfold_train_val_test,
                                          get_cv_subjects)
from sklearn.utils.class_weight import compute_class_weight

from data_load_helpers import _GLOBAL_ACTIVITY_ID_TO_NAME
import pandas as pd
from typing import Dict, List

def print_dataset_class_summary(dataset: pd.DataFrame) -> None:
    counts = dataset["activity_label"].value_counts().sort_index()
    percentages = counts / counts.sum() * 100

    distribution = pd.DataFrame({
        "count": counts,
        "percent": percentages.round(2)
    })

    distribution.index = distribution.index.map(_GLOBAL_ACTIVITY_ID_TO_NAME)
    print(distribution)

def get_class_weights(labels: np.ndarray, class_names: List[str]) -> Dict[int, float]:
    labels_one_hot = one_hot_encoding_from_activity_ids(
        activity_ids=labels,
        class_names=class_names)

    labels_idxs = np.argmax(labels_one_hot, axis=1)
    classes_unique = np.unique(labels_idxs)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes_unique,
        y=labels_idxs
    )

    class_weights = dict(zip(classes_unique, weights))

    print("\n[INFO] : Class weights:")
    for key, value in class_weights.items():
        print(f"{key}: {value}")
    print("\n")

    num_classes = len(class_names)
    assert set(class_weights.keys()) == set(range(num_classes)), \
    "Class-weight keys must be 0-based and contiguous"

    return class_weights

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

    print_dataset_class_summary(datasets[1][0])
    print_dataset_class_summary(datasets[1][1])
    print_dataset_class_summary(datasets[1][2])

    labels = datasets[1][0]['activity_label'].values

    class_weights = get_class_weights(labels=labels, class_names=configs.dataset.class_names)

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
                      callbacks=callbacks,
                      class_weights=class_weights)
