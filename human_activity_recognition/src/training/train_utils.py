import tensorflow as tf
from omegaconf import DictConfig
from typing import Tuple, Dict, List
import pandas as pd

from preprocessing.preprocess import (load_and_preprocess_dataset,
                                      segment_presplit_dataset_using_config)
from experiments.tuner_utils import (get_class_weights, print_dataset_class_summary)
from experiments.experiment_utils import (kfold_train_val_test,
                                          get_cv_subjects)

def check_tuner_cfg(cfg):
    tuner_cfg = cfg.tuner
    if tuner_cfg is None:
        raise ValueError("\nNo Tuner CFG found. Please check the 'tuner' section of your configuration file.")

    if tuner_cfg.max_trials is None:
        raise ValueError("\nNo max_trials found. Please check the 'tuner.max_trials' section of your configuration file.")

    if tuner_cfg.executions_per_trial is None:
        raise ValueError("\nNo executions_per_trial found. Please check the 'tuner.executions_per_trial' section of your configuration file.")

def get_early_stopping_cb(cfg: DictConfig):
    if cfg.training.EarlyStopping is not None:
        return tf.keras.callbacks.EarlyStopping(
            monitor=cfg.training.EarlyStopping.monitor,
            mode=cfg.training.EarlyStopping.mode,
            patience=cfg.training.EarlyStopping.patience,
            restore_best_weights=cfg.training.EarlyStopping.restore_best_weights
        )

def get_split_datasets(cfg: DictConfig) -> Tuple[pd.DataFrame,
                                                 pd.DataFrame,
                                                 pd.DataFrame, Dict[int, float]]:
    dataset = load_and_preprocess_dataset(cfg=cfg)

    train_subjects, cv_subjects, test_subjects, excluded_subjects = (
        get_cv_subjects(experiment_cv_config=cfg.dataset.train_val_test_cv_split,
                        dataset=dataset))

    seed = cfg.dataset.seed

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

    class_weights = get_class_weights(labels=labels, class_names=cfg.dataset.class_names)

    # We just use one split (don't do kfold for tuner)
    train_ds, valid_ds, test_ds = (datasets[0][0],
                                   datasets[0][1],
                                   datasets[0][2])

    return train_ds, valid_ds, test_ds, class_weights

def segment_datasets(
    cfg: DictConfig,
    train_ds: pd.DataFrame,
    valid_ds: pd.DataFrame,
    test_ds: pd.DataFrame
) -> Tuple[pd.DataFrame,
           pd.DataFrame,
           pd.DataFrame, List[tf.keras.callbacks.Callback]]:
    # We just use one split (don't do kfold for tuner)
    train_ds, valid_ds, test_ds, callbacks = (
        segment_presplit_dataset_using_config(train_ds=train_ds,
                                              val_ds=valid_ds,
                                              test_ds=test_ds,
                                              cfg=cfg)
    )

    return train_ds, valid_ds, test_ds, callbacks
