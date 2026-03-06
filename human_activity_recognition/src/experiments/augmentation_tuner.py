import mlflow
import numpy as np
import tensorflow as tf
from datetime import datetime
import uuid

from omegaconf import DictConfig
import pandas as pd
from typing import Dict, List, Optional
from sklearn.utils.class_weight import compute_class_weight
import optuna, optunahub

from preprocessing.preprocess import (load_and_preprocess_dataset,
                        segment_presplit_dataset_using_config_augmentation_tuning)
from experiments.experiment_utils import (kfold_train_val_test,
                                          get_cv_subjects)
from models.richard_v1 import get_richard_v1
from hydra.core.hydra_config import HydraConfig
from training.train_utils import get_early_stopping_cb, check_tuner_cfg
from experiments.tuner_utils import (get_class_weights, print_dataset_class_summary)
from preprocessing.data_augmentation import (NoiseConfig, AmplitudeScaleConfig, RotationConfig,
                               AugmentationConfig)
from models.keras_tuner_model_utils import get_model_maccs, get_model_num_params
from common.utils.cfg_utils import get_random_seed

def run_augmentation_tuner(configs: DictConfig) -> None:
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
    train_ds, valid_ds, test_ds = (datasets[0][0],
                                   datasets[0][1],
                                   datasets[0][2])

    train_augmentation_tuner(cfg=configs,
                      train_ds=train_ds,
                      valid_ds=valid_ds,
                      test_ds=test_ds,
                      class_weights=class_weights)

def get_tuned_augmentation_config(trial, seed) -> AugmentationConfig:
    noise_enabled = trial.suggest_categorical("noise_enabled", [True, False])
    if noise_enabled:
        noise_std = trial.suggest_float("noise_std", 0.0, 0.3)
        noise_config = NoiseConfig(noise_std=noise_std)
    else:
        noise_config = None

    amplitude_scaling_enabled = trial.suggest_categorical("amplitude_scaling_enabled",
                                                          [True, False])
    if amplitude_scaling_enabled:
        amplitude_scaling_min = trial.suggest_float("amplitude_scaling_min", 0.5, 1.0)
        amplitude_scaling_max = trial.suggest_float("amplitude_scaling_max", 1.0, 1.5)

        amplitude_scaling_config = AmplitudeScaleConfig(
            amplitude_scaling_min=amplitude_scaling_min,
            amplitude_scaling_max=amplitude_scaling_max
        )
    else:
        amplitude_scaling_config = None

    rotation_enabled = trial.suggest_categorical("rotation_enabled", [True, False])

    if rotation_enabled:
        rotation_max_roll_deg = trial.suggest_float("rotation_max_roll_deg", 0.0, 5.0)
        rotation_max_pitch_deg = trial.suggest_float("rotation_max_pitch_deg", 0.0, 5.0)
        rotation_max_yaw_deg = trial.suggest_float("rotation_max_yaw_deg", 0.0, 15.0)

        rotation_config = RotationConfig(
            rotation_max_roll_deg=rotation_max_roll_deg,
            rotation_max_pitch_deg=rotation_max_pitch_deg,
            rotation_max_yaw_deg=rotation_max_yaw_deg
        )
    else:
        rotation_config = None

    augmentation_config = AugmentationConfig(
        seed=seed,
        noise_cfg=noise_config,
        amplitude_scale_cfg=amplitude_scaling_config,
        rotation_cfg=rotation_config
    )

    return augmentation_config

def get_tuned_datasets(trial,
                       cfg: DictConfig,
                       train_ds: pd.DataFrame,
                       valid_ds: pd.DataFrame,
                       test_ds: pd.DataFrame,
                       seed):
    augmentation_config = get_tuned_augmentation_config(trial, seed)

    train_ds_tf, valid_ds_tf, test_ds_tf, callbacks = (
        segment_presplit_dataset_using_config_augmentation_tuning(
            train_ds=train_ds,
            val_ds=valid_ds,
            test_ds=test_ds,
            cfg=cfg,
            augmentation_config=augmentation_config
        )
    )

    return train_ds_tf, valid_ds_tf, test_ds_tf, callbacks

def get_tuner_objective(train_ds: pd.DataFrame,
                        valid_ds: pd.DataFrame,
                        test_ds: pd.DataFrame,
                        reusable_callbacks: Optional[List[tf.keras.callbacks.Callback]],
                        class_weights: Dict[int, float],
                        num_classes: int,
                        cfg,
                        ):
    base_train_ds, base_valid_ds, base_test_ds = train_ds, valid_ds, test_ds

    def objective(trial):
        # Clear clutter from previous TensorFlow graphs.
        tf.keras.backend.clear_session()

        callbacks = []

        # Create a new early stopping CB for each model, this fixes bug
        # with early stopping callback being reused
        early_stop_cb = get_early_stopping_cb(cfg)

        if early_stop_cb is not None:
            print("[INFO] Adding early stopping callback")
            callbacks.append(early_stop_cb)

        if reusable_callbacks is not None:
            callbacks.extend(reusable_callbacks)

        tuned_train_ds, tuned_valid_ds, tuned_test_ds, ds_callbacks = get_tuned_datasets(
            trial, cfg,
            base_train_ds, base_valid_ds, base_test_ds,
            cfg.dataset.seed
        )

        if ds_callbacks is not None:
            callbacks.extend(ds_callbacks)

        epochs = cfg.training.epochs
        steps_per_epoch = cfg.training.steps_per_epoch
        input_shape = cfg.training.model.input_shape

        model = get_richard_v1(
            input_shape=input_shape,
            num_classes=num_classes,
        )

        model.summary()

        monitor = "val_f1_macro"

        # Train model.
        history = model.fit(tuned_train_ds,
                            validation_data=tuned_valid_ds,
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            callbacks=callbacks,
                            class_weight=class_weights,
                            verbose=1)

        if early_stop_cb is not None:
            print("Early stopping metrics:")
            # Epoch with best monitored metric
            best_epoch = early_stop_cb.best_epoch  # 0-indexed
            print("Best epoch:", best_epoch)
            print("Stopped epoch:", early_stop_cb.stopped_epoch)
            print("Best value:", early_stop_cb.best)
            print("Wait counter:", early_stop_cb.wait)


        f1_val_best = max(history.history[monitor])

        # Check for error with F1 score (saw this happening once)
        if f1_val_best > 1.0:
            # Failed trial
            return None

        return f1_val_best

    return objective

def train_augmentation_tuner(
    cfg: DictConfig,
    train_ds: pd.DataFrame,
    valid_ds: pd.DataFrame,
    test_ds: pd.DataFrame,
    class_weights: Dict[int, float],
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
) -> str:
    """
    Runs Tuner using the provided configuration and datasets.

    Args:
        cfg (DictConfig): The entire configuration file dictionary.
        train_ds (pd.DataFrame): training dataset
        valid_ds (pd.DataFrame): validation dataset
        test_ds (Optional, pd.DataFrame): Optional test dataset
        callbacks (Optional, List[tf.keras.callbacks.Callback]): list of callbacks.
        run_name (str): name of the run

    Returns:
        Path to the best model obtained
    """
    check_tuner_cfg(cfg)

    output_dir = HydraConfig.get().runtime.output_dir

    class_names = cfg.dataset.class_names
    num_classes = len(class_names)

    module = optunahub.load_module(package="samplers/auto_sampler")

    sampler = module.AutoSampler(
        seed=get_random_seed(cfg)
    )

    study_name = f"{cfg.tuner.experiment_name}"
    storage_name = f"sqlite:///{output_dir}/{study_name}.db"

    study = optuna.create_study(
        directions=["maximize"],
        sampler=sampler,
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )

    objective = get_tuner_objective(
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        reusable_callbacks=callbacks,
        class_weights=class_weights,
        num_classes=num_classes,
        cfg=cfg
    )

    try:
        study.optimize(objective, n_trials=cfg.tuner.max_trials)
    except KeyboardInterrupt:
        print("Study interrupted by user. You can safely resume later.")
        print(f"Study is saved to {storage_name}")
