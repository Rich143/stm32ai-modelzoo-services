import mlflow
import numpy as np
import tensorflow as tf
from datetime import datetime
import uuid

from omegaconf import DictConfig
from preprocess import load_and_preprocess_dataset, segment_presplit_dataset_using_config
from train import train
from experiments.experiment_utils import (kfold_train_val_test, start_child_run,
                                          DatasetTriplet, SubjectListTriplet,
                                          get_cv_subjects)

def set_mlflow_tags(train_subjects, cv_subjects, test_subjects, excluded_subjects,
                   noise_sweep_start, noise_sweep_end, noise_sweep_step, num_runs_per_step):
    mlflow.set_tags({
        "train_subjects": train_subjects,
        "cv_subjects": cv_subjects,
        "test_subjects": test_subjects,
        "excluded_subjects": excluded_subjects,
        "noise_sweep_start": noise_sweep_start,
        "noise_sweep_end": noise_sweep_end,
        "noise_sweep_step": noise_sweep_step,
        "num_runs_per_step": num_runs_per_step,
        "kfold": True
    })

def gaussian_noise_experiment(configs: DictConfig = None) -> None:
    noise_sweep_start, noise_sweep_end, noise_sweep_step, num_runs_per_step = (
        get_experiment_params(cfg=configs))

    dataset = load_and_preprocess_dataset(cfg=configs)

    train_subjects, cv_subjects, test_subjects, excluded_subjects = (
        get_cv_subjects(experiment_cv_config=configs.dataset.train_val_test_cv_split,
                        dataset=dataset))

    seed = configs.dataset.seed
    input_shape = configs.training.model.input_shape

    mlflow.log_params({"seed": configs.dataset.seed})
    mlflow.log_params({"input_shape": input_shape})
    mlflow.log_params({"input_length": input_shape[0]})


    set_mlflow_tags(train_subjects=train_subjects,
                    cv_subjects=cv_subjects,
                    test_subjects=test_subjects,
                    excluded_subjects=excluded_subjects,
                    noise_sweep_start=noise_sweep_start,
                    noise_sweep_end=noise_sweep_end,
                    noise_sweep_step=noise_sweep_step,
                    num_runs_per_step=num_runs_per_step)

    noise_values = np.arange(noise_sweep_start, noise_sweep_end, noise_sweep_step)

    (datasets, subjects_in_folds) = kfold_train_val_test(dataset=dataset,
                                    test_subjects=test_subjects,
                                    always_train_subjects=train_subjects,
                                    cv_subjects=cv_subjects,
                                    n_splits=num_runs_per_step,
                                    random_state=seed,
                                    shuffle=True)


    for noise_std in noise_values:
        with start_child_run(run_name=f"Noise_STD_{noise_std}",
                             inherit_params=True,
                             inherit_tags=True):
            child_run(noise_std=noise_std,
                      num_runs_per_step=num_runs_per_step,
                      configs=configs,
                      datasets=datasets,
                      subjects_in_folds=subjects_in_folds)

def log_gaussian_noise_mlflow(cfg):
    mlflow.log_param("gaussian_noise", cfg.preprocessing.gaussian_noise)
    mlflow.log_param("gaussian_std", cfg.preprocessing.gaussian_std)

def get_experiment_params(cfg: DictConfig):
    if cfg.experiment.experiment_params is None:
        raise ValueError("experiment_params is None")

    params_config = cfg.experiment.experiment_params

    if params_config.noise_sweep_start is None:
        raise ValueError("noise_sweep_start is None")
    if params_config.noise_sweep_end is None:
        raise ValueError("noise_sweep_end is None")
    if params_config.noise_sweep_step is None:
        raise ValueError("noise_sweep_step is None")
    if params_config.num_runs_per_step is None:
        raise ValueError("num_runs_per_step is None")

    noise_sweep_start = params_config.noise_sweep_start
    noise_sweep_end = params_config.noise_sweep_end
    noise_sweep_step = params_config.noise_sweep_step

    num_runs_per_step = params_config.num_runs_per_step

    return noise_sweep_start, noise_sweep_end, noise_sweep_step, num_runs_per_step

def child_run(noise_std: float,
              num_runs_per_step: int,
              configs: DictConfig,
              datasets: list[DatasetTriplet],
              subjects_in_folds: list[SubjectListTriplet]) -> None:
    if noise_std > 0:
        configs.preprocessing.gaussian_noise = True
        configs.preprocessing.gaussian_std = noise_std
    else:
        configs.preprocessing.gaussian_noise = False
        configs.preprocessing.gaussian_std = 0

    print("[INFO] : Noise STD: ", configs.preprocessing.gaussian_std)
    log_gaussian_noise_mlflow(cfg=configs)

    for i in range(num_runs_per_step):
        with start_child_run(run_name="Noise_STD_{}_Run_{}".format(noise_std, i),
                             inherit_params=True,
                             inherit_tags=True):
            # Indicate that this run has no more children
            mlflow.set_tag("worker_run", "true")

            subjects_in_fold = subjects_in_folds[i]

            train_ds, valid_ds, test_ds, callbacks = (
                segment_presplit_dataset_using_config(train_ds=datasets[i][0],
                                                      val_ds=datasets[i][1],
                                                      test_ds=datasets[i][2],
                                                      cfg=configs))

            print(f"[INFO] : Run number {i} for noise std {noise_std}")
            print(f"[INFO] : Train subjects: {subjects_in_fold[0]}")
            print(f"[INFO] : Validation subjects: {subjects_in_fold[1]}")
            print(f"[INFO] : Test subjects: {subjects_in_fold[2]}")

            mlflow.log_params({"train_subjects": subjects_in_fold[0],
                               "validation_subjects": subjects_in_fold[1],
                               "test_subjects": subjects_in_fold[2]})

            train(cfg=configs, train_ds=train_ds, valid_ds=valid_ds,
                  test_ds=test_ds, run_name=f"Noise_STD_{noise_std}_Run_{i}",
                  callbacks=callbacks)
