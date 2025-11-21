import mlflow
import tensorflow as tf
from datetime import datetime
import uuid

from omegaconf import DictConfig
from preprocess import load_and_filter_dataset_from_config, segment_presplit_dataset_using_config
from train import train
from experiments.experiment_utils import (kfold_train_val_test, start_child_run,
                                          DatasetTriplet, SubjectListTriplet)

def input_len_experiment(configs: DictConfig = None) -> None:
    input_len_start, input_len_end, input_len_step, num_runs_per_input_len = get_experiment_params(cfg=configs)

    input_lengths = [i for i in range(input_len_start, input_len_end, input_len_step)]

    datasets = []

    dataset = load_and_filter_dataset_from_config(cfg=configs)

    train_subjects = configs.dataset.train_val_test_cv_split.train_subjects
    cv_subjects = configs.dataset.train_val_test_cv_split.cv_subjects
    test_subjects = configs.dataset.train_val_test_cv_split.test_subjects
    excluded_subjects = configs.dataset.train_val_test_cv_split.excluded_subjects

    seed = configs.dataset.seed

    (datasets, subjects_in_folds) = kfold_train_val_test(dataset=dataset,
                                    subject_col="User",
                                    test_subjects=test_subjects,
                                    always_train_subjects=train_subjects,
                                    cv_subjects=cv_subjects,
                                    excluded_subjects=excluded_subjects,
                                    n_splits=num_runs_per_input_len,
                                    random_state=seed,
                                    shuffle=True)

    log_gaussian_noise_mlflow(cfg=configs)

    for i in range(len(input_lengths)):
        input_shape = (input_lengths[i], configs.training.model.input_shape[1], configs.training.model.input_shape[2])

        with start_child_run(run_name=f"Input Shape: {input_shape}",
                             inherit_params=True,
                             inherit_tags=True):
            child_run(input_shape=input_shape,
                      num_runs_per_input_len=num_runs_per_input_len,
                      configs=configs,
                      datasets=datasets,
                      subjects_in_folds=subjects_in_folds)

def log_gaussian_noise_mlflow(cfg):
    if cfg.preprocessing.gaussian_noise:
        mlflow.log_params({"gaussian_noise": cfg.preprocessing.gaussian_noise})
        mlflow.log_params({"gaussian_std": cfg.preprocessing.gaussian_std})
    else:
        mlflow.log_params({"gaussian_noise": False})
        mlflow.log_params({"gaussian_std": 0})

def get_experiment_params(cfg: DictConfig = None):
    if cfg.experiment.experiment_params is None:
        raise ValueError("experiment_params is None")

    params_config = cfg.experiment.experiment_params

    if params_config.input_len_sweep_start is None:
        raise ValueError("input_len_sweep_start is None")
    if params_config.input_len_sweep_end is None:
        raise ValueError("input_len_sweep_end is None")
    if params_config.input_len_sweep_step is None:
        raise ValueError("input_len_sweep_step is None")
    if params_config.num_runs_per_input_len is None:
        raise ValueError("num_runs_per_input_len is None")

    input_len_start = params_config.input_len_sweep_start
    input_len_end = params_config.input_len_sweep_end
    input_len_step = params_config.input_len_sweep_step

    num_runs_per_input_len = params_config.num_runs_per_input_len

    return input_len_start, input_len_end, input_len_step, num_runs_per_input_len

def child_run(input_shape: tuple,
              num_runs_per_input_len: int,
              configs: DictConfig,
              datasets: list[DatasetTriplet],
              subjects_in_folds: list[SubjectListTriplet]) -> None:
    configs.training.model.input_shape = input_shape
    input_len = input_shape[0]

    print("[INFO] : Input shape: ", configs.training.model.input_shape)
    mlflow.log_params({"input_shape": configs.training.model.input_shape})
    mlflow.log_params({"input_length": input_len})

    base_seed = configs.dataset.seed

    for i in range(num_runs_per_input_len):
        with start_child_run(run_name="InputLen_{}_Run_{}".format(input_len, i),
                             inherit_params=True,
                             inherit_tags=True):
            # Indicate that this run has no more children
            mlflow.set_tag("worker_run", "true")

            train_ds, valid_ds, test_ds = (
                segment_presplit_dataset_using_config(train_ds=datasets[i][0],
                                                      val_ds=datasets[i][1],
                                                      test_ds=datasets[i][2],
                                                      cfg=configs))

            subjects_in_fold = subjects_in_folds[i]

            # TODO: should do multiple runs per fold with different seeds?
            # seed = base_seed + 111 * i
            # configs.dataset.seed = seed

            print(f"[INFO] : Run number {i} for input len {input_len}")
            print(f"[INFO] : Train subjects: {subjects_in_fold[0]}")
            print(f"[INFO] : Validation subjects: {subjects_in_fold[1]}")
            print(f"[INFO] : Test subjects: {subjects_in_fold[2]}")

            mlflow.log_params({"seed": configs.dataset.seed})
            mlflow.log_params({"train_subjects": subjects_in_fold[0],
                               "validation_subjects": subjects_in_fold[1],
                               "test_subjects": subjects_in_fold[2]})

            train(cfg=configs, train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds, run_name=f"InputLen_{input_len}_Run_{i}")

    configs.dataset.seed = base_seed
