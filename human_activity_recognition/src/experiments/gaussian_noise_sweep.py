import mlflow
import numpy as np
import tensorflow as tf
from datetime import datetime
import uuid

from omegaconf import DictConfig
from preprocess import load_and_filter_dataset_from_config, segment_dataset_from_config, train_test_split_pandas_df
from train import train
from experiments.experiment_utils import start_child_run

def gaussian_noise_experiment(configs: DictConfig = None) -> None:
    noise_sweep_start, noise_sweep_end, noise_sweep_step, num_runs_per_step = get_experiment_params(cfg=configs)

    noise_values = np.arange(noise_sweep_start, noise_sweep_end, noise_sweep_step)

    datasets = []
    input_shapes = []

    dataset = load_and_filter_dataset_from_config(cfg=configs)

    train_dataset, test_dataset = train_test_split_pandas_df(dataset=dataset,
                                                             test_split=configs.dataset.test_split,
                                                             seed=configs.dataset.seed)

    input_shape = configs.training.model.input_shape
    print("[INFO] : Preprocessing for input shape: ", input_shape)
    train_ds, valid_ds, test_ds = segment_dataset_from_config(cfg=configs, dataset=train_dataset, test_dataset=test_dataset)

    mlflow.log_params({"input_shape": input_shape})
    mlflow.log_params({"input_length": input_shape[0]})

    for noise_std in noise_values:
        with start_child_run(run_name=f"Noise_STD_{noise_std}",
                             inherit_params=True,
                             inherit_tags=True):
            child_run(noise_std=noise_std,
                      num_runs_per_step=num_runs_per_step,
                      configs=configs,
                      train_ds=train_ds,
                      valid_ds=valid_ds,
                      test_ds=test_ds)

def log_gaussian_noise_mlflow(cfg):
    pass
    # mlflow.log_param("gaussian_noise", cfg.preprocessing.gaussian_noise)
    # mlflow.log_param("gaussian_std", cfg.preprocessing.gaussian_std)

def get_experiment_params(cfg: DictConfig = None):
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
              train_ds: tf.data.Dataset,
              valid_ds: tf.data.Dataset,
              test_ds: tf.data.Dataset) -> None:
    if noise_std > 0:
        configs.preprocessing.gaussian_noise = True
        configs.preprocessing.gaussian_std = noise_std
    else:
        configs.preprocessing.gaussian_noise = False
        configs.preprocessing.gaussian_std = 0

    print("[INFO] : Noise STD: ", configs.preprocessing.gaussian_std)
    log_gaussian_noise_mlflow(cfg=configs)

    base_seed = configs.dataset.seed

    for i in range(num_runs_per_step):
        with start_child_run(run_name="Noise_STD_{}_Run_{}".format(noise_std, i),
                             inherit_params=True,
                             inherit_tags=True):
            # Indicate that this run has no more children
            mlflow.set_tag("worker_run", "true")

            seed = base_seed + 111 * i
            configs.dataset.seed = seed

            print("[INFO] : Run number {} for noise std {}, seed {}".format(i, noise_std, configs.dataset.seed))

            mlflow.log_params({"seed": configs.dataset.seed})

            train(cfg=configs, train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds, run_name=f"Noise_STD_{noise_std}_Run_{i}")

    configs.dataset.seed = base_seed
