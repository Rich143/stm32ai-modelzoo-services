import mlflow
import tensorflow as tf
from datetime import datetime
import uuid

from omegaconf import DictConfig
from preprocess import load_and_filter_dataset_from_config, segment_dataset_from_config, train_test_split_pandas_df
from train import train
from experiments.experiment_utils import start_child_run

def input_len_experiment(configs: DictConfig = None) -> None:
    input_len_start, input_len_end, input_len_step, num_runs_per_input_len = get_experiment_params(cfg=configs)

    input_lengths = [i for i in range(input_len_start, input_len_end, input_len_step)]

    datasets = []
    input_shapes = []

    dataset = load_and_filter_dataset_from_config(cfg=configs)

    train_dataset, test_dataset = train_test_split_pandas_df(dataset=dataset,
                                                             test_split=configs.dataset.test_split,
                                                             seed=configs.dataset.seed)

    log_gaussian_noise_mlflow(cfg=configs)

    for input_len in input_lengths:
        input_shape = (input_len, 3, 1)
        input_shapes.append(input_shape)

        print("[INFO] : Preprocessing for input shape: ", input_shape)
        configs.training.model.input_shape = input_shape
        train_ds, valid_ds, test_ds = segment_dataset_from_config(cfg=configs, dataset=train_dataset, test_dataset=test_dataset)

        datasets.append((train_ds, valid_ds, test_ds))

    for i in range (len(input_shapes)):
        with start_child_run(run_name=f"Input Shape: {input_shapes[i]}",
                             inherit_params=True,
                             inherit_tags=True):
            child_run(input_shape=input_shapes[i],
                      num_runs_per_input_len=num_runs_per_input_len,
                      configs=configs,
                      train_ds=datasets[i][0],
                      valid_ds=datasets[i][1],
                      test_ds=datasets[i][2])

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
              train_ds: tf.data.Dataset,
              valid_ds: tf.data.Dataset,
              test_ds: tf.data.Dataset) -> None:
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

            seed = base_seed + 111 * i
            configs.dataset.seed = seed

            print("[INFO] : Run number {} for input len {}, seed {}".format(i, input_len, configs.dataset.seed))

            mlflow.log_params({"seed": configs.dataset.seed})

            train(cfg=configs, train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds, run_name=f"InputLen_{input_len}_Run_{i}")

    configs.dataset.seed = base_seed
