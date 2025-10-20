# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import setuptools
import os
import shutil
import sys
from datetime import datetime
from hydra.core.hydra_config import HydraConfig
import hydra
import uuid
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from omegaconf import DictConfig
import mlflow
import mlflow.keras
import argparse
from clearml import Task
from clearml.backend_config.defs import get_active_config_file

sys.path.append(os.path.join(os.path.dirname(__file__), '../../common'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/benchmarking'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/deployment'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/quantization'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/optimization'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/evaluation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/training'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../deployment'))
sys.path.append(os.path.join(os.path.dirname(__file__), './preprocessing'))
sys.path.append(os.path.join(os.path.dirname(__file__), './training'))
sys.path.append(os.path.join(os.path.dirname(__file__), './utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), './evaluation'))
sys.path.append(os.path.join(os.path.dirname(__file__), './models'))


from logs_utils import mlflow_ini
from gpu_utils import set_gpu_memory_limit
from cfg_utils import get_random_seed
from preprocess import preprocess, load_and_filter_dataset_from_config, segment_dataset_from_config, train_test_split_pandas_df
from visualize_utils import display_figures
from parse_config import get_config
from train import train
from evaluate import evaluate
from deploy import deploy
from common_benchmark import benchmark, cloud_connect
from typing import Optional
from logs_utils import log_to_file
import pathlib


def mlflow_init(cfg: DictConfig, tracking_uri: str) -> None:
    """
    Initializes MLflow tracking with the given configuration.

    Args:
        cfg (dict): A dictionary containing the configuration parameters for MLflow tracking.

    Returns:
        None
    """

    if cfg is None:
        raise ValueError("Config is None")

    if tracking_uri == "":
        raise ValueError("Tracking URI is None")


    mlflow.set_tracking_uri(tracking_uri)

    if cfg.name is None:
        raise ValueError("Experiment name is None")

    print("[INFO] Experiment name is: ", cfg.name)
    mlflow.set_experiment(cfg.name)

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mlflow.start_run(run_name=run_name)

    params = {"operation_mode": cfg.operation_mode}
    mlflow.log_params(params)

    mlflow.tensorflow.autolog(log_models=False)

    mlflow.set_tags(cfg.tags)

    sweep_id = f"{cfg.name}_{datetime.now().strftime('%y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    mlflow.set_tag("sweep_id", sweep_id)


def chain_tb(cfg: DictConfig = None, train_ds: tf.data.Dataset = None,
             valid_ds: tf.data.Dataset = None, test_ds: tf.data.Dataset = None) -> None:
    """
    Runs the chain_tb pipeline, performs training and then benchmarking.

    Args:
        cfg (DictConfig): Configuration dictionary. Defaults to None.
        train_ds (tf.data.Dataset): Training dataset. Defaults to None.
        valid_ds (tf.data.Dataset): Validation dataset. Defaults to None.
        test_ds (tf.data.Dataset): Test dataset. Defaults to None.

    Returns:
        None
    """

    # Connect to STM32Cube.AI Developer Cloud
    credentials = None
    if cfg.tools.stm32ai.on_cloud:
        _, _, credentials = cloud_connect(stm32ai_version=cfg.tools.stm32ai.version)

    if test_ds:
        trained_model_path = train(cfg=cfg, train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds, run_idx=0)
    else:
        trained_model_path = train(cfg=cfg, train_ds=train_ds, valid_ds=valid_ds, run_idx=0)
    print('[INFO] : Training complete.')
    benchmark(cfg=cfg, model_path_to_benchmark=trained_model_path, credentials=credentials)
    print('[INFO] : benchmarking complete.')
    display_figures(cfg)

def start_child_run(run_name=None, inherit_params=True, inherit_tags=True, **kwargs):
    parent = mlflow.active_run()

    if parent is None:
        raise ValueError("Parent run is None")

    child = mlflow.start_run(run_name=run_name, nested=True, **kwargs)

    if parent and (inherit_params or inherit_tags):
        client = mlflow.tracking.MlflowClient()
        parent_run = client.get_run(parent.info.run_id)

        if inherit_params:
            parent_params = parent_run.data.params
            mlflow.log_params(parent_params)

        if inherit_tags:
            parent_tags = parent_run.data.tags

            parent_tags = {
                k: v for k, v in parent_run.data.tags.items()
                if not k.startswith("mlflow.")
            }

            mlflow.set_tags(parent_tags)

    return child

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

            train(cfg=configs, train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds, run_idx=input_len)

    configs.dataset.seed = base_seed

def experiment_mode(configs: DictConfig = None) -> None:
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

def process_mode(mode: str = None,
                 configs: DictConfig = None,
                 train_ds: tf.data.Dataset = None,
                 valid_ds: tf.data.Dataset = None,
                 test_ds: tf.data.Dataset = None,
                 float_model_path: Optional[str] = None) -> None:
    """
    Process the selected mode of operation.

    Args:
        mode (str): The selected mode of operation. Must be one of 'benchmarking', 'deployment',
                    'evaluation', 'training', or 'chain_tb'.
        configs (DictConfig): The configuration object.
        train_ds (tf.data.Dataset): The training dataset. Required if mode is 'train'.
        valid_ds (tf.data.Dataset): The validation dataset. Required if mode is 'train' or 'evaluate'.
        test_ds (tf.data.Dataset): The test dataset. Required if mode is 'evaluate'.
        float_model_path(str, optional): Model path . Defaults to None
    Returns:
        None
    Raises:
        ValueError: If an invalid operation_mode is selected or if required datasets are missing.
    """
    # Check the selected mode and perform the corresponding operation
    if mode == 'training':
        if test_ds:
            train(cfg=configs, train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds, run_idx=0)
        else:
            train(cfg=configs, train_ds=train_ds, valid_ds=valid_ds, run_idx=0)
        display_figures(configs)
        print('[INFO] : Training complete.')
    elif mode == 'experiment':
        print("[ERROR] : Experiment mode not supported.")
    elif mode == 'evaluation':
        if test_ds:
            evaluate(cfg=configs, eval_ds=test_ds, name_ds="test_set")
        else:
            evaluate(cfg=configs, eval_ds=valid_ds, name_ds="validation_set")
        display_figures(configs)
        print('[INFO] : Evaluation complete.')
    elif mode == 'deployment':
        deploy(cfg=configs)
        print('[INFO] : Deployment complete.')
    elif mode == 'benchmarking':
        benchmark(cfg=configs)
        print('[INFO] : Benchmark complete.')
    elif mode == 'chain_tb':
        chain_tb(cfg=configs,
                 train_ds=train_ds,
                 valid_ds=valid_ds,
                 test_ds=test_ds)
        print('[INFO] : chain_tb complete.')
    # Raise an error if an invalid mode is selected
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Record the whole hydra working directory to get all info
    folder = configs.output_dir  # this is a directory
    zip_path = shutil.make_archive("run_outputs", "zip", folder)
    mlflow.log_artifact(zip_path)  # single file => no copy_tree
    if mode in ['benchmarking', 'chain_tb']:
        mlflow.log_param("model_path", configs.general.model_path)
        mlflow.log_param("stm32ai_version", configs.tools.stm32ai.version)
        mlflow.log_param("target", configs.benchmarking.board)

    # logging the completion of the chain
    log_to_file(configs.output_dir, f'operation finished: {mode}')

    # ClearML - Example how to get task's context anywhere in the file.
    # Checks if there's a valid ClearML configuration file
    if get_active_config_file() is not None:
        print(f"[INFO] : ClearML task connection")
        task = Task.current_task()
        task.connect(configs)


@hydra.main(version_base=None, config_path="", config_name="user_config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point of the script.

    Args:
        cfg: Configuration dictionary.

    Returns:
        None
    """

    # Configure the GPU (the 'general' section may be missing)
    if "general" in cfg and cfg.general:
        # Set upper limit on usable GPU memory
        if "gpu_memory_limit" in cfg.general and cfg.general.gpu_memory_limit:
            set_gpu_memory_limit(cfg.general.gpu_memory_limit)
            print(f"[INFO] : Setting upper limit of usable GPU memory to {int(cfg.general.gpu_memory_limit)}GBytes.")
        else:
            print("[WARNING] The usable GPU memory is unlimited.\n"
                  "Please consider setting the 'gpu_memory_limit' attribute "
                  "in the 'general' section of your configuration file.")

    # Parse the configuration file
    cfg = get_config(cfg)
    cfg.output_dir = HydraConfig.get().run.dir

    # TODO! use the config file for experiment name
    mlflow_init(cfg.experiment, cfg.mlflow.uri)

    # Checks if there's a valid ClearML configuration file
    print(f"[INFO] : ClearML config check")
    if get_active_config_file() is not None:
        print(f"[INFO] : ClearML initialization and configuration")
        # ClearML - Initializing ClearML's Task object.
        task = Task.init(project_name=cfg.general.project_name,
                         task_name='har_modelzoo_task')
        # ClearML - Optional yaml logging
        task.connect_configuration(name=cfg.operation_mode,
                                   configuration=cfg)

    # Seed global seed for random generators
    seed = get_random_seed(cfg)
    print(f'[INFO] : The random seed for this simulation is {seed}')
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    # Extract the mode from the command-line arguments
    mode = cfg.operation_mode
    valid_modes = ['training',  'experiment', 'evaluation', 'chain_tb']
    if mode in valid_modes:
        if mode == 'experiment':
            print("[INFO] Starting experiment mode")
            # logging the operation_mode in the output_dir/stm32ai_main.log file
            log_to_file(cfg.output_dir, f'operation_mode: {mode}')
            experiment_mode(configs=cfg)
            print('[INFO] : Experiment complete.')
        else:
            # Perform further processing based on the selected mode
            preprocess_output = preprocess(cfg=cfg)
            train_ds, valid_ds, test_ds = preprocess_output
            # Process the selected mode
            process_mode(mode=mode,
                         configs=cfg,
                         train_ds=train_ds,
                         valid_ds=valid_ds,
                         test_ds=test_ds)
    else:
        # Process the selected mode
        process_mode(mode=mode,
                     configs=cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='', help='Path to folder containing configuration file')
    parser.add_argument('--config-name', type=str, default='user_config', help='name of the configuration file')
    # add arguments to the parser
    parser.add_argument('params', nargs='*',
                        help='List of parameters to over-ride in config.yaml')
    args = parser.parse_args()

    # Call the main function
    main()

    # log the config_path and config_name parameters
    mlflow.log_param('config_path', args.config_path)
    mlflow.log_param('config_name', args.config_name)
    mlflow.end_run()
