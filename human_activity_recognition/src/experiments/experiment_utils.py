import mlflow
import tensorflow as tf
from datetime import datetime
import uuid

from omegaconf import DictConfig
from preprocess import load_and_filter_dataset_from_config, segment_dataset_from_config, train_test_split_pandas_df
from train import train

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
            print("parent_params: ", parent_params)
            mlflow.log_params(parent_params)

        if inherit_tags:
            parent_tags = parent_run.data.tags

            parent_tags = {
                k: v for k, v in parent_run.data.tags.items()
                if not k.startswith("mlflow.")
            }

            mlflow.set_tags(parent_tags)

    return child
