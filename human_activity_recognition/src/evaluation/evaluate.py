# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import sys
from pathlib import Path
import warnings
import sklearn
import mlflow
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tqdm
from typing import Optional

from models_utils import compute_confusion_matrix, count_h5_parameters
from visualize_utils import plot_confusion_matrix
from models_mgt import get_loss
from logs_utils import log_to_file

def log_per_class_metrics(y_true, y_pred, class_names, prefix="test"):
    # 1) Per-class PRF1 as scalar metrics (best for Compare view)
    rpt = sklearn.metrics.classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    for cls in class_names:
        mlflow.log_metric(f"{prefix}/precision/{cls}", rpt[cls]["precision"])
        mlflow.log_metric(f"{prefix}/recall/{cls}",    rpt[cls]["recall"])
        mlflow.log_metric(f"{prefix}/f1/{cls}",        rpt[cls]["f1-score"])
        mlflow.log_metric(f"{prefix}/support/{cls}",   rpt[cls]["support"])

    # 2) Overall aggregates you’ll likely compare across runs
    for agg in ["macro avg", "weighted avg"]:
        mlflow.log_metric(f"{prefix}/f1_{agg.replace(' ', '_')}", rpt[agg]["f1-score"])
        mlflow.log_metric(f"{prefix}/precision_{agg.replace(' ', '_')}", rpt[agg]["precision"])
        mlflow.log_metric(f"{prefix}/recall_{agg.replace(' ', '_')}", rpt[agg]["recall"])

    # 3) (Optional) Normalized confusion-matrix cells as metrics for deep diffs
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums
    for i, ti in enumerate(class_names):
        for j, pj in enumerate(class_names):
            mlflow.log_metric(f"{prefix}/cm/{ti}-{pj}", float(cm_norm[i, j]))

# def log_confusion_matrix_to_mlflow(labels: np.ndarray, logits: np.ndarray, name_ds: str):
    # # after you've built `labels` and `logits` (int class ids) in compute_confusion_matrix():
    # eval_df = pd.DataFrame({"label": labels, "prediction": logits})

    # dataset = mlflow.data.from_pandas(
        # eval_df, targets="label", predictions="prediction", name=f"{name_ds}_preds"
    # )

    # mlflow.evaluate(                 # creates CM image + table + standard classifier metrics
        # data=dataset,
        # model_type="classifier",
        # evaluators=["default"],
    # )

    # log_per_class_metrics(labels, logits, class_names, prefix=f"{name_ds}")

def evaluate_h5_model(model_path: str = None,
                      eval_ds: tf.data.Dataset = None,
                      class_names: list = None,
                      output_dir: str = None,
                      name_ds: str = 'test_set') -> float:
    """
    Evaluates a trained Keras model saved in .h5 format on the provided test data.

    Args:
        model_path (str): The file path to the .h5 model.
        eval_ds (tf.data.Dataset): The test data to evaluate the model on.
        class_names (list): A list of class names for the confusion matrix.
        output_dir (str): The directory where to save the confusion matrix image.
        name_ds (str): The name of the chosen eval_ds to be mentioned in the prints and figures.
    Returns:
        float: The accuracy of the model on the test data.
    """

    # Load the .h5 model
    model = tf.keras.models.load_model(model_path)
    loss = get_loss(len(class_names))
    model.compile(loss=loss, metrics=['accuracy'])

    # Evaluate the model on the test data
    tf.print(f'[INFO] : Evaluating the float model using {name_ds}...')
    loss, accuracy = model.evaluate(eval_ds)

    # Calculate the confusion matrix.
    cm, test_accuracy, labels, logits = compute_confusion_matrix(test_set=eval_ds, model=model)
    ##########################################
    # Log the confusion matrix as an image summary.
    model_name = f"float_model_confusion_matrix_{name_ds}"
    plot_confusion_matrix(cm=cm, class_names=class_names, model_name=model_name,
                          title=f'{model_name}\naccuracy: {test_accuracy}', output_dir=output_dir)
    print(f"[INFO] : Accuracy of float model = {test_accuracy}%")
    print(f"[INFO] : Loss of float model = {loss}")
    mlflow.log_metric(f"float_acc_{name_ds}", test_accuracy)
    mlflow.log_metric(f"float_loss_{name_ds}", loss)
    log_to_file(output_dir, f"Float model {name_ds}:")
    log_to_file(output_dir, f"Accuracy of float model : {test_accuracy} %")
    log_to_file(output_dir, f"Loss of float model : {round(loss,2)} ")

    log_per_class_metrics(labels, logits, class_names, prefix=name_ds)

    mlflow.log_artifact(f"{output_dir}/float_model_confusion_matrix_{name_ds}.png")

    return accuracy


def evaluate(cfg: DictConfig = None, eval_ds: tf.data.Dataset = None,
             model_path_to_evaluate: Optional[str] = None, name_ds: Optional[str] = 'test_set') -> None:
    """
    Evaluates and benchmarks a TensorFlow Lite or Keras model, and generates a Config header file if specified.

    Args:
        cfg (config): The configuration file.
        eval_ds (tf.data.Dataset): The validation dataset.
        model_path_to_evaluate (str, optional): Model path to evaluate
        name_ds (str): The name of the chosen test_data to be mentioned in the prints and figures.

    Returns:
        None
    """
    output_dir = HydraConfig.get().runtime.output_dir
    class_names = cfg.dataset.class_names
    
    model_path = model_path_to_evaluate if model_path_to_evaluate else cfg.general.model_path

    try:
        # Check if the model is a TensorFlow Lite model
        file_extension = Path(model_path).suffix
        if file_extension == '.h5':
            count_h5_parameters(output_dir=output_dir, 
                                model_path=model_path)
            # Evaluate Keras model
            evaluate_h5_model(model_path=model_path, eval_ds=eval_ds,
                              class_names=class_names, output_dir=output_dir, name_ds=name_ds)
    except Exception:
        raise ValueError("Model accuracy evaluation failed because of wrong model type!\n",
                         f"Received model path: {model_path}")
