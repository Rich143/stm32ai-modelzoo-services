import mlflow
import tensorflow as tf
from datetime import datetime
import uuid
from typing import List, Tuple, TypeAlias
import pandas as pd

from omegaconf import DictConfig
from sklearn.model_selection import KFold

# from preprocess import load_and_filter_dataset_from_config, segment_dataset_from_config
# from train import train

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

DatasetTriplet: TypeAlias = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
SubjectListTriplet: TypeAlias = Tuple[List[int], List[int], List[int]]

def kfold_train_val_test(
    dataset: pd.DataFrame,
    subject_col: str,
    test_subjects: List[int],
    always_train_subjects: List[int],
    cv_subjects: List[int],
    excluded_subjects: List[int],
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
) -> Tuple[List[DatasetTriplet], List[SubjectListTriplet]]:
    """
    K-Fold split where some subjects are always in training, and others rotate between train/val.

    Returns a list of (train_df, val_df, test_df) tuples.
    """

    check_subject_coverage(
        dataset=dataset,
        subject_col=subject_col,
        test_subjects=test_subjects,
        train_subjects=always_train_subjects,
        cv_subjects=cv_subjects,
        excluded_subjects=excluded_subjects
    )

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    folds = []
    subjects_in_folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(cv_subjects)):
        # Subjects for this fold
        fold_train_subjects = [cv_subjects[i] for i in train_idx] + always_train_subjects
        fold_val_subjects = [cv_subjects[i] for i in val_idx]

        # Split datasets
        train_df = dataset[dataset[subject_col].isin(fold_train_subjects)].copy().reset_index(drop=True)
        val_df = dataset[dataset[subject_col].isin(fold_val_subjects)].copy().reset_index(drop=True)
        test_df = dataset[dataset[subject_col].isin(test_subjects)].copy().reset_index(drop=True)

        print(f"[FOLD {fold_idx+1}]")
        print(f"  Train subjects: {sorted(fold_train_subjects)}")
        print(f"  Val subjects:   {sorted(fold_val_subjects)}")
        print(f"  Test subjects:  {sorted(test_subjects)}")

        folds.append((train_df, val_df, test_df))
        subjects_in_folds.append((fold_train_subjects, fold_val_subjects, test_subjects))

    return (folds, subjects_in_folds)

class SubjectCoverageError(Exception):
    """Custom exception for subject coverage issues."""
    pass


def check_subject_coverage(
    dataset: pd.DataFrame,
    subject_col: str,
    test_subjects: list,
    train_subjects: list,
    cv_subjects: list,
    excluded_subjects: list,
) -> None:
    """
    Verify that all subjects in the dataset are included in exactly one of the provided lists.
    Raises an error if duplicates, missing, or extra subjects are found.
    """
    all_subjects_in_dataset = set(dataset[subject_col].unique())
    defined_subjects = {
        "test": set(test_subjects),
        "train": set(train_subjects),
        "cv": set(cv_subjects),
        "excluded": set(excluded_subjects),
    }

    # Combine all defined subjects
    all_defined_subjects = set.union(*defined_subjects.values())

    # Identify issues
    missing_subjects = all_subjects_in_dataset - all_defined_subjects
    extra_subjects = all_defined_subjects - all_subjects_in_dataset

    duplicates = []
    keys = list(defined_subjects.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            overlap = defined_subjects[keys[i]] & defined_subjects[keys[j]]
            if overlap:
                duplicates.append((keys[i], keys[j], overlap))

    # Build summary message
    messages = []
    messages.append("=== Subject Coverage Check ===")
    messages.append(f"Total unique subjects in dataset: {len(all_subjects_in_dataset)}")
    messages.append(f"Subjects defined across all lists: {len(all_defined_subjects)}\n")

    if missing_subjects:
        messages.append(f"⚠️ Missing subjects (not in any list): {sorted(missing_subjects)}")
    if extra_subjects:
        messages.append(f"⚠️ Extra subjects (not in dataset): {sorted(extra_subjects)}")
    if duplicates:
        messages.append("⚠️ Duplicates found across lists:")
        for a, b, overlap in duplicates:
            messages.append(f"  {a} ↔ {b}: {sorted(overlap)}")

    if not (missing_subjects or extra_subjects or duplicates):
        messages.append("✅ All subjects are uniquely assigned to one list.")
        print("\n".join(messages))
        print("==============================\n")
        return

    # If any problem, raise error with detailed info
    error_message = "\n".join(messages)
    raise SubjectCoverageError(error_message)
