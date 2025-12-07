import mlflow
import tensorflow as tf
from datetime import datetime
import uuid
from typing import List, Tuple, TypeAlias
import pandas as pd
import re

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
SubjectListTriplet: TypeAlias = Tuple[List[str], List[str], List[str]]
SubjectListQuad: TypeAlias = Tuple[List[str], List[str], List[str], List[str]]

def get_cv_subjects(experiment_cv_config: DictConfig,
                    dataset: pd.DataFrame,
                    subject_col: str) -> SubjectListQuad:
    """
    Get CV subjects from experiment config.
    """

    if experiment_cv_config.train_subjects is None:
        raise ValueError("train_subjects is None")
    if experiment_cv_config.test_subjects is None:
        raise ValueError("test_subjects is None")
    if experiment_cv_config.excluded_subjects is None:
        raise ValueError("excluded_subjects is None")

    train_subjects = experiment_cv_config.train_subjects
    test_subjects = experiment_cv_config.test_subjects
    excluded_subjects = experiment_cv_config.excluded_subjects

    cv_subjects = check_subject_coverage_get_cv_subjects(dataset=dataset,
                                           subject_col=subject_col,
                                           test_subjects=test_subjects,
                                           train_subjects=train_subjects,
                                           excluded_subjects=excluded_subjects)

    return train_subjects, cv_subjects, test_subjects, excluded_subjects


def kfold_train_val_test(
    dataset: pd.DataFrame,
    subject_col: str,
    test_subjects: List[str],
    always_train_subjects: List[str],
    cv_subjects: List[str],
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
) -> Tuple[List[DatasetTriplet], List[SubjectListTriplet]]:
    """
    K-Fold split where some subjects are always in training, and others rotate between train/val.

    Returns a list of (train_df, val_df, test_df) tuples.
    """

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


def check_subject_coverage_get_cv_subjects(
    dataset: pd.DataFrame,
    subject_col: str,
    test_subjects: list,
    train_subjects: list,
    excluded_subjects: list,
) -> list:
    """
    Verify correct assignment of subjects into test, train, cv (computed), and excluded.
    Any subjects not in test/train/excluded are automatically assigned to cv.
    Raises an error if duplicates or extra subjects are found.

    Returns:
        cv_subjects (list of strings): Subjects automatically assigned to CV.
    """

    # Convert lists to sets
    all_subjects_in_dataset = set(dataset[subject_col].unique())
    test_set = set(test_subjects)
    train_set = set(train_subjects)
    excluded_set = set(excluded_subjects)

    # Check for unknown subjects
    user_defined_union = test_set | train_set | excluded_set
    extra_subjects = user_defined_union - all_subjects_in_dataset
    if extra_subjects:
        raise SubjectCoverageError(
            f"⚠️ Extra subjects (not in dataset): {sorted(extra_subjects)}"
        )

    # Check for duplicates between user-defined categories
    duplicates = []
    categories = {
        "test": test_set,
        "train": train_set,
        "excluded": excluded_set,
    }
    keys = list(categories.keys())

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            overlap = categories[keys[i]] & categories[keys[j]]
            if overlap:
                duplicates.append((keys[i], keys[j], overlap))

    if duplicates:
        msg = ["⚠️ Duplicate subjects across lists:"]
        for a, b, overlap in duplicates:
            msg.append(f"  {a} ↔ {b}: {sorted(overlap)}")
        raise SubjectCoverageError("\n".join(msg))

    # Compute CV subjects
    cv_set = all_subjects_in_dataset - user_defined_union
    cv_subjects = sorted(cv_set)

    # Summary output
    print("=== Subject Coverage Check ===")
    print(f"Total subjects in dataset: {len(all_subjects_in_dataset)}")
    print(f"Test subjects: {sorted(test_set)}")
    print(f"Train subjects: {sorted(train_set)}")
    print(f"Excluded subjects: {sorted(excluded_set)}")
    print(f"Computed CV subjects ({len(cv_subjects)}): {cv_subjects}")
    print("==============================\n")

    print("✅ All subjects are uniquely assigned to one category (test/train/cv/excluded).")

    return cv_subjects
