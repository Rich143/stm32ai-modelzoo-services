import mlflow
import tensorflow as tf
from datetime import datetime
import uuid
from typing import List, Tuple, TypeAlias
import pandas as pd
import re
import numpy as np

from omegaconf import DictConfig
from sklearn.model_selection import StratifiedGroupKFold

from data_load_helpers import (get_datasets_list,
                               dataset_subject_id_to_global_subject_id,
                               dataset_name_to_id)

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
SubjectListTriplet: TypeAlias = Tuple[List[int], List[int], List[int]]
SubjectListQuad: TypeAlias = Tuple[List[int], List[int], List[int], List[int]]

def config_subject_to_global_subject_id(subject: str) -> int:
    datasets = get_datasets_list()
    config_subject_pattern = re.compile(r"(.+)_([0-9]+)$")

    match = config_subject_pattern.match(subject)
    if not match:
        raise ValueError(f"Invalid subject format: {subject}")

    dataset_name = match.group(1).lower()

    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    subject_id = int(match.group(2))

    return dataset_subject_id_to_global_subject_id(subject_id=subject_id,
                                                   dataset_name=dataset_name)

def convert_subject_list_to_global_subject_id(subject_list: List[str]) -> List[int]:
    return [config_subject_to_global_subject_id(subject) for subject in subject_list]

def get_cv_subjects(experiment_cv_config: DictConfig,
                    dataset: pd.DataFrame) -> SubjectListQuad:
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

    train_subjects = convert_subject_list_to_global_subject_id(train_subjects)
    test_subjects = convert_subject_list_to_global_subject_id(test_subjects)
    excluded_subjects = convert_subject_list_to_global_subject_id(excluded_subjects)

    cv_subjects = check_subject_coverage_get_cv_subjects(dataset=dataset,
                                           test_subjects=test_subjects,
                                           train_subjects=train_subjects,
                                           excluded_subjects=excluded_subjects)

    return train_subjects, cv_subjects, test_subjects, excluded_subjects

import math
import matplotlib.pyplot as plt

def plot_class_distribution_per_user(df, dataset_name):
    """
    Plots histograms of class distribution (activity_label) for each user
    within the specified dataset. Creates up to 4 subplots per figure.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: ['dataset', 'user', 'activity_label']
    dataset_name : str
        The dataset to visualize
    """

    # Filter dataframe for the selected dataset
    sub = df[df['dataset'] == dataset_name]

    # Unique users in this dataset
    users = sorted(sub['user'].unique())
    n_users = len(users)

    # 4 subplots per figure
    plots_per_fig = 4
    n_figs = math.ceil(n_users / plots_per_fig)

    for fig_idx in range(n_figs):
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.ravel()

        start = fig_idx * plots_per_fig
        end = min(start + plots_per_fig, n_users)

        for ax_idx, user_idx in enumerate(range(start, end)):
            user = users[user_idx]
            ax = axes[ax_idx]

            user_data = sub[sub['user'] == user]

            ax.hist(user_data['activity_label'], bins='auto')
            ax.set_title(f"User {user}")
            ax.set_xlabel("Activity Label")
            ax.set_ylabel("Count")

        # Hide any remaining empty axes
        for ax in axes[end - start:]:
            ax.set_visible(False)

        fig.suptitle(f"Class Distribution – Dataset: {dataset_name} (Figure {fig_idx+1})")
        plt.tight_layout()
        plt.show()

def summarize_dataset_stats(
    df,
    dataset_col="dataset",
    subject_col="user",
    label_col="activity_label",
):
    results = {}

    # -------------------------
    # 1. Number of subjects per dataset
    # -------------------------
    subjects_per_dataset = (
        df.groupby(dataset_col)[subject_col]
        .nunique()
        .sort_values(ascending=False)
    )
    results["subjects_per_dataset"] = subjects_per_dataset

    # -------------------------
    # 2. Sample counts per subject
    # -------------------------
    sample_counts_per_subject = (
        df.groupby(subject_col)
        .size()
        .sort_values(ascending=False)
    )
    results["sample_counts_per_subject"] = sample_counts_per_subject

    # -------------------------
    # 3. Label distribution per subject
    # -------------------------
    datasets = df[dataset_col].unique()
    results["label_distribution_per_subject"] = {}
    for dataset in datasets:
        df_dataset = df[df[dataset_col] == dataset]
        label_dist_per_subject = (
            df_dataset.groupby([subject_col, label_col])
            .size()
            .rename("count")
            .reset_index()
            .pivot(index=subject_col, columns=label_col, values="count")
            .fillna(0)
            .astype(int)
        )
        results["label_distribution_per_subject"][dataset] = label_dist_per_subject

    return results

def plot_dataset_stats(stats):
    """
    Plot the output of summarize_dataset_stats().
    stats: dict returned by summarize_dataset_stats()
    """

    subjects_per_dataset = stats["subjects_per_dataset"]
    sample_counts_per_subject = stats["sample_counts_per_subject"]
    label_distribution = stats["label_distribution_per_subject"]

    # -----------------------------
    # 1. Subjects per dataset
    # -----------------------------
    plt.figure(figsize=(8, 4))
    subjects_per_dataset.plot(kind="bar")
    plt.title("Number of Subjects per Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Subject Count")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 2. Sample counts per subject
    # -----------------------------
    plt.figure(figsize=(10, 4))
    sample_counts_per_subject.plot(kind="bar")
    plt.title("Sample Counts per Subject")
    plt.xlabel("Subject")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 3. Label distribution per subject (heatmap-like)
    # -----------------------------
    datasets = stats["label_distribution_per_subject"].keys()
    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(10, 6))

        data = label_distribution[dataset].values
        subjects = label_distribution[dataset].index.astype(str)
        labels = label_distribution[dataset].columns.astype(str)

        im = ax.imshow(data, aspect="auto")

        # Tick labels
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")

        ax.set_yticks(np.arange(len(subjects)))
        ax.set_yticklabels(subjects)

        ax.set_title(f"Label Distribution per Subject (Dataset: {dataset})")
        ax.set_xlabel("Label")
        ax.set_ylabel("Subject")

        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()


def kfold_train_val_test(
    dataset: pd.DataFrame,
    test_subjects: List[int],
    always_train_subjects: List[int],
    cv_subjects: List[int],
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
) -> Tuple[List[DatasetTriplet], List[SubjectListTriplet]]:
    """
    Group aware K-Fold split where some subjects are always in training, and others rotate between train/val.
    Uses stratified K-Fold to ensure equal distribution of activities between train/val

    Returns a list of (train_df, val_df, test_df) tuples.
    """

    # Filter df down to only the rows belonging to cv subjects
    cv_df = dataset[dataset['user'].isin(cv_subjects)].reset_index(drop=True)

    ## Stats for debugging
    # plot_class_distribution_per_user(cv_df, "pamap2")
    # stats = summarize_dataset_stats(dataset)
    # plot_dataset_stats(stats)

    labels = cv_df['activity_label'].values
    groups = cv_df['user'].values
    datasets = (cv_df['dataset']
                .apply(dataset_name_to_id)
                .astype(int).values)

    # Stratified split (by dataset and label)
    y_strat = datasets * (labels.max() + 1) + labels

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    folds = []
    subjects_in_folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(cv_df, y_strat, groups)):
        # Convert row indices → subject IDs
        fold_train_subjects = sorted(cv_df.iloc[train_idx]['user'].unique().tolist())

        # Add always-train subjects if applicable
        if len(always_train_subjects) > 0:
            fold_train_subjects = sorted(set(fold_train_subjects).union(always_train_subjects))

        fold_val_subjects   = sorted(cv_df.iloc[val_idx]['user'].unique().tolist())

        # # Split datasets
        train_df = dataset[dataset['user'].isin(fold_train_subjects)].copy().reset_index(drop=True)
        val_df = dataset[dataset['user'].isin(fold_val_subjects)].copy().reset_index(drop=True)
        test_df = dataset[dataset['user'].isin(test_subjects)].copy().reset_index(drop=True)

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
    all_subjects_in_dataset = set(dataset['user'].unique())
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
