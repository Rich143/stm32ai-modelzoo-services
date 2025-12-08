import numpy as np
import pandas as pd
import os
from glob import glob
import re
from typing import Tuple
import sys

from data_load_helpers import (fill_nans, copy_accel_to_xyz, dataset_subject_id_to_global_subject_id,
                               global_activity_name_to_id)

# Central lookup: name (lowercase) → ID
_USERNAME_TO_ID = {
    'richard': 1,
    'elizabeth': 2,
    'sonja': 3,
}

def user_name_to_id(name: str) -> int:
    """
    Convert a username to its ID.
    Raises KeyError if the name is not found.
    """
    key = name.lower().strip()
    if key not in _USERNAME_TO_ID:
        raise KeyError(f"Username not found: '{key}'")
    return _USERNAME_TO_ID[key]

# Central lookup: name (lowercase) → ID
_ACTIVITY_NAME_TO_ID = {
    'walking': 1,
    'running': 2,
    'cycling': 3,
    'stationary': 4,
}

# Reverse: ID → name
_ACTIVITY_ID_TO_NAME = {v: k for k, v in _ACTIVITY_NAME_TO_ID.items()}

# Map dataset activities to standardized activity names
# Format is {dataset_name: global_name}
_RAD_NAME_TO_GLOBAL_NAME_MAPPING = {
    'walking': 'walking',
    'running': 'running',
    'stationary': 'stationary',
    'cycling': 'cycling'
}

def class_name_to_id(name: str) -> int:
    """
    Convert a PAMAP2 activity name to its ID.
    Raises KeyError if the name is not found.
    """
    key = name.lower().strip()
    if key not in _ACTIVITY_NAME_TO_ID:
        raise KeyError(f"Activity name not found: '{name}'")
    return _ACTIVITY_NAME_TO_ID[key]


def class_id_to_name(class_id: int) -> str:
    """
    Convert an activity ID to its name.
    Raises KeyError if the ID is not found.
    """
    if class_id not in _ACTIVITY_ID_TO_NAME:
        raise KeyError(f"Activity ID not found: {class_id}")
    return _ACTIVITY_ID_TO_NAME[class_id]

# dataset = dataset[['x', 'y', 'z', 'activity_label', 'segment_id', 'user']]
def rename_cols_drop_unused(df: pd.DataFrame):
    df = df[['x', 'y', 'z', 'activity_label', 'segment_id', 'user']]

    return df

def convert_local_activity_ids_to_global(dataset: pd.DataFrame):
    mapping_dict = {}
    unknown_activity_id = global_activity_name_to_id('unknown')

    for rad_name, global_name in _RAD_NAME_TO_GLOBAL_NAME_MAPPING.items():
        global_id = global_activity_name_to_id(global_name)
        rad_id = class_name_to_id(rad_name)

        mapping_dict[rad_id] = global_id

    dataset['activity_label'] = (
        dataset['activity_label']
        .map(mapping_dict)
        .fillna(unknown_activity_id)
        .astype(int))


def process_rad(df: pd.DataFrame):
    df = copy_accel_to_xyz(df, source='unproc')

    df = rename_cols_drop_unused(df)

    df = fill_nans(df)

    convert_local_activity_ids_to_global(df)

    return df

def load_rad_from_file_and_segment(dataset_path: str,
                                   global_segment_id: int) -> Tuple[pd.DataFrame, int]:
    """
    Load the RAD dataset as a pandas DataFrame.

    Args:
        dataset_path (str): Path to the root of the dataset.

    Returns:
        pd.DataFrame: Combined DataFrame with columns unproc_x, unproc_y, unproc_z,
                      plus 'user' and 'activity' columns.
    """
    data_frames = []
    segment_counter = global_segment_id

    for user_name in os.listdir(dataset_path):
        user_path = os.path.join(dataset_path, user_name)
        if not os.path.isdir(user_path):
            continue
        for activity_name in os.listdir(user_path):
            activity_path = os.path.join(user_path, activity_name)
            if not os.path.isdir(activity_path):
                continue
            for file_name in os.listdir(activity_path):
                if file_name.endswith(".csv"):
                    file_path = os.path.join(activity_path, file_name)
                    df = pd.read_csv(file_path, usecols=['unproc_x', 'unproc_y', 'unproc_z'])

                    print(f"[INFO] Loading RAD file: {file_path}")
                    df['username'] = user_name

                    local_subject_id = user_name_to_id(user_name)
                    global_subject_id = (
                        dataset_subject_id_to_global_subject_id(subject_id=local_subject_id,
                                                                dataset_name='rad')
                    )

                    df['user'] = global_subject_id
                    df['activity_label'] = class_name_to_id(activity_name)
                    df['activity_name'] = activity_name

                    df['segment_id'] = segment_counter
                    segment_counter += 1

                    data_frames.append(df)

    if data_frames:
        return pd.concat(data_frames, ignore_index=True), segment_counter

    raise FileNotFoundError(f"No .csv files found in {dataset_path}")

# Public Functions

def load_and_process_rad(dataset_path: str, global_segment_id: int) -> Tuple[pd.DataFrame, int]:
    df, globel_segment_id = load_rad_from_file_and_segment(dataset_path,
                                                           global_segment_id=global_segment_id)

    df = process_rad(df)

    return df, globel_segment_id
