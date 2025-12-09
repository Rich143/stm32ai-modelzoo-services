import numpy as np
import pandas as pd
import os
from glob import glob
from data_load_helpers import dataset_subject_id_to_global_subject_id, global_activity_name_to_id, copy_accel_to_xyz, fill_nans

def load_pamap2_from_file_and_segment(dataset_path: str,
                                      global_segment_id: int,
                                      max_gap_s: float = 0.1):
    """
    Load and combine PAMAP2 protocol dataset, keeping only timestamp, subject_id, activity_id,
    and 16g accelerometer data. Filters out activity_id == 0 and segments the data into
    time-continuous chunks per subject and activity.

    Adds a time_diff_ms column (milliseconds between samples) and a globally unique segment_id.

    Parameters:
        dataset_path (str): Path to the PAMAP2 dataset
        max_gap_s (float): Max allowed gap (in seconds) to consider data as continuous.

    Returns:
        pd.DataFrame: Combined and segmented DataFrame with globally unique segment_id.
    """
    # Columns: 0 = timestamp, 1 = activity_id
    # 4-6 = hand acc16 x/y/z, 21-23 = chest acc16 x/y/z, 38-40 = ankle acc16 x/y/z
    usecols = [0, 1] + list(range(4, 7)) + list(range(21, 24)) + list(range(38, 41))
    col_names = ['Arrival_Time', 'Activity_Label'] + \
                [f'hand_acc16_{axis}' for axis in ['x', 'y', 'z']] + \
                [f'chest_acc16_{axis}' for axis in ['x', 'y', 'z']] + \
                [f'ankle_acc16_{axis}' for axis in ['x', 'y', 'z']]

    all_dfs = []
    protocol_folder = os.path.join(dataset_path, 'Protocol')
    file_pattern = os.path.join(protocol_folder, 'subject1*.dat')
    segment_counter = global_segment_id

    for file_path in sorted(glob(file_pattern)):
        subject_id = int(os.path.basename(file_path)[7:10])
        print("[INFO] Loading pamap2 subject ", subject_id)

        df = pd.read_csv(file_path, sep=' ', header=None, usecols=usecols, names=col_names)
        df['user'] = dataset_subject_id_to_global_subject_id(subject_id,
                                                             'pamap2')
        df['dataset'] = 'pamap2'

        # Sort and compute time differences
        df = df.sort_values('Arrival_Time').reset_index(drop=True)
        df['time_diff_ms'] = df['Arrival_Time'].diff().fillna(0) * 1000

        # Segment using activity transitions (including to/from activity_id == 0)
        activity_change = df['Activity_Label'] != df['Activity_Label'].shift()
        time_gap = df['time_diff_ms'] > max_gap_s * 1000
        discontinuity = activity_change | time_gap
        df['local_segment_id'] = discontinuity.cumsum().astype(int)

        # Only keep rows where activity_id != 0
        df_valid = df[df['Activity_Label'] != 0].copy()
        if df_valid.empty:
            continue

        # Assign global segment IDs only to valid segments
        unique_local_segments = df_valid['local_segment_id'].unique()
        segment_id_map = {local_id: segment_counter + i for i, local_id in enumerate(unique_local_segments)}
        df_valid['segment_id'] = df_valid['local_segment_id'].map(segment_id_map)
        segment_counter += len(unique_local_segments)

        df_valid.drop(columns='local_segment_id', inplace=True)
        all_dfs.append(df_valid)

    df = pd.concat(all_dfs, ignore_index=True)

    return df, segment_counter

# dataset = dataset[['x', 'y', 'z', 'activity_label', 'segment_id',
#                    'user', 'dataset']]
def rename_cols_drop_unused(df: pd.DataFrame):
    df = df.rename(columns={
        'Arrival_Time': 'timestamp',
        'Activity_Label': 'activity_label'
    })

    df = df[['x', 'y', 'z', 'activity_label', 'segment_id',
             'user', 'dataset']]

    return df

# Central lookup: name (lowercase) → ID
_ACTIVITY_NAME_TO_ID = {
    'lying': 1,
    'sitting': 2,
    'standing': 3,
    'walking': 4,
    'running': 5,
    'cycling': 6,
    'nordic walking': 7,
    'watching tv': 9,
    'computer work': 10,
    'car driving': 11,
    'ascending stairs': 12,
    'descending stairs': 13,
    'vacuum cleaning': 16,
    'ironing': 17,
    'folding laundry': 18,
    'house cleaning': 19,
    'playing soccer': 20,
    'rope jumping': 24,
    'other': 0,
    'stationary': 25,
    'stairs': 26
}

# Reverse: ID → name
_ACTIVITY_ID_TO_NAME = {v: k for k, v in _ACTIVITY_NAME_TO_ID.items()}

def activity_name_to_id(name: str) -> int:
    """
    Convert a PAMAP2 activity name to its ID.
    Raises KeyError if the name is not found.
    """
    key = name.lower().strip()
    if key not in _ACTIVITY_NAME_TO_ID:
        raise KeyError(f"Activity name not found: '{name}'")
    return _ACTIVITY_NAME_TO_ID[key]


def activity_id_to_name(class_id: int) -> str:
    """
    Convert an activity ID to its name.
    Raises KeyError if the ID is not found.
    """
    if class_id not in _ACTIVITY_ID_TO_NAME:
        raise KeyError(f"Activity ID not found: {class_id}")
    return _ACTIVITY_ID_TO_NAME[class_id]

def group_activity_ids(dataset):
    """
    Replace fine-grained activity IDs in `Activity_Label` with grouped activity IDs
    using 'stationary' and 'stairs' where applicable.

    Modifies the dataset in-place.
    """
    # Grouped activity name → group label
    grouping_map = {
        'lying': 'stationary',
        'sitting': 'stationary',
        'standing': 'stationary',
        'ascending stairs': 'stairs',
        'descending stairs': 'stairs',
    }

    # Build id-to-group-id mapping using your class_name_to_id()
    id_to_group_id = {
        activity_name_to_id(orig): activity_name_to_id(grouped)
        for orig, grouped in grouping_map.items()
    }

    # Apply mapping (leave other IDs unchanged)
    dataset['activity_label'] = dataset['activity_label'].map(id_to_group_id).fillna(dataset['activity_label'])

# Map dataset activities to standardized activity names
# Format is {dataset_name: global_name}
_PAMAP2_NAME_TO_GLOBAL_NAME_MAPPING = {
    'walking': 'walking',
    'running': 'running',
    'stationary': 'stationary',
    'stairs': 'stairs',
    'cycling': 'cycling'
}

def convert_local_activity_ids_to_global(dataset: pd.DataFrame):
    mapping_dict = {}
    unknown_activity_id = global_activity_name_to_id('unknown')

    for pamap2_name, global_name in _PAMAP2_NAME_TO_GLOBAL_NAME_MAPPING.items():
        global_id = global_activity_name_to_id(global_name)
        pamap2_id = activity_name_to_id(pamap2_name)

        mapping_dict[pamap2_id] = global_id

    dataset['activity_label'] = (
        dataset['activity_label']
        .map(mapping_dict)
        .fillna(unknown_activity_id)
        .astype(int))

def downsample_pamap2(df: pd.DataFrame, downsample_factor: int):
    df_down = df.iloc[::downsample_factor].reset_index(drop=True)

    return df_down

def process_pamap2(df):
    df = copy_accel_to_xyz(df, source='chest_acc16')

    df = rename_cols_drop_unused(df)

    df = fill_nans(df)

    group_activity_ids(df)

    convert_local_activity_ids_to_global(df)

    # Convert to 50 Hz
    df = downsample_pamap2(df, downsample_factor=2)

    return df

def load_and_process_pamap2(dataset_path: str,
                            global_segment_id: int):
    df, global_segment_id = load_pamap2_from_file_and_segment(dataset_path,
                                                              global_segment_id=global_segment_id)

    df = process_pamap2(df)

    return df, global_segment_id
