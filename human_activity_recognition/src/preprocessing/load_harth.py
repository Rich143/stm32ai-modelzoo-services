import numpy as np
import pandas as pd
import os
from glob import glob
import re
from data_load_helpers import global_activity_name_to_id, copy_accel_to_xyz, fill_nans

# Central lookup: name (lowercase) → ID
_ACTIVITY_NAME_TO_ID = {
    'walking': 1,
    'running': 2,
    'shuffling': 3,
    'stairs (ascending)': 4,
    'stairs (descending)': 5,
    'standing': 6,
    'sitting': 7,
    'lying': 8,
    'cycling (sit)': 13,
    'cycling (stand)': 14,
    'cycling (sit, inactive)': 130,
    'cycling (stand, inactive)': 140,

    # Added ids
    'cycling': 200,
    'stationary': 201,
    'stairs': 202
}

# Reverse: ID → name
_ACTIVITY_ID_TO_NAME = {v: k for k, v in _ACTIVITY_NAME_TO_ID.items()}

# Map dataset activities to standardized activity names
# Format is {dataset_name: global_name}
_HARTH_NAME_TO_GLOBAL_NAME_MAPPING = {
    'walking': 'walking',
    'running': 'running',
    'stationary': 'stationary',
    'stairs': 'stairs',
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
        'stairs (ascending)': 'stairs',
        'stairs (descending)': 'stairs',
        'cycling (sit)': 'cycling',
        'cycling (stand)': 'cycling',
        'cycling (sit, inactive)': 'cycling',
        'cycling (stand, inactive)': 'cycling',
    }

    # Build id-to-group-id mapping using your class_name_to_id()
    id_to_group_id = {
        class_name_to_id(orig): class_name_to_id(grouped)
        for orig, grouped in grouping_map.items()
    }

    # Apply mapping (leave other IDs unchanged)
    dataset['activity_label'] = dataset['activity_label'].map(id_to_group_id).fillna(dataset['activity_label'])

def convert_local_activity_ids_to_global(dataset: pd.DataFrame):
    mapping_dict = {}
    unknown_activity_id = global_activity_name_to_id('unknown')

    for harth_name, global_name in _HARTH_NAME_TO_GLOBAL_NAME_MAPPING.items():
        global_id = global_activity_name_to_id(global_name)
        harth_id = class_name_to_id(harth_name)

        mapping_dict[harth_id] = global_id

    dataset['activity_label'] = (
        dataset['activity_label']
        .map(mapping_dict)
        .fillna(unknown_activity_id)
        .astype(int))


def get_subject_id(file_path):
    filename = os.path.basename(file_path)

    match = re.match(r"S(\d+)\.csv$", filename)
    if match:
        subject_id = int(match.group(1))
        return subject_id
    else:
        raise ValueError(f"Invalid file name format: {filename}")

def check_sample_rate(df, expected_sample_rate_hz=50, tolerance=0.05):
    target = 1 / expected_sample_rate_hz * 1000

    lower = target * (1 - tolerance)   # 19
    upper = target * (1 + tolerance)   # 21

    df['bad_interval'] = ~df['time_diff_ms'].between(lower, upper)

    if (df['bad_interval']).any():
        bad_rows = df[df['bad_interval']]
        print(bad_rows)
        raise ValueError(f"Sample rate is not {expected_sample_rate_hz}Hz")

def parse_and_check_timestamps(df):
    df['timestamp_parsed'] = pd.to_datetime(
        df['timestamp'],
        format="%Y-%m-%d %H:%M:%S.%f",
        errors='coerce'
    )

    bad_rows = df[df['timestamp_parsed'].isna()]

    if len(bad_rows) > 0:
        print(bad_rows)
        raise ValueError("Timestamps could not be parsed")

    df['timestamp'] = df['timestamp_parsed']
    df = df.drop(columns='timestamp_parsed')

    return df


# dataset = dataset[['timestamp', 'x', 'y', 'z', 'activity_label', 'segment_id', 'user']]
def rename_cols_drop_unused(df: pd.DataFrame):
    df = df.rename(columns={'label': 'activity_label'})

    df = df[['timestamp', 'x', 'y', 'z', 'activity_label', 'segment_id', 'user']]

    return df

def convert_gs_to_m_s_2(df):
    df['x'] = df['x'] * 9.81
    df['y'] = df['y'] * 9.81
    df['z'] = df['z'] * 9.81

    return df

def process_harth(df: pd.DataFrame):
    df = copy_accel_to_xyz(df, source='back')

    df = rename_cols_drop_unused(df)

    df = fill_nans(df)

    group_activity_ids(df)

    convert_local_activity_ids_to_global(df)

    df = convert_gs_to_m_s_2(df)

    return df

def print_time_gaps(df, time_gap, pad=3):
    """
    Print a few rows around each time gap in the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.
    time_gap : pandas.Series[bool]
        Boolean mask where True marks the start of a gap.
    pad : int
        Number of rows before and after each gap to display.
    """
    gap_indices = df.index[time_gap].to_list()
    
    if not gap_indices:
        print("No time gaps found.")
        return

    for idx in gap_indices:
        start = max(idx - pad, 0)
        end = min(idx + pad, len(df) - 1)
        
        print(f"\n=== Time gap around index {idx} ===")
        print(df.iloc[start:end+1])

def load_harth_from_file_and_segment(dataset_path: str,
                                     global_segment_id: int,
                                     max_gap_s: float = 0.2,
                                     expected_sample_rate_hz = 50):
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

    col_names = [
        "timestamp","back_x","back_y","back_z","thigh_x","thigh_y","thigh_z","label"]

    col_names_with_index = [
        "timestamp","index", "back_x","back_y","back_z","thigh_x","thigh_y","thigh_z","label"]

    subjects_with_idx = [21, 15]

    all_dfs = []
    # file_pattern = os.path.join(dataset_path, 'S006.csv')
    file_pattern = os.path.join(dataset_path, 'S*.csv')
    segment_counter = global_segment_id

    for file_path in sorted(glob(file_pattern)):
        subject_id = get_subject_id(file_path)
        print(f"Loading harth subject {subject_id}")

        subject_col_names = col_names

        has_index_col = subject_id in subjects_with_idx
        if has_index_col:
            subject_col_names = col_names_with_index

        df = pd.read_csv(file_path,
                         sep=',',
                         dtype={'timestamp': 'string'},  # force raw strings
                         skiprows=1,
                         names=subject_col_names)

        if has_index_col:
            df = df.drop(columns='index')

        df = parse_and_check_timestamps(df)

        df['user'] = f"HARTH_{subject_id}"

        is_sorted = df['timestamp'].is_monotonic_increasing
        if not is_sorted:
            bad_indices = df['timestamp'].diff() < pd.Timedelta(0)
            print(df[bad_indices])

            raise ValueError(f"Timestamps are not sorted in file {file_path}")

        # Compute time differences
        df['time_diff_ms'] = df['timestamp'].diff().dt.total_seconds() * 1000

        exptected_time_diff_ms = 1 / expected_sample_rate_hz * 1000
        df.loc[0, 'time_diff_ms'] = exptected_time_diff_ms

        # Segment using time gaps
        time_gap = df['time_diff_ms'] > max_gap_s * 1000

        # print_time_gaps(df, time_gap)

        df['local_segment_id'] = time_gap.cumsum().astype(int)

        # Assign global segment IDs only to valid segments
        unique_local_segments = df['local_segment_id'].unique()

        segment_id_map = {
            local_id:
            segment_counter + i for i, local_id in enumerate(unique_local_segments)
        }

        df['segment_id'] = df['local_segment_id'].map(segment_id_map)
        segment_counter += len(unique_local_segments)

        df.drop(columns='local_segment_id', inplace=True)

        all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)

    return df, segment_counter


# Public Functions

def load_and_process_harth(dataset_path: str, global_segment_id: int):
    df, global_segment_id = load_harth_from_file_and_segment(dataset_path,
                                          global_segment_id=global_segment_id)

    df = process_harth(df)

    return df, global_segment_id
