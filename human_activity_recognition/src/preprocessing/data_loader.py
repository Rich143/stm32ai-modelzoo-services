import numpy as np
from typing import Tuple, List
import pandas as pd
from preprocessing import gravity_rotation

from load_harth import load_and_process_harth
from load_pamap2 import load_and_process_pamap2
from data_load_helpers import global_activity_name_to_id

def apply_filter_by_segment(dataset,
                            axis_cols=['x', 'y', 'z'],
                            segment_col='segment_id',
                            fs=100,
                            mean_group_delay=270,
                            verbose=False):
    """
    Applies IIR filter (in SOS form) to accelerometer data, segment by segment,
    reinitializing zi for each segment. Drops initial rows trimmed during decomposition.

    Parameters:
        dataset (pd.DataFrame): DataFrame with accelerometer columns and segment_id column.
        sos (np.ndarray): Second-order sections of IIR filter.
        axis_cols (list): Names of accelerometer axis columns.
        segment_col (str): Name of the segment ID column.
        verbose (bool): If True, prints debug info per segment.

    Returns:
        pd.DataFrame: Filtered dataset with ['x_grav', 'x_dyn', ...] columns and trimmed rows.
    """
    output_suffix_dyn = '_dyn'

    filtered_segments = []

    for seg_id, seg_df in dataset.groupby(segment_col):
        seg_data = seg_df[axis_cols].values
        orig_len = len(seg_data)

        if orig_len < 10:
            if verbose:
                print(f"[SKIP] Segment {seg_id}: too short (len = {orig_len})")
            continue

        data_dyn = gravity_rotation(seg_data, fs=fs, mean_group_delay=mean_group_delay)

        if data_dyn.shape[0] == 0:
            if verbose:
                print(f"[SKIP] Segment {seg_id}: decomposition failed or dropped all samples (original len = {orig_len})")
            continue

        drop_count = orig_len - data_dyn.shape[0]
        kept_ratio = data_dyn.shape[0] / orig_len
        if verbose:
            print(f"[INFO] Segment {seg_id}: kept {kept_ratio:.0%} ( {data_dyn.shape[0]} / {orig_len} ) rows (dropped {drop_count})")

        # Trim segment to match decomposed data
        trimmed_seg_df = seg_df.iloc[-data_dyn.shape[0]:].copy()

        for i, axis in enumerate(axis_cols):
            trimmed_seg_df[axis] = data_dyn[:, i]

        filtered_segments.append(trimmed_seg_df)

    dataset_filtered = pd.concat(filtered_segments, axis=0).reset_index(drop=True)

    return dataset_filtered

def load_datasets(dataset_names: List[str],
                  dataset_paths: List[str]) -> pd.DataFrame:
    datasets = []
    global_segment_id = 0

    for dataset_name, dataset_path in zip(dataset_names, dataset_paths):
        dataset, global_segment_id = load_individual_dataset(dataset_name=dataset_name,
                                          dataset_path=dataset_path,
                                          global_segment_id=global_segment_id)
        datasets.append(dataset)


    dataset = pd.concat(datasets, ignore_index=True)

    return dataset

def preprocess_dataset(dataset: pd.DataFrame,
                       class_names: List[str],
                       fs: int,
                       mean_group_delay: int,
                       gravity_rot_sup: bool = True) -> pd.DataFrame:
    activities_of_interest = [
        global_activity_name_to_id(activity) for activity in class_names
    ]

    # Keep only activities of interest
    dataset = dataset[dataset["activity_label"].isin(activities_of_interest)]

    if gravity_rot_sup:
        dataset = apply_filter_by_segment(dataset,
                                          axis_cols=['x', 'y', 'z'],
                                          segment_col='segment_id',
                                          fs=fs,
                                          mean_group_delay=mean_group_delay,
                                          verbose=True)

    print("Dataset stats:")
    print(f"Train size: {len(dataset)}")
    print(f"Classes: {len(class_names)}")

    return dataset

def load_individual_dataset(dataset_name: str,
                            dataset_path: str,
                            global_segment_id: int
                           ) -> Tuple[pd.DataFrame, int]:
    if dataset_name == "HARTH":
        dataset, global_segment_id = load_and_process_harth(dataset_path=dataset_path,
                                                            global_segment_id=global_segment_id)

        return dataset, global_segment_id
    elif dataset_name == "PAMAP2":
        dataset, global_segment_id = load_and_process_pamap2(dataset_path=dataset_path,
                                                   global_segment_id=global_segment_id)

        return dataset, global_segment_id
    else:
        raise NameError('Only \'PAMAP2\' and \'HARTH\' datasets supported!')

if __name__ == "__main__":
    try:
        from pandasgui import show
    except ImportError:
        show = None

    # dataset_names = ["PAMAP2"]
    dataset_names = ["PAMAP2", "HARTH"]
    # dataset_paths = [
        # "../../datasets/PAMAP2_Dataset",
    # ]
    dataset_paths = [
        "../../datasets/PAMAP2_Dataset",
        "../../datasets/harth",
    ]

    df = load_datasets(dataset_names=dataset_names,
                  dataset_paths=dataset_paths)

    df = preprocess_dataset(dataset=df,
                            class_names=["stationary", "walking", "running", "cycling"],
                            fs=50,
                            mean_group_delay=123,
                            gravity_rot_sup=True)

    if show is not None:
        show(df)
