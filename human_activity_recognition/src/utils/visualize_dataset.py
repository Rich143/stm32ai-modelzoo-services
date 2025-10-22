import os
from typing import Tuple, List
import sys
from glob import glob
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../preprocessing'))

from data_loader import (
    group_activity_ids,
    load_pamap2_from_file_and_segment,
    copy_accel_to_xyz,
    fill_nans,
    group_activity_ids,
    class_name_to_id,
)

def load_pamap2_and_process(dataset_path: str, class_names: List[str], max_gap_s: float = 0.1):
    dataset = load_pamap2_from_file_and_segment(dataset_path, max_gap_s)

    dataset = copy_accel_to_xyz(dataset, source='chest')

    dataset = dataset.drop(['hand_acc16_x', 'hand_acc16_y', 'hand_acc16_z', 'ankle_acc16_x', 'ankle_acc16_y', 'ankle_acc16_z'], axis=1)

    dataset = dataset[['Arrival_Time', 'x', 'y', 'z', 'Activity_Label', 'User', 'segment_id']]

    # Fill NaNs via linear interpolation
    dataset = fill_nans(dataset)

    # Group activities
    group_activity_ids(dataset)

    # Keep only activities of interest
    activities_of_interest = [class_name_to_id(activity) for activity in class_names]
    dataset = dataset[dataset["Activity_Label"].isin(activities_of_interest)]

    return dataset

class_names = ["stationary", "walking", "running", "cycling"]
df = load_pamap2_and_process("../datasets/PAMAP2_Dataset/", class_names, max_gap_s=0.1)

result = df.groupby("segment_id", group_keys=False)[["User", "Activity_Label"]].first().reset_index()
print(result)

result2 = df.groupby("Activity_Label", group_keys=False)[["User"]].nunique().reset_index().rename(columns={"User": "num_users"})
print(result2)

segments_per_user_activity = (
    df.groupby(["User", "Activity_Label"])["segment_id"]
    .nunique()
    .reset_index()
    .rename(columns={"segment_id": "num_segments"})
)
print(segments_per_user_activity)

rows_per_user_activity = (
    df.groupby(["User", "Activity_Label"])
    .size()
    .reset_index()
    # .rename(columns={"segment_id": "num_segments"})
)
print(rows_per_user_activity)
