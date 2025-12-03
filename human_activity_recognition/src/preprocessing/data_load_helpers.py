# Central lookup: name (lowercase) → ID
_ACTIVITY_NAME_TO_GLOBAL_ACTIVITY_ID = {
    'walking': 1,
    'running': 2,
    'cycling': 3,
    'stationary': 4,
    'stairs': 5,
    'unknown': 0
}

# Reverse: ID → name
_GLOBAL_ACTIVITY_ID_TO_NAME = {v: k for k, v in _ACTIVITY_NAME_TO_GLOBAL_ACTIVITY_ID.items()}


def global_activity_name_to_id(name: str) -> int:
    """
    Convert a activity name to global activity ID.
    Raises KeyError if the name is not found.
    """
    key = name.lower().strip()
    if key not in _ACTIVITY_NAME_TO_GLOBAL_ACTIVITY_ID:
        raise KeyError(f"Activity name not found: '{name}'")
    return _ACTIVITY_NAME_TO_GLOBAL_ACTIVITY_ID[key]


def global_activity_id_to_name(id: int) -> str:
    """
    Convert an activity ID to its name.
    Raises KeyError if the ID is not found.
    """
    if id not in _GLOBAL_ACTIVITY_ID_TO_NAME:
        raise KeyError(f"Activity ID not found: {id}")
    return _GLOBAL_ACTIVITY_ID_TO_NAME[id]

def copy_accel_to_xyz(df, source='chest_acc16'):
    """
    Copies the specified accelerometer data (hand, chest, or ankle) to generic 'x', 'y', 'z' columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        source (str): Which set of accelerometer data to copy.

    Returns:
        pd.DataFrame: Modified DataFrame with new 'x', 'y', 'z' columns copied from the source set.
    """
    df = df.copy()
    df['x'] = df[f'{source}_x']
    df['y'] = df[f'{source}_y']
    df['z'] = df[f'{source}_z']
    return df

def fill_nans(df):
    # Interpolate to fill NaNs
    df[["x", "y", "z"]] = (
    df.groupby("segment_id", group_keys=False)[["x", "y", "z"]]
    .apply(lambda group: group.interpolate())
    .reset_index(drop=True)
    )

    # Fill any remaining NaNs with 0
    df = df.fillna(0)

    return df
