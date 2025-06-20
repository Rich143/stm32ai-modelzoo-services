# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import pickle
import scipy.io
from scipy import stats
from scipy.signal import butter
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from typing import Tuple, List
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from preprocessing import gravity_rotation
import statistics
import os
from glob import glob


def read_pkl_dataset(pkl_file_path: str,
                     class_names: List[str]):
    '''
    read_pkl_dataset reads a pkl dataset and returns a pandas DataFrame
    arg:
        pkl_file_path: path to the pkl file to be read
        class_names: a list of strings containing the names of the activities
    returns:
        dataset: a pandas dataframe containing all the data combined in a single object
                 containing four columns 'x', 'y', 'z', 'Activity_Labels'.
    '''
    # initialize the script
    class_id = 0
    file_nr = 0
    my_array = []

    # read pkl dataset
    dataset = pd.read_pickle(pkl_file_path)

    # list with nr files for every activity
    nr_files_per_class = []
    # we know there are only five activities in the dataset
    #  with labels from 0->4 so let us count nr of files for every activity
    for lbl in range(len(class_names)):
        nr_files_per_class.append(dataset['act'].count(lbl))

    # acceleration data in the dataset
    array_data = dataset['acc']

    # now let us get data for every activity one by one
    for nr_files in nr_files_per_class:
        # for every occurance of the label
        for i in range(file_nr, file_nr + nr_files):
            my_array.append([])
            for j in range(array_data[i].shape[0]):
                # for every sample in the file
                my_array[i].append([])
                my_array[i][j].extend(array_data[i][j])
                my_array[i][j].append(class_id)
        file_nr += nr_files
        class_id += 1

    # preparing a vertical stack for the dataset
    my_array = np.vstack(my_array[:])

    # creating a data frame withonly four columns 
    # 'x', 'y', 'z', and 'Activity_Label' to be 
    # consistent with WISDM data
    columns = ['x', 'y', 'z', 'Activity_Label']
    my_dataset = pd.DataFrame(my_array, columns=columns)

    # replace activity code with activity labels to be consistent with WISDM dataset
    my_dataset['Activity_Label'] = [str(num).replace(str(num),
                                                     class_names[int(num)])
                                    for num in my_dataset['Activity_Label']]
    return my_dataset


def clean_csv(file_path):
    '''
    This function is specifically written for WISDM dataset.
    This function takes as an input the path to the csv file,
    cleans it and rewrites the cleaned data in the same file.
    args:
        file_path: path of the csv file to be cleaned.
    '''
    # read data file
    with open(file_path, "rt", encoding='utf-8') as fin:
        # read file contents to string
        data = fin.read()

    # fix all problems by replacing ';\n' with '\n's etc
    data = data.replace(';\n', '\n').replace(
        '\n;\n', '\n').replace(';', '\n').replace(',\n', '\n')

    # open the data file in write mode
    with open(file_path, "wt", encoding='utf-8', newline='') as f_out:
        # overrite the data file with the correct data
        f_out.write(data)

    # close the file
    fin.close()


def get_segment_indices(data_column: List, win_len: int):
    '''
    this function gets the start and end indices for the segments
    args:
        data_column: indices
        win_len: segment length
    yields:
        init: int
        end: int
    '''
    # get segment indices to window the data into overlapping frames
    init = 0
    while init < len(data_column):
        yield int(init), int(init + win_len)
        init = init + win_len


def get_data_segments(dataset: pd.DataFrame,
                      seq_len: int) -> Tuple[np.ndarray, np.ndarray]:

    '''
    This function segments the data into (non)overlaping frames
    args:
        dataset: a dataframe containing 'x', 'y', 'z', and 'Activity_Label' columns
    returns:
        A Tuple of np.ndarray containing segments and labels
    '''
    data_indices = dataset.index.tolist()
    n_samples = len(data_indices)
    segments = []
    labels = []

    # need the following variable for tqdm to show the progress bar
    num_segments = int(np.floor((n_samples - seq_len) / seq_len)) + 1

    # creating segments until the get_segment_indices keep on yielding the start and end of the segments
    for (init, end) in tqdm(get_segment_indices(data_indices, seq_len),
                            unit=' segments', desc='Segments built ',
                            total=num_segments):

        # check if the nr of remaing samples are enough to create a frame
        if end < n_samples:
            segments.append(np.transpose([dataset['x'].values[init: end],
                            dataset['y'].values[init: end],
                            dataset['z'].values[init: end]]))

            # use the label which occured the most in the frame
            # print(labels, statistics.mode(dataset['Activity_Label'][init: end]))
            labels.append(statistics.mode(dataset['Activity_Label'][init: end]))

    # converting the segments from list to numpy array
    segments = np.asarray(segments, dtype=np.float)
    segments = segments.reshape(segments.shape[0], segments.shape[1],
                                segments.shape[2], 1)
    labels = np.asarray(labels)
    return segments, labels


def preprocess_data_new(data: np.ndarray,
                        gravity_rot_sup: bool = True,
                        normalization: bool = False) -> np.ndarray:
    '''
    Preprocess the data
    '''
    if gravity_rot_sup:
        # choose a sampling frequency
        # hardcoding to have code equivalency with C-implementation
        f_sample = 26
        # create a copy of data to avoid overwriting the passed dataframe
        data_copy = data.copy()
        # create highpass filter to remove dc components
        f_cut = 0.4
        filter_type = 'highpass'
        f_norm = f_cut / f_sample
        num, den = butter(4, f_norm, btype=filter_type)
        data_x = data_copy[data_copy.columns[:3]]
        # print('going in gravity_rotation filter ')
        data_copy[ data_copy.columns[:3]] = gravity_rotation(np.array(data_x, 
                                                                      dtype=float),
                                                             A_COEFF = den,
                                                             B_COEFF = num )
        return data_copy
    else:
        return data


def preprocess_data(data: np.ndarray,
                    gravity_rot_sup: bool,
                    normalization: bool) -> np.ndarray:

    '''
    Preprocess the data
    '''

    if gravity_rot_sup:
        # choose a sampling frequency
        # hardcoding to have code equivalency with C-implementation
        f_sample = 26
        # create a copy of data to avoid overwriting the passed dataframe
        data_copy = data.copy()
        # create highpass filter to remove dc components
        f_cut = 0.4
        filter_type = 'highpass'
        f_norm = f_cut / f_sample
        num, den = butter(4, f_norm, btype=filter_type)

        # preprocess the dataset by finding and rotating the gravity axis
        for i in range(data_copy.shape[0]):
            data_copy[i, :, :, 0] = gravity_rotation(np.array(data_copy[i, :, :, 0],
                                                              dtype=float),
                                                     A_COEFF=den, B_COEFF=num)
        return data_copy
    if normalization:
        data_copy = data.copy()

        # Reshape the array to treat each seq_len x 3 window as a seq_len x 1 window
        data_copy = np.reshape(data_copy, (data_copy.shape[0], data_copy.shape[1], -1))

        # Calculate the mean and standard deviation of each seq_len x 1 window
        mean = np.mean(data_copy, axis=(1, 2), keepdims=True)
        std = np.std(data_copy, axis=(1, 2), keepdims=True)

        # Normalize each seq_len x 1 window
        data_norm = (data_copy - mean) / std

        # Reshape the normalized array back to its original shape
        data_norm = np.reshape(data_norm, (data_norm.shape[0], data_norm.shape[1], 3, 1))
        return data_norm

    return data


def load_pamap2_from_file_and_segment(dataset_path: str,
                                      max_segment_gap_ms: float = 0.1):
    """
    Load and combine PAMAP2 protocol dataset, keeping only timestamp, subject_id, activity_id,
    and 16g accelerometer data. Filters out activity_id == 0 and segments the data into
    time-continuous chunks per subject and activity.

    Adds a time_diff_ms column (milliseconds between samples) and a globally unique segment_id.

    Parameters:
        protocol_folder (str): Path to the 'protocol' folder containing the .dat files.
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
    file_pattern = os.path.join(protocol_folder, 'subject1*.dat')
    segment_counter = 0  # global segment ID counter

    for file_path in sorted(glob(file_pattern)):
        subject_id = int(os.path.basename(file_path)[7:10])
        df = pd.read_csv(file_path, sep=' ', header=None, usecols=usecols, names=col_names)
        df['User'] = subject_id

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

    return df


def copy_accel_to_xyz(df, source='chest'):
    """
    Copies the specified accelerometer data (hand, chest, or ankle) to generic 'x', 'y', 'z' columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame from `load_pamap2_filtered_segmented`.
        source (str): Which set of accelerometer data to copy. Options: 'hand', 'chest', 'ankle'.

    Returns:
        pd.DataFrame: Modified DataFrame with new 'x', 'y', 'z' columns copied from the source set.
    """
    valid_sources = ['hand', 'chest', 'ankle']
    if source not in valid_sources:
        raise ValueError(f"source must be one of {valid_sources}")

    df = df.copy()
    df['x'] = df[f'{source}_acc16_x']
    df['y'] = df[f'{source}_acc16_y']
    df['z'] = df[f'{source}_acc16_z']
    return df

def fill_nans(df):
    # Interpolate to fill NaNs
    df[["x", "y", "z"]] = (
    df.groupby("segment_id")[["x", "y", "z"]]
    .apply(lambda group: group.interpolate())
    .reset_index(drop=True)
    )
    
    # Fill any remaining NaNs with 0
    df = df.fillna(0)
    
    return df

def class_name_to_id(name):
    """
    Convert a PAMAP2 activity name (case-insensitive) to its corresponding activity ID.

    Parameters:
        name (str): Activity name (e.g., 'sitting', 'walking')

    Returns:
        int: Corresponding activity ID, or None if not found
    """
    activity_map = {
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
        'other': 0
    }

    return activity_map.get(name.lower().strip())


def load_pamap2(dataset_path: str,
               class_names: List[str],
               input_shape: Tuple,
               gravity_rot_sup: bool,
               normalization: bool,
               val_split: float,
               test_split: float,
               seed: int,
               batch_size: int,
               to_cache: bool = False):
    """
    Loads the pamap2 dataset and return two TensorFlow datasets for training and validation.
    """

    dataset = load_pamap2_from_file_and_segment(dataset_path)

    # Use the chest accelerometer data
    dataset = copy_accel_to_xyz(dataset, source='chest')

    # Remove the ankle and hand data
    dataset = dataset.drop(['hand_acc16_x', 'hand_acc16_y', 'hand_acc16_z', 'ankle_acc16_x', 'ankle_acc16_y', 'ankle_acc16_z'], axis=1)

    # Fill NaNs via linear interpolation
    dataset = fill_nans(dataset)

    # Keep only activities of interest
    activities_of_interest = [class_name_to_id(activity) for activity in class_names]
    df = df[df["Activity_Label"].isin(activities_of_interest)]

    # clean_csv(dataset_path)

    # # read all the data in csv 'WISDM_ar_v1.1_raw.txt' into a dataframe
    # #  called dataset
    # columns = ['User', 'Activity_Label', 'Arrival_Time',
               # 'x', 'y', 'z']  # headers for the columns

    # dataset = pd.read_csv(dataset_path, header=None,
                          # names=columns, delimiter=',')

    # # removing the ; at the end of each line and casting the last variable
    # #  to datatype float from string
    # dataset['z'] = [float(str(char).replace(";", "")) for char in dataset['z']]

    # # remove the user column as we do not need it
    # dataset = dataset.drop('User', axis=1)

    # # as we are workign with numbers, let us replace all the empty columns
    # # entries with NaN (not a number)
    # dataset.replace(to_replace='null', value=np.NaN)

    # # remove any data entry which contains NaN as a member
    # dataset = dataset.dropna(axis=0, how='any')
    if len(class_names) == 4:
        dataset['Activity_Label'] = ['Stationary' if activity == 'Standing' or activity == 'Sitting' else activity
                                     for activity in dataset['Activity_Label']]
        dataset['Activity_Label'] = ['Stairs' if activity == 'Upstairs' or activity == 'Downstairs' else activity
                                     for activity in dataset['Activity_Label']]

    dataset = preprocess_data(dataset, gravity_rot_sup, normalization)

    # removing the columns for time stamp and rearranging remaining columns
    dataset = dataset[['x', 'y', 'z', 'Activity_Label']]
    segments, labels = get_data_segments(dataset=dataset,
                                         seq_len=input_shape[0])

    labels = to_categorical([class_names.index(label)
                            for label in labels], num_classes=len(class_names))

    # split data into train and test
    train_x, test_x, train_y, test_y = train_test_split(segments, labels,
                                                        test_size=test_split,
                                                        shuffle=True,
                                                        random_state=seed)
    # split data into train and valid
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y,
                                                          test_size=val_split,
                                                          shuffle=True,
                                                          random_state=seed)
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    if batch_size is None:
        batch_size=32
    train_ds = train_ds.shuffle(train_x.shape[0],
                                reshuffle_each_iteration=True,
                                seed=seed).batch(batch_size)
    valid_ds = tf.data.Dataset.from_tensor_slices((valid_x, valid_y))
    valid_ds = valid_ds.shuffle(valid_x.shape[0],
                                reshuffle_each_iteration=True,
                                seed=seed).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_ds = test_ds.shuffle(test_x.shape[0],
                              reshuffle_each_iteration=True,
                              seed=seed).batch(batch_size)
    if to_cache:
        train_ds = train_ds.cache()
        valid_ds = valid_ds.cache()
        test_ds = test_ds.cache()
    return train_ds, valid_ds, test_ds


def load_wisdm(dataset_path: str,
               class_names: List[str],
               input_shape: Tuple,
               gravity_rot_sup: bool,
               normalization: bool,
               val_split: float,
               test_split: float,
               seed: int,
               batch_size: int,
               to_cache: bool = False):
    """
    Loads the wisdm dataset and return two TensorFlow datasets for training and validation.
    """
    clean_csv(dataset_path)

    # read all the data in csv 'WISDM_ar_v1.1_raw.txt' into a dataframe
    #  called dataset
    columns = ['User', 'Activity_Label', 'Arrival_Time',
               'x', 'y', 'z']  # headers for the columns

    dataset = pd.read_csv(dataset_path, header=None,
                          names=columns, delimiter=',')

    # removing the ; at the end of each line and casting the last variable
    #  to datatype float from string
    dataset['z'] = [float(str(char).replace(";", "")) for char in dataset['z']]

    # # remove the user column as we do not need it
    # dataset = dataset.drop('User', axis=1)

    # as we are workign with numbers, let us replace all the empty columns
    # entries with NaN (not a number)
    dataset.replace(to_replace='null', value=np.NaN)

    # remove any data entry which contains NaN as a member
    dataset = dataset.dropna(axis=0, how='any')
    if len(class_names) == 4:
        dataset['Activity_Label'] = ['Stationary' if activity == 'Standing' or activity == 'Sitting' else activity
                                     for activity in dataset['Activity_Label']]
        dataset['Activity_Label'] = ['Stairs' if activity == 'Upstairs' or activity == 'Downstairs' else activity
                                     for activity in dataset['Activity_Label']]

    dataset = preprocess_data(dataset, gravity_rot_sup, normalization)

    # removing the columns for time stamp and rearranging remaining columns
    dataset = dataset[['x', 'y', 'z', 'Activity_Label']]
    segments, labels = get_data_segments(dataset=dataset,
                                         seq_len=input_shape[0])

    labels = to_categorical([class_names.index(label)
                            for label in labels], num_classes=len(class_names))

    # split data into train and test
    train_x, test_x, train_y, test_y = train_test_split(segments, labels,
                                                        test_size=test_split,
                                                        shuffle=True,
                                                        random_state=seed)
    # split data into train and valid
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y,
                                                          test_size=val_split,
                                                          shuffle=True,
                                                          random_state=seed)
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    if batch_size is None:
        batch_size=32
    train_ds = train_ds.shuffle(train_x.shape[0],
                                reshuffle_each_iteration=True,
                                seed=seed).batch(batch_size)
    valid_ds = tf.data.Dataset.from_tensor_slices((valid_x, valid_y))
    valid_ds = valid_ds.shuffle(valid_x.shape[0],
                                reshuffle_each_iteration=True,
                                seed=seed).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_ds = test_ds.shuffle(test_x.shape[0],
                              reshuffle_each_iteration=True,
                              seed=seed).batch(batch_size)
    if to_cache:
        train_ds = train_ds.cache()
        valid_ds = valid_ds.cache()
        test_ds = test_ds.cache()
    return train_ds, valid_ds, test_ds


def load_mobility_v1(train_path: str,
                     test_path: str,
                     validation_split: float,
                     class_names: List[str],
                     input_shape: Tuple[int],
                     gravity_rot_sup: bool,
                     normalization: bool,
                     batch_size: int,
                     seed: int,
                     to_cache: bool = False):
    """
    Loads the mobility dataset and return two TensorFlow datasets for training, validation and test.
    """
    train_dataset = read_pkl_dataset(train_path, class_names)
    test_dataset = read_pkl_dataset(test_path, class_names)

    train_dataset[train_dataset.columns[:3]] = train_dataset[train_dataset.columns[:3]] * 9.8
    test_dataset[test_dataset.columns[:3]] = test_dataset[test_dataset.columns[:3]] * 9.8

    print('[INFO] : Building train segments!')
    train_segments, train_labels = get_data_segments(dataset=train_dataset,
                                                     seq_len=input_shape[0])
    print('[INFO] : Building test segments!')
    train_segments = preprocess_data(train_segments, gravity_rot_sup, normalization)
    train_labels = to_categorical([class_names.index(label)
                                  for label in train_labels], num_classes=len(class_names))
    test_segments, test_labels = get_data_segments(dataset=test_dataset,
                                                   seq_len=input_shape[0])
    test_segments = preprocess_data(test_segments, gravity_rot_sup, normalization)
    test_labels = to_categorical([class_names.index(label)
                                  for label in test_labels], num_classes=len(class_names))

    train_x, valid_x, train_y, valid_y = train_test_split(train_segments, train_labels,
                                                          test_size=validation_split,
                                                          shuffle=True,
                                                          random_state=seed)
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_ds = train_ds.shuffle(train_x.shape[0],
                                reshuffle_each_iteration=True,
                                seed=seed).batch(batch_size)
    valid_ds = tf.data.Dataset.from_tensor_slices((valid_x, valid_y))
    valid_ds = valid_ds.shuffle(valid_x.shape[0],
                                reshuffle_each_iteration=True,
                                seed=seed).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_segments, test_labels))
    test_ds = test_ds.shuffle(test_segments.shape[0],
                              reshuffle_each_iteration=True,
                              seed=seed).batch(batch_size)

    if to_cache:
        train_ds = train_ds.cache()
        test_ds = test_ds.cache()
        valid_ds = valid_ds.cache()
    return train_ds, valid_ds, test_ds


def load_dataset(dataset_name: str = None,
                 training_path: str = None,
                 validation_path: str = None,
                 validation_split: float = 0.2,
                 test_path: str = None,
                 test_split: float = 0.2,
                 class_names: List[str] = None,
                 input_shape: tuple[int] = None,
                 gravity_rot_sup: bool = True,
                 normalization: bool = False,
                 batch_size: int = None,
                 seed: int = None,
                 to_cache: bool = False) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Loads the dataset from the given files and returns training,
    validation, and test tf.data.Datasets.
    The datasets can have different structure. 
        WISDM: it has all the data in a single csv file and has to be prepared:
        mobility_v1: has .dat files and contains seperate train and test files
        
    Args:
        dataset_name (str): Name of the dataset to load.
        training_path (str): Path to the file containing the training dataset.
        validation_path (str): Path to the file containing the validation dataset.
        test_path (str): Path to the file containing the test dataset.
        validation_split (float): Fraction of the data to use for validation.
        test_split (float): Fraction of the data to use for validation.
        class_names (list[str]): List of class names to use for confusion matrix.
        input_shape (tuple[int]): shape of the input (width, 3,1) of accelerometer segments
        gravity_rot_sup (bool): a flag to implement gravity rotation and supression
        normalization (bool): a flag to implement standard normalization on accelerometer frames
        batch_size (int): Batch size to use for the datasets.
        seed (int): Seed to use for shuffling the data.
        to_cache (boolean): flag to cache the tensorflow_datasets

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        Training, Validation, Test datasets.
    """
    if dataset_name == "wisdm":
        # Load wisdm dataset
        train_ds, val_ds, test_ds = load_wisdm(dataset_path=training_path,
                                               class_names=class_names,
                                               input_shape=input_shape,
                                               gravity_rot_sup=gravity_rot_sup,
                                               normalization=normalization,
                                               val_split=validation_split,
                                               test_split=test_split,
                                               seed=seed,
                                               batch_size=batch_size,
                                               to_cache=to_cache)
    elif dataset_name == "mobility_v1":
        train_ds, val_ds, test_ds = load_mobility_v1(train_path=training_path,
                                                     test_path=test_path,
                                                     validation_split=validation_split,
                                                     class_names=class_names,
                                                     input_shape=input_shape,
                                                     gravity_rot_sup=gravity_rot_sup,
                                                     normalization=normalization,
                                                     batch_size=batch_size,
                                                     seed=seed,
                                                     to_cache=to_cache)
    else:
        raise NameError('Only \'wisdm\' and \'mobility_v1\' datasets are supported!')

    return train_ds, val_ds, test_ds
