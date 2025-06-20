# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import numpy as np
import scipy.signal as signal

np.random.seed(611)


# Hi-pass IIR filter to separate the low-varying signal
# 26 Hz
A_COEFF = [1.0, -3.868656635, 5.614526749, -3.622760773, 0.8768966198]
B_COEFF = [0.9364275932, -3.745710373, 5.618565559, -3.745710373, 0.9364275932]
#
# 20 Hz
# A_COEFF = [1.0, -3.83582554, 5.52081914, -3.53353522, 0.848556]
# B_COEFF = [0.92117099, -3.68468397, 5.52702596, -3.68468397, 0.92117099]
#

def delay_signal(data, delay_samples):
    """
    Delay a signal by an integer number of samples (no interpolation).
    Pads the start with NaNs and truncates the end.

    Parameters:
    data (ndarray): Input signal of shape (N,) or (N, C)
    delay_samples (int): Integer delay in samples

    Returns:
    ndarray: Delayed signal with NaNs at the beginning.
    """
    delay_samples = int(round(delay_samples))
    if delay_samples == 0:
        return data.copy()

    if data.ndim == 1:
        # 1D signal
        return np.concatenate([np.full(delay_samples, np.nan), data[:-delay_samples]])
    else:
        # 2D signal: (N, C)
        pad = np.full((delay_samples, data.shape[1]), np.nan)
        return np.vstack([pad, data[:-delay_samples]])

def get_filter_zi(data, filt_sos):
    init_data = np.mean(data[:10, :], axis=0)

    zi_states = []

    for axis in range(3):
        zi = signal.sosfilt_zi(filt_sos)  # shape (n_sections, 2)
        zi *= init_data[axis]
        zi_states.append(zi)

    return zi_states

def lowpass_filter(data):
    """Filter signal on all axis with an low-pass filter."""

    print("[INFO] : Low-pass filtering data - Len {}, Shape {}".format(len(data), data.shape))

    # Sampling frequency
    fs = 26.0  # Hz

    # Filter specifications
    wp = 0.2 # Passband edge
    ws = 0.5 # Stopband edge
    gpass = 0.2         # Maximum passband loss (dB)
    gstop = 30        # Minimum stopband attenuation (dB)

    # Design IIR filter (Butterworth, Chebyshev, Elliptic available)
    lowpass_iir_sos = signal.iirdesign(wp, ws, gpass, gstop, ftype='butter', output='sos', fs=fs)

    zi_states = get_filter_zi(data, lowpass_iir_sos)

    data_filtered = np.empty_like(data)
    for axis in range(3):
        data_filtered[:, axis], zf = signal.sosfilt(lowpass_iir_sos, data[:, axis], zi=zi_states[axis])

    # Hardcoded group delay in passband (where passband is to -20dB)
    mean_group_delay = 68

    return data_filtered, mean_group_delay


def decompose_dyn(data, A_COEFF=A_COEFF, B_COEFF=B_COEFF):
    """ separate acceleration in low-varying and dynamic component """
    data_g, mean_group_delay = lowpass_filter(data)

    if data_g.shape[0] <= mean_group_delay:
        # Data too short to decompose
        print("[INFO] : Decomposing data - Data too short to decompose. Shape {}, group delay {}".format(data_g.shape, mean_group_delay))
        return np.empty((0,)), np.empty((0,))

    data_g_delayed = delay_signal(data_g, mean_group_delay)

    print("[INFO] : Decomposing data - data_g_delayed shape {}, data shape {}".format(data_g_delayed.shape, data.shape))

    data_dyn = data - data_g_delayed

    valid_start = np.argmax(~np.isnan(data_dyn[:, 0]))  # Assumes X, Y, Z are all aligned
    data_dyn = data_dyn[valid_start:]
    data_g = data_g[valid_start:]

    print("[INFO] : Decomposing data - dropped {} samples, new len {}".format(valid_start, data_dyn.shape[0]))

    return data_g, data_dyn


def colwise_dot(lhs, rhs):
    """ compute the dot product column by column"""
    return np.sum(lhs * rhs, axis=1)


def gravity_rotation(data, A_COEFF=A_COEFF, B_COEFF=B_COEFF):

    # Rotate the coordinate system in order to have z pointing in the gravity direction
    #
    data_g, data_dyn = decompose_dyn(data, A_COEFF, B_COEFF)

    if data_g.shape[0] == 0:
        # Data was too short to decompose
        return np.empty((0,))

    # Normalize gravity
    data_g = data_g / np.sqrt(colwise_dot(data_g, data_g))[:, np.newaxis]

    # Cross product between z and g versors
    axis = np.concatenate(
        (-data_g[:, 1:2], data_g[:, 0:1], np.zeros((data_g.shape[0], 1))), axis=1)
    sin, cos = np.sqrt(colwise_dot(axis, axis))[:, np.newaxis], -data_g[:, 2:3]

    # Normalize rotation axis and handle degenerate configurations
    with np.errstate(divide='ignore', invalid='ignore'):
        axis = np.true_divide(axis, sin)
        # Set rotation to 0 if gravity aligned to z
        axis[axis == np.inf] = 0.0
        axis = np.nan_to_num(axis)

    data_dyn = data_dyn * cos + np.cross(axis, data_dyn) * sin + \
        axis * colwise_dot(axis, data_dyn)[:, np.newaxis] * (1.0 - cos)
    # print( data_dyn[0,:])
    return data_dyn


if __name__ == '__main__':
    # #
    # dataExample = [[1, 0, 0], [1, 0, 0], [0, 0, 1]]
    # dataExample = np.array(dataExample)
    # print(dataExample)
    # A_COEFF = [1.0, -3.868656635, 5.614526749, -3.622760773, 0.8768966198]
    # B_COEFF = [0.9364275932, -3.745710373,
               # 5.618565559, -3.745710373, 0.9364275932]
    # print(gravity_rotation(dataExample, A_COEFF, B_COEFF))



    import matplotlib.pyplot as plt


    # Parameters
    fs = 26               # Sampling frequency (Hz)
    duration_sec = 500    # Total duration in seconds
    n_samples = duration_sec * fs
    f_walk = 1.5          # Walking frequency (Hz)

    # Time vector
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # Simulate gravity vector
    gravity = np.full((n_samples, 3), [0,0,-9.81])

    # Dynamic motion (Walking)
    walking_x = np.zeros_like(t)
    walking_y = np.zeros_like(t)
    walking_z = 2 * np.sin(2 * np.pi * f_walk * t)
    walking = np.column_stack((walking_x, walking_y, walking_z))

    # # Add rotation to gravity and walking (eg to simulate phone rotating in pocket)

    # pitch_freq=0.01
    # pitch_mag_deg=400 
    # roll_freq=0.05
    # roll_mag_deg=0
    # yaw_freq=0.025
    # yaw_mag_deg=0
    # rotation_noise_std=0

    # rotated_walking = rotate_signal_with_orientation(
        # walking,
        # t,
        # pitch_freq=pitch_freq,
        # roll_freq=roll_freq,
        # yaw_freq=yaw_freq,
        # pitch_mag_deg=pitch_mag_deg,
        # roll_mag_deg=roll_mag_deg,
        # yaw_mag_deg=yaw_mag_deg,
        # noise_std=rotation_noise_std)

    # rotated_gravity = rotate_signal_with_orientation(
        # gravity,
        # t,
        # pitch_freq=pitch_freq,
        # roll_freq=roll_freq,
        # yaw_freq=yaw_freq,
        # pitch_mag_deg=pitch_mag_deg,
        # roll_mag_deg=roll_mag_deg,
        # yaw_mag_deg=yaw_mag_deg,
        # noise_std=rotation_noise_std)

    # # Add noise
    # burst_probability = 0.01
    # burst_magnitude = 3.0
    # bursts = np.random.rand(*gravity.shape) < burst_probability
    # burst_noise = bursts * np.random.normal(0.0, burst_magnitude, gravity.shape)
    # background_noise = np.random.normal(0.0, 0.2, gravity.shape)
    # noise = burst_noise + background_noise
    noise = np.zeros_like(walking)

    # Final accelerometer signal
    # accel_data = rotated_walking + rotated_gravity + noise
    accel_data = walking + gravity + noise

    # Plot Before
    plt.figure(figsize=(10, 4))
    for i, label in enumerate(['X', 'Y', 'Z']):
        plt.plot(t[:1000], accel_data[:1000, i], label='Accel Data ' + label)
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Accel data in Device Frame (First 50s)')
    plt.legend()
    plt.grid(True)
    plt.show()

    data_g, data_dyn = decompose_dyn(accel_data)
    data_dyn = gravity_rotation(accel_data)

    # Plot After
    plt.figure(figsize=(10, 4))
    for i, label in enumerate(['X', 'Y', 'Z']):
        plt.plot(t[:1000], data_dyn[:1000, i], label='Data Dyn ' + label)
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Data Dyn rotated')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    for i, label in enumerate(['X', 'Y', 'Z']):
        plt.plot(t[:1000], data_g[:1000, i], label='Data Dyn ' + label)
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Data g rotated')
    plt.legend()
    plt.grid(True)
    plt.show()

    # plot_all_axes_fft(rotated_gravity, rotated_walking, fs, Nf, zero_padding_factor=4, db_scale=True)
    #plot_vector_trajectory_3d(gravity, step=10)

    #animate_vector_trajectory(rotated_gravity + rotated_walking, step=5)
