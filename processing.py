#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time

# ToDo: Add a visualization, i e. exporting corresponding video frames to the sound.


def array_image(data_array: np.array):
    img = Image.fromarray(data_array)
    img.show()


def powspace(start, stop, num, power=2):
    # Stolen from https://stackoverflow.com/a/53345687
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    return np.power(np.linspace(start, stop, num=num), power)


def scale_minmax(data: np.array):
    """Scales a numpy array to lie between 0 and 1.
    Formula from: https://stats.stackexchange.com/a/178629"""
    minval = np.min(data)
    maxval = np.max(data)
    out = (data - minval) / (maxval - minval)
    return out


def scale_audio(data: np.array):
    """Scales a signal to lie between -1 and +1.
    Formula from: https://stats.stackexchange.com/a/178629"""
    minval = np.min(data)
    maxval = np.max(data)
    out = 2 * ((data - minval) / (maxval - minval)) - 1
    return out


def stretch_array(data, target_col_count):
    """
    Stretches an array out to have a specified target column amount.
    Fills empty spaces in between with zeroes.
    """
    assert data.shape[1] < target_col_count, 'Target array width is smaller or equal than source data. Can not stretch.'
    len_data = data.shape[1]
    # First -1 because you start indexing at 0 (but length starts at 1)
    # Second -1 because you want also one element in the first spot
    step_size = (target_col_count - 1) / (len_data - 1)
    # Calculate the new indeces of the spread out data.
    indices_float = np.array([step_size * i for i in range(data.shape[1])])
    indices = indices_float.astype(int)
    # Prepare output data
    out = np.zeros((data.shape[0], target_col_count))
    # Write old data into new array at corresponding indices
    for i_old, i_new in enumerate(indices):
        out[:, i_new] = data[:, i_old]
    return out


def squeeze_array(data, target_col_count, verbose=False):
    assert data.shape[1] > target_col_count
    line_count = len(data)
    batches = np.array_split(data, target_col_count, axis=1)
    batch_count = len(batches)
    data_squeezed = np.zeros(line_count)
    for i, array in enumerate(batches):
        if verbose:
            if i % 100 == 0:
                prog = np.around((i / batch_count) * 100, 2)

                print(f'\rBatch averaging: {prog} %', end='')
        batch_avg = array.mean(axis=1)
        data_squeezed = np.vstack((data_squeezed, batch_avg))
    if verbose: print()
    data_squeezed = data_squeezed[1:]
    data_squeezed = data_squeezed.T
    return data_squeezed


def align_data(data, hz_vector, interpolate=False, verbose=False):
    target_col_count = len(hz_vector)
    if data.ndim == 2:
        # Spread the data points when the data is shorter that the Hz range:
        if data.shape[1] < target_col_count:
            return stretch_array(data, target_col_count)
        # Squeeze data if it's wider than the Hz range:
        elif data.shape[1] > target_col_count:
            return squeeze_array(data, target_col_count, verbose)
        # Just forward the data if it already fits:
        elif data.shape[1] == target_col_count:
            return data
        else:
            raise Exception('Something seems to be wrong with the input data dimensionality.')


def signal_from_line(line, hz_vector, sample_rate=44100, duration_s=0.2, x_values=None, verbose=False):
    # ToDo: Normalization
    assert len(line) == len(hz_vector)
    sig_len = int(sample_rate * duration_s)
    frequencies = np.empty(sig_len)
    if x_values is None:
        x_values = np.linspace(0, 2, sample_rate)
        x_values = x_values[-int(sample_rate * duration_s):]
    for Hz, intensity in zip(hz_vector, line):
        if verbose:
            print(f'Processing frequency: {Hz}.')
        if intensity > 0:
            signal = np.sin(x_values * np.pi * Hz) * (1 / intensity)
            frequencies = np.vstack((frequencies, signal))
    # print('frequencies:', type(frequencies), frequencies.shape)
    signal = frequencies.sum(axis=0)
    # plt.figure('Frequencies')
    # plt.plot(x_values, frequencies.T)
    # plt.xlim((0, 0.1))
    # plt.draw()
    # plt.figure('Signal')
    # plt.plot(x_values, signal)
    # plt.xlim((0, 0.1))
    # plt.show()
    return signal


def signals_from_array(data_array, hz_vector, sample_rate=44100, duration_s=0.2, verbose=False):
    assert data_array.ndim == 2
    assert data_array.shape[1] == len(hz_vector)
    # Create an empty array to fill with the signal for each line, already cut to duration
    # Adjust signal length (or rather, its x values)
    sig_len = int(sample_rate * duration_s)
    line_count = len(data_array)
    signals = np.empty(sig_len)
    x_values = np.linspace(0, 2 * duration_s, sample_rate)
    x_values = x_values[-int(sample_rate * duration_s):]
    for i, line in enumerate(data_array):
        if verbose and i % 5 == 0:
            prog = np.around(i / line_count * 100, 2)
            print(f'\rExtracting signals: {prog} % (signal {i}).    ', end='')
        signal = signal_from_line(line, hz_vector, sample_rate, duration_s, x_values)
        signals = np.vstack((signals, signal))
    if verbose: print()
    signals = signals[1:]
    return signals