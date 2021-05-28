from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import math
import numpy as np
import random
from tqdm import trange

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from math import sqrt
from pandas import read_csv, DataFrame
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def win_start_end(ith, stride_size, data_start, train):
    # This function calculates the start and end possition of the ith window
    # based on the number i, the stride size, and the data start index 
    if train:
        window_start = data_start + (stride_size * ith)
    else:
        window_start = stride_size * ith
    window_end = window_start + window_size
    
    return window_start, window_end

def calculate_windows(data, data_start, window_size, stride_size, train):
    # This function calculates the number of windows per series (different method for train and test)
    # based on the starting index of the data, the window size, the stride size 
    # and whether the train variable is true or false
    # needs more explanation of how is it working
    time_len = data.shape[0]
    input_size = window_size - stride_size
    windows_per_series = np.full((num_series), (time_len - input_size) // stride_size)
    if train: windows_per_series -= (data_start + stride_size - 1) // stride_size
        
    return windows_per_series

def prep_data(data, covariates, data_start, train = True):
    # Calculate the number of windows per series (different method for train and test)
    windows_per_series = calculate_windows(data, data_start, window_size, stride_size, train)
    # Calculate the total number of windows being used
    total_windows = np.sum(windows_per_series)
    
    # Calculate some lengths and other variables
    time_len = data.shape[0]
    input_size = window_size - stride_size
    
    # Init results arrays 
    x_input = np.zeros((total_windows, window_size, 1 + num_covariates + 1), dtype='float32')
    label = np.zeros((total_windows, window_size), dtype='float32')
    v_input = np.zeros((total_windows, 2), dtype='float32')
    #cov = 3: ground truth + age + day_of_week + hour_of_day + num_series
    #cov = 4: ground truth + age + day_of_week + hour_of_day + month_of_year + num_series
    
    # Calculate age covariates per series and fill in the arrays with the data + covariates
    count = 0
    if not train:
        covariates = covariates[-time_len:]
    for series in trange(num_series):
        cov_age = stats.zscore(np.arange(total_time-data_start[series]))
        if train:
            covariates[data_start[series]:time_len, 0] = cov_age[:time_len-data_start[series]]
        else:
            covariates[:, 0] = cov_age[-time_len:]
        for i in range(windows_per_series[series]):
            # Calculate the start and end position of the corresponding window
            window_start, window_end = win_start_end(i, stride_size, data_start[series], train)
    
            x_input[count, 1:, 0] = data[window_start:window_end-1, series]
            x_input[count, :, 1:1+num_covariates] = covariates[window_start:window_end, :]
            x_input[count, :, -1] = series
            label[count, :] = data[window_start:window_end, series]
            nonzero_sum = (x_input[count, 1:input_size, 0]!=0).sum()
            if nonzero_sum == 0:
                v_input[count, 0] = 0
            else:
                v_input[count, 0] = np.true_divide(x_input[count, 1:input_size, 0].sum(),nonzero_sum)+1
                x_input[count, :, 0] = x_input[count, :, 0]/v_input[count, 0]
                if train:
                    label[count, :] = label[count, :]/v_input[count, 0]
            count += 1
    # Prepare the path and save the data to .npy files
    prefix = os.path.join(save_path, 'train_' if train else 'test_')
    np.save(prefix+'data_'+save_dir, x_input)
    np.save(prefix+'v_'+save_dir, v_input)
    np.save(prefix+'label_'+save_dir, label)

def gen_covariates(times, num_covariates):
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 1] = input_time.weekday()
        covariates[i, 2] = input_time.year
        covariates[i, 3] = input_time.month
    for i in range(1,num_covariates):
        covariates[:,i] = stats.zscore(covariates[:,i])
    return covariates

def visualize(data, week_start):
    x = np.arange(window_size)
    f = plt.figure()
    plt.plot(x, data[week_start:week_start+window_size], color='b')
    f.savefig("visual.png")
    plt.close()

if __name__ == '__main__':

    global save_path
    name = 'katametriseis_10_processed.csv'
    save_dir = 'elect'
    window_size = 8
    stride_size = 1
    num_covariates = 4
    train_start = '2008-07-31'
    train_end = '2013-10-31'
    test_start = '2014-01-31' #need additional 7 days as given info
    test_end = '2018-10-31'
    scale_0_1 = True

    save_path = os.path.join('data', save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    csv_path = os.path.join(save_path, name)
    if not os.path.exists(csv_path):
        print('Please provide a valid csv file or path inside the save directory!')
        sys.exit()


    data_frame = pd.read_csv(csv_path, sep=",", index_col=0, parse_dates=True, decimal='.')
    if scale_0_1:
        data_frame -= data_frame.min()  # subtract by the min
        data_frame /= data_frame.max()  # then divide by the max

    covariates = gen_covariates(data_frame[train_start:test_end].index, num_covariates)
    train_data = data_frame[train_start:train_end].values
    test_data = data_frame[test_start:test_end].values
    data_start = (train_data!=0).argmax(axis=0) # find the first nonzero value in each time series
    total_time = data_frame.shape[0] # number of time point
    num_series = data_frame.shape[1] # number of series
    prep_data(train_data, covariates, data_start)
    prep_data(test_data, covariates, data_start, train=False)