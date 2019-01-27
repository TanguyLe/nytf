from tensorflow.python.lib.io import file_io
import pandas as pd
import numpy as np


def transfer_file_from_bucket(filename, bucket="nytf"):
    # Assumes that the file is at the root of the bucket, and to be transferred to the current directory
    with file_io.FileIO(filename, mode='wb') as f_out:
        with file_io.FileIO('gs://nytf/' + filename, mode='rb') as f_in:
            f_out.write(f_in.read())
            

def transfer_file_to_bucket(filename, bucket="nytf"):
    # Assumes that the file is at the root of the bucket, and to be transferred to the current directory
    with file_io.FileIO(filename, mode='rb') as f_in:
        with file_io.FileIO('gs://nytf/' + filename, mode='wb') as f_out:
            f_out.write(f_in.read())
        
        
# That file is created only so that you get less conflicts while moving your estimators out of utils Tristan <3
def decompose_data_to_arrays_list(data):
    if isinstance(data, pd.Series):
        col_names = [data.name]
        index = data.index
        data = [data.values]
    elif isinstance(data, pd.DataFrame):
        col_names = list(data.columns)
        index = data.index
        data = [data[name].values for name in col_names]
    elif isinstance(data, np.ndarray) and (len(data.shape) == 1 or data.shape[1] == 1):
        col_names = [0]
        index = [i for i in range(data.shape[0])]
        data = [data.reshape(data.shape[0], )]
    elif isinstance(data, np.ndarray) and len(data.shape) == 2:
        col_names = [i for i in range(data.shape[1])]
        index = [i for i in range(data.shape[0])]
        data = [data[:, i] for i in range(data.shape[1])]
    else:
        raise ValueError('The data must be Series, DataFrame or array of shape 1 or 2.')

    return data, index, col_names
