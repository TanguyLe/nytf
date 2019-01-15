import pandas as pd
import numpy as np


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
