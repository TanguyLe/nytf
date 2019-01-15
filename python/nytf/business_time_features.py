import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .utils2 import decompose_data_to_arrays_list


class BusinessFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, hour_col, night_hours=True, peak_hours=True):
        self.hour_col = hour_col
        self.night_hours = night_hours
        self.peak_hours = peak_hours

    def fit(self, X):
        return self

    def transform(self, data):
        data, index, col_names = decompose_data_to_arrays_list(data=data)
        hour_col = data[col_names.index(self.hour_col)]

        night_hour = (hour_col > 20) | (hour_col < 6)
        peak_hour = (hour_col > 16) & (hour_col < 20)

        return pd.DataFrame(np.array([night_hour, peak_hour]).T, columns=["night_hour", "peak_hour"], dtype="int8", index=index)
