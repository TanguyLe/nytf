import os
import pickle

from pandas import Series, DataFrame, read_csv, to_datetime
from pytz import timezone
from sklearn.base import BaseEstimator, TransformerMixin

PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
RAW_DIRECTORY = os.path.join(PROJECT_DIR, 'data', 'raw')
PROCESSING_DIRECTORY = os.path.join(PROJECT_DIR, 'data', 'processing')


def load_dataframe(name, extension=None, drop_key=None, save_pkl=None):
    """Load train or test data frame.

        The data frame is load from RAW_DIRECTORY for csv file and else from PROCESSING_DIRECTORY.

        Parameters
        ----------
        name : str
            The name of the data frame file, without extension. Usually 'train' or 'test'.
        extension : 'csv', 'pkl' or None
            The extension of the file to use. If None, 'pkl' is used if a pickle file exists else 'csv'.
        drop_key : True, False or None
            If True, the 'key' column is not read in csv file, if None it is read only for the test data frame.
        save_pkl : True, False or None
            If True, read csv file are saved ase pickle, if None it is only saved if extension is None.

        Returns
        -------
        pandas.DataFrame
            The read data frame.
        """
    pickle_path = os.path.join(PROCESSING_DIRECTORY, name + '.pkl')
    csv_path = os.path.join(RAW_DIRECTORY, name + '.csv')

    if save_pkl is None:
        save_pkl = extension is None
    if extension is None:
        extension = 'pkl' if os.path.exists(pickle_path) else 'csv'
    if drop_key is None:
        drop_key = 'train' in name

    if extension is 'pkl':
        with open(pickle_path, 'rb') as file:
            return pickle.load(file)

    elif extension is 'csv':
        dataframe = read_csv(csv_path,
                             usecols=(lambda colname: colname != 'key') if drop_key else None,
                             dtype={'fare_amount': 'float32', 'pickup_longitude': 'float32',
                                    'pickup_latitude': 'float32', 'dropoff_longitude': 'float32',
                                    'dropoff_latitude': 'float32', 'passenger_count': 'int8'})
        dataframe.pickup_datetime = to_datetime(dataframe.pickup_datetime, format='%Y-%m-%d %H:%M:%S UTC', utc=True)
        if save_pkl:
            with open(pickle_path, 'wb') as file:
                pickle.dump(dataframe, file)
        return dataframe

    raise ValueError('Not supported extension.')


class BasicTemporalFeatures(BaseEstimator, TransformerMixin):

    def fit(self, *args, **kwargs):
        return self

    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    @classmethod
    def implemented_features(cls):
        return (
            'timestamp', 'minute', 'hour', 'day', 'month', 'year', 'dayofweek', 'dayofyear', 'days_in_month',
            'is_leap_year', 'day_progress', 'week_progress', 'month_progress', 'year_progress')

    @property
    def feature_names(self):
        return tuple(self._feature_names)

    @feature_names.setter
    def feature_names(self, feature_names):
        self._feature_names = list(feature_names if feature_names else self.implemented_features())
        to_compute = set(self._feature_names)

        if 'year_progress' in to_compute:
            to_compute.update(('day_progress', 'dayofyear', 'is_leap_year'))
        if 'month_progress' in to_compute:
            to_compute.update(('day_progress', 'day', 'days_in_month'))
        if 'week_progress' in to_compute:
            to_compute.update(('day_progress', 'dayofweek'))
        if 'day_progress' in to_compute:
            to_compute.update(('hour', 'minute'))

        for feature in ('timestamp', 'day_progress', 'week_progress', 'month_progress', 'year_progress'):
            setattr(self, '_compute_' + feature, feature in to_compute)
        self._attribute_to_extract = to_compute.intersection(
            ('minute', 'hour', 'day', 'month', 'year', 'dayofweek', 'dayofyear', 'days_in_month', 'is_leap_year'))

        if not to_compute.issubset(self.implemented_features()):
            raise ValueError('Not implemented feature(s): ' + ', '.join(
                feature for feature in to_compute.difference(self.implemented_features())))

    _timezone = timezone('US/Eastern')
    _attribute_types = {'minute': 'int8', 'hour': 'int8', 'day': 'int8', 'month': 'int8', 'year': 'int16',
                       'dayofweek': 'int8', 'dayofyear': 'int16', 'days_in_month': 'int8', 'is_leap_year': 'bool'}

    def transform(self, X):
        datetime_series = X["pickup_datetime"]
        temporal_features = DataFrame(index=datetime_series.index)

        if self._compute_timestamp:
            temporal_features['timestamp'] = (datetime_series.astype('int64').values // 10 ** 9).astype('int32')

        if self._attribute_to_extract:
            local_datetime = datetime_series.dt.tz_convert(self._timezone).dt
            for attribute in self._attribute_to_extract:
                temporal_features[attribute] = getattr(local_datetime, attribute).values.astype(
                    self._attribute_types[attribute])

        if self._compute_day_progress:
            temporal_features['day_progress'] = (temporal_features.hour.values.astype('float32') +
                                                 temporal_features.minute.values.astype('float32') / 60) / 24
            if self._compute_week_progress:
                temporal_features['week_progress'] = (temporal_features.dayofweek.values.astype('float32') +
                                                      temporal_features.day_progress.values.astype('float32')) / 7
            if self._compute_month_progress:
                temporal_features['month_progress'] = (temporal_features.day.values.astype('float32') - 1 +
                                                       temporal_features.day_progress.values.astype('float32')) / \
                                                      temporal_features.days_in_month.values.astype('float32')
            if self._compute_year_progress:
                temporal_features['year_progress'] = (temporal_features.dayofyear.values.astype('float32') - 1 +
                                                      temporal_features.day_progress.values.astype('float32')) / \
                                                     (365 + temporal_features.is_leap_year.values.astype('float32'))

        return temporal_features[self._feature_names]
