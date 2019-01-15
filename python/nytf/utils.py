import os
import pickle
from math import pi

from numpy import cos, sin, array
from pandas import DataFrame, read_csv, to_datetime
from pytz import timezone
from sklearn.base import BaseEstimator, TransformerMixin

from .utils2 import decompose_data_to_arrays_list

PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
RAW_DIRECTORY = os.path.join(PROJECT_DIR, 'data', 'raw')
PROCESSING_DIRECTORY = os.path.join(PROJECT_DIR, 'data', 'processing')


def load_dataframe(name, pkl=True, cloud=True, inf_passenger=0, sup_passenger=7, inf_fare=0, sup_fare=100,
                   inf_lat=40.56, sup_lat=41.71, inf_long=-74.27, sup_long=-72.98):
    """Load train or test data frame.

        The data frame is load from RAW_DIRECTORY for csv file and else from PROCESSING_DIRECTORY.

        Parameters
        ----------
        name : str
            The name of the data frame file, without extension. Usually 'train' or 'test'.
        pkl : bool
            Whether to load a pickle file or the csv. Assuming that the csv is always saved as a pickle afterwards.
        cloud : bool
            Whether to load from the cloud bucket
        inf_passenger, inf_fare, inf_lat, inf_long : real
            The values less or equal are drop for the training data frame.
        sup_passenger, sup_fare, sup_lat, sup_long : real
            The values greater or equal are drop for the training data frame.

        Returns
        -------
        pandas.DataFrame
            The read data frame.
        """
    if cloud:
        import datalab.storage as storage
        from io import BytesIO

        nytf_bucket = storage.Bucket("nytf")
        remote_pickle = nytf_bucket.item('train.pkl').read_from()
        return pickle.load(BytesIO(remote_pickle))

    pkl_path = os.path.join(PROCESSING_DIRECTORY, name + '.pkl')

    if pkl:
        with open(pkl_path, 'rb') as file:
            return pickle.load(file)

    dataframe = read_csv(os.path.join(RAW_DIRECTORY, name + '.csv'),
                         usecols=(lambda colname: colname != 'key') if 'train' in name else None,
                         dtype={'fare_amount': 'float32', 'pickup_longitude': 'float32',
                                'pickup_latitude': 'float32', 'dropoff_longitude': 'float32',
                                'dropoff_latitude': 'float32', 'passenger_count': 'int8'})
    dataframe.pickup_datetime = to_datetime(dataframe.pickup_datetime, format='%Y-%m-%d %H:%M:%S UTC', utc=True)

    if 'train' in name:
        dataframe = dataframe[
            (inf_passenger < dataframe.passenger_count) & (dataframe.passenger_count < sup_passenger) &
            (inf_fare < dataframe.fare_amount) & (dataframe.fare_amount < sup_fare) &
            (inf_lat < dataframe.pickup_latitude) & (dataframe.pickup_latitude < sup_lat) &
            (inf_lat < dataframe.dropoff_latitude) & (dataframe.dropoff_latitude < sup_lat) &
            (inf_long < dataframe.pickup_longitude) & (dataframe.pickup_longitude < sup_long) &
            (inf_long < dataframe.dropoff_longitude) & (dataframe.dropoff_longitude < sup_long)].copy(deep=False)

    with open(pkl_path, 'wb') as file:
        pickle.dump(dataframe, file)

    return dataframe


class BasicTemporalFeatures(BaseEstimator, TransformerMixin):
    def fit(self, *args, **kwargs):
        return self

    def __init__(self, feature_names=None):
        self.feature_names = feature_names
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

    @staticmethod
    def implemented_features():
        return (
            'timestamp', 'minute', 'hour', 'day', 'month', 'year', 'dayofweek', 'dayofyear', 'days_in_month',
            'is_leap_year', 'day_progress', 'week_progress', 'month_progress', 'year_progress')

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


class SegmentToCircle(BaseEstimator, TransformerMixin):
    def __init__(self, segment_min=0, segment_max=1):
        self.segment_min = segment_min
        self.segment_max = segment_max

    def fit(self, *args, **kwargs):
        return self

    def transform(self, data):
        data, index, col_names = decompose_data_to_arrays_list(data=data)

        circle_data = []
        for col in data:
            circle_data.append(cos((col - self.segment_min) / ((self.segment_max - self.segment_min) * 2 * pi)))
            circle_data.append(sin((col - self.segment_min) / ((self.segment_max - self.segment_min) * 2 * pi)))

        if len(col_names) == 0:
            return array(circle_data).transpose()

        dataframe = DataFrame(index=index)
        for i, name in enumerate(col_names):
            dataframe[name + '_cos'] = circle_data[2 * i]
            dataframe[name + '_sin'] = circle_data[2 * i + 1]
        return dataframe
