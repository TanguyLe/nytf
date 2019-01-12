from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import BasicTemporalFeatures
from .holidays_extractor import HolidayFeaturesExtractor

feature_names = ['timestamp', 'minute', 'hour', 'day', 'month', 'year', 'dayofweek', 'dayofyear', 'days_in_month',
                 'is_leap_year', 'day_progress', 'week_progress', 'month_progress', 'year_progress']

main_estimator = ColumnTransformer(
    transformers=[("basic_temporal_features", BasicTemporalFeatures(feature_names), ["pickup_datetime"]),
                  ("holiday_features",
                   HolidayFeaturesExtractor(date_col="pickup_datetime", interest_col="fare_amount", state="NY"),
                   ["pickup_datetime", "fare_amount"]),
                  ("geo_features", 'drop', ["pickup_longitude", "pickup_latitude",
                                            "dropoff_latitude", "dropoff_longitude"]),
                  ("labels", "passthrough", ["fare_amount"])])

main_pipeline = Pipeline([('feature_engineering', main_estimator),
                         ('scaling', StandardScaler())])
