from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .business_time_features import BusinessFeatures
from .holidays_extractor import HolidayFeaturesExtractor
from .utils import BasicTemporalFeatures

feature_names = ['timestamp', 'minute', 'hour', 'day', 'month', 'year', 'dayofweek', 'dayofyear', 'days_in_month',
                 'is_leap_year', 'day_progress', 'week_progress', 'month_progress', 'year_progress']

add_business = ColumnTransformer([("business_temporal_features", BusinessFeatures(hour_col='hour'), ['hour'])],
                                 remainder="passthrough")
temporal_features = Pipeline([("basic_temporal_features", BasicTemporalFeatures(feature_names)),
                              ("add_business", add_business)])

holidays_features = HolidayFeaturesExtractor(date_col="pickup_datetime", interest_col="fare_amount", state="NY")

main_estimator = ColumnTransformer(
    transformers=[
        ("temporal_features", temporal_features, ["pickup_datetime"]),
        ("holiday_features", holidays_features, ["pickup_datetime", "fare_amount"]),
        ("geo_features", 'drop', ["pickup_longitude", "pickup_latitude",
                                  "dropoff_latitude", "dropoff_longitude"]),
        ("labels", "passthrough", ["fare_amount"])])

main_pipeline = Pipeline([('feature_engineering', main_estimator),
                          ('scaling', StandardScaler())])
