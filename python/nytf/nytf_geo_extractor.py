#!/usr/bin/env python3
# -*- coding: utf-8-unix -*-

# Geographical features for the NYC taxi fares challenge.

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from importlib import reload
import nytf_geo as geo; reload(geo);

dist_unit = 'km'

GEO_EXTRACTED_FEATURES = ['flying_distance_' + dist_unit,
                          'L1_distance_' + dist_unit]

class GeoFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, df_geo_cols, ang_unit='rad', dist_unit='km', r=6371, plane_rot_angle=0):
        
        self.df_geo_cols = df_geo_cols
        self.ang_unit = ang_unit
        self.dist_unit = dist_unit
        self.r = r
        self.plane_rot_angle = plane_rot_angle

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_geo_features = self.df_geo_cols.copy(deep=False)
        df_geo_features['flying_distance_' + dist_unit] = df_geo_features.list(map(
            lambda a, b, c, d: geo.flying_distance_AB(a, b, c, d, ang_unit, r), 
            df_geo_features['pickup_latitude'],
            df_geo_features['pickup_longitude'], 
            df_geo_features['dropoff_latitude'], 
            df_geo_features['dropoff_longitude']))
        df_geo_features['L1_distance_' + dist_unit] = df_geo_features.list(map(
            lambda a, b, c, d: geo.L1_distance_AB(a, b, c, d, ang_unit, r, plane_rot_angle),
            df_geo_features['pickup_latitude'],
            df_geo_features['pickup_longitude'], 
            df_geo_features['dropoff_latitude'], 
            df_geo_features['dropoff_longitude']))
        
        return df_geo_features[GEO_EXTRACTED_FEATURES]