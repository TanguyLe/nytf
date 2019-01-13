#!/usr/bin/env python3
# -*- coding: utf-8-unix -*-

# Functions related to the geographical features - NYC taxi fares challenge.

from math import *
import numpy as np

def removeOutlyingCoordinates(df, minmaxCoordinates):
    """
    Remove records with incoherent coordinates (specific to the training set). 
    
    Parameters:
    df - dataframe -- (training) dataframe in which to remove outliers
    minmaxCoordinates - array of 4 floats -- extrema for coordinates in the following order: [latitudeMin, latitudeMax, longitudeMin, longitudeMax]
    
    Return:
    dataframe without the spotted incoherent records
    """
    #remove outliers based on longitude and latitude (specific to training set)
    return df[(df['pickup_latitude']>=minmaxCoordinates[0])
              & (df['pickup_latitude']<=minmaxCoordinates[1])
              & (df['pickup_longitude']>=minmaxCoordinates[2])
              & (df['pickup_longitude']<=minmaxCoordinates[3])
              & (df['dropoff_latitude']>=minmaxCoordinates[0])
              & (df['dropoff_latitude']<=minmaxCoordinates[1])
              & (df['dropoff_longitude']>=minmaxCoordinates[2])
              & (df['dropoff_longitude']<=minmaxCoordinates[3])
              ]

def flying_distance_AB(latitudeA, longitudeA, latitudeB, longitudeB, angUnit, r):
    """
    Calculates flying distance between A and B (in the same unit as r).
    
    Parameters:
    latitudeA - float -- latitude of point A (rad)
    longitudeA - float -- longitude of point A (rad)
    latitudeB - float -- latitude of point B (rad)
    longitudeB - float -- longitude of point B (rad)
    angUnit - str -- unit used for angles ('rad' or 'deg')
    r - float -- radius of the sphere in the chosen unit for the flying distance.
    
    Return:
    Flying distance between A and B in the same unit as r.
    """
    if angUnit.lower() in ('rad', 'radian'):
        conv_k = 1
    elif angUnit.lower() in ('deg', 'degree', '°'):
        conv_k = pi/180
    else:
        print('Unknown unit for angles:' + angUnit)
        pass
    return r*(acos(
        cos(latitudeA*conv_k)*cos(latitudeB*conv_k)*cos(fabs(longitudeA-longitudeB)*conv_k)
        +sin(latitudeA*conv_k)*sin(latitudeB*conv_k)
    ))

def flying_distance_row(row, angUnit, r):
    """
    Calculates flying distance between pickup location and dropoff location for a given row (in the same unit as r).
    """
    return flying_distance_AB(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude'], angUnit, r)

def add_flying_distance(df, angUnit='rad', r=6371, unit='km'):
    """
    Adds a 'flying_distance' column, whose unit has to be the one of in which r is expressed.
    By default, the value and unit of r correspond to the average radius of the Earth (6371 km).
    
    Parameters:
    df - dataframe -- training or test dataframe in which to add the column
    angUnit - str (optional, default 'rad') -- chosen unit for angles ('rad'/'deg')
    r - float (optional, default 6371) -- radius of the sphere in the chosen unit
    unit - str (optional, default 'km') -- chosen unit for distances, used to name the new column
    
    Return:
    df_out - dataframe -- input dataframe with flying_distance column added
    """
    df_out = df.copy()
    df_out['flying_distance_' + unit] = df_out.apply(lambda row: flying_distance_row(row, angUnit, r), axis=1)
    return df_out
   
def L1_distance_AB(latitudeA, longitudeA, latitudeB, longitudeB, angUnit, r, plane_rot_angle):
    """
    Given the specificity of streets in Manhattan, calculates an approximated L1-distance between points A and B
    in the plane where the (N,E) spatial system (resp. the map) is rotated by plane_rot_angle radians
    trigonometric wise (resp. clockwise).
    """
    if angUnit.lower() in ('rad', 'radian'):
        conv_k = 1
    elif angUnit.lower() in ('deg', 'degree', '°'):
        conv_k = pi/180
    else:
        print('Unknown unit for angles:' + angUnit)
        pass
    
    cos_lat_squared_avg = (sin(2*latitudeB*conv_k)/4
                           + latitudeB*conv_k/2
                           - sin(2*latitudeA*conv_k)/4
                           - latitudeA*conv_k/2
                          )/((latitudeB - latitudeA)*conv_k)
    sin_lat_squared_avg = (- sin(2*latitudeB*conv_k)/4
                           + latitudeB*conv_k/2
                           + sin(2*latitudeA*conv_k)/4
                           - latitudeA*conv_k/2
                            )/((latitudeB - latitudeA)*conv_k)   
    
    X = [r*acos((cos_lat_squared_avg*cos(longitudeA*conv_k - longitudeB*conv_k) + sin_lat_squared_avg)),
         r*fabs(latitudeA*conv_k - latitudeB*conv_k)      
        ]
    a=cos(plane_rot_angle)
    b=sin(plane_rot_angle)
    #la matrice de passage correspondant au changement de repère est np.matrix([[a, -b], [b, a]])
    # sa matrice inverse est:
    P_inv = np.matrix([[a, b], [-b, a]]) #car rotation appartient à SO2(R)
    X_prime = P_inv.dot(X)
   
    return fabs(X_prime[0, 0]) + fabs(X_prime[0, 1])
    
def L1_distance_row(row, angUnit, r, plane_rot_angle):
    """
    Calculates 1-distance between pickup location and dropoff location for a given row (in the same unit as r).
    """
    return L1_distance_AB(row['pickup_latitude'],
                          row['pickup_longitude'],
                          row['dropoff_latitude'],
                          row['dropoff_longitude'],
                          angUnit,
                          r,
                          plane_rot_angle)

def add_L1_distance(df, angUnit='rad', r=6371, unit='km', plane_rot_angle=0):
    """
    Adds a '1-distance' column, whose unit has to be the one in which r is expressed.
    By default, the value and unit of r correspond to the average radius of the Earth (6371 km).
    
    Parameters:
    df - dataframe -- training or test dataframe in which to add the column
    angUnit - str (optional, default 'rad') -- chosen unit for angles ('rad'/'deg')
    r - float (optional) -- radius of the sphere in the chosen unit. By default, r is the average radius of the Earth (r=6371 km).
    unit - str (optional, default 'km') -- chosen unit for distances, used to name the new column
    plane_rot_angle - float -- Rotation (rad). Used to get the spatial system lined up on the direction of the street of Manhattan.
        Trigonometric wise if rotating the spatial system. Clockwise if rotating the spatial map.
    
    Return:
    df_out - dataframe -- input dataframe with flying_distance column added
    """
    df_out = df.copy()
    df_out['L1_distance_' + unit] = df_out.apply(lambda row: L1_distance_row(row, angUnit, r, plane_rot_angle), axis=1)
    return df_out
    


    
