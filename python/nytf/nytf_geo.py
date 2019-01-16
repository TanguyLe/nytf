#!/usr/bin/env python3
# -*- coding: utf-8-unix -*-

# Functions related to the geographical features - NYC taxi fares challenge.

from math import *
import numpy as np

def remove_outlying_coordinates(df, minmax_coordinates):
    """
    Remove records with incoherent coordinates (specific to the training set). 
    
    Parameters:
    df - dataframe -- (training) dataframe in which to remove outliers
    minmax_coordinates - array of 4 floats -- extrema for coordinates in the following order: [latitude min, latitude max, longitude min, longitude max]
    
    Return:
    dataframe without the spotted incoherent records
    """
    return df[(df['pickup_latitude']>=minmax_coordinates[0])
              & (df['pickup_latitude']<=minmax_coordinates[1])
              & (df['pickup_longitude']>=minmax_coordinates[2])
              & (df['pickup_longitude']<=minmax_coordinates[3])
              & (df['dropoff_latitude']>=minmax_coordinates[0])
              & (df['dropoff_latitude']<=minmax_coordinates[1])
              & (df['dropoff_longitude']>=minmax_coordinates[2])
              & (df['dropoff_longitude']<=minmax_coordinates[3])
              ]

def flying_distance_AB(latitude_A, longitude_A, latitude_B, longitude_B, ang_unit, r):
    """
    Calculates flying distance between A and B (in the same unit as r).
    
    Parameters:
    latitude_A - float -- latitude of point A (rad)
    longitude_A - float -- longitude of point A (rad)
    latitude_B - float -- latitude of point B (rad)
    longitude_B - float -- longitude of point B (rad)
    ang_unit - str -- unit used for angles ('rad' or 'deg')
    r - float -- radius of the sphere in the chosen unit for the flying distance.
    
    Return:
    Flying distance between A and B in the same unit as r.
    """
    if ang_unit.lower() in ('rad', 'radian'):
        conv_k = 1
    elif ang_unit.lower() in ('deg', 'degree', '°'):
        conv_k = pi/180
    else:
        print('Unknown unit for angles:' + ang_unit)
        pass
    
    if latitude_A==latitude_B and longitude_A==longitude_B:
        return 0
        #without this, the calculated flying_distance for some cases where A=B is slightly positive (around E-5 km) because of
        #rounding errors (the mathematical expression below is however exact). As the trips where A=B seem to be of particular
        #interest, we make sure they are correctly spotted and lead to a distance of zero.
    else:
        cos_norm_res = min(max(-1.0, cos(latitude_A*conv_k)*cos(latitude_B*conv_k)*cos(fabs(longitude_A-longitude_B)*conv_k)
                               +sin(latitude_A*conv_k)*sin(latitude_B*conv_k)), 1.0)
        #preventing math error due to rounding errors (the mathematical expression is however exact)
        return r*acos(cos_norm_res)

def add_flying_distance(df, ang_unit='rad', r=6371, dist_unit='km'):
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
    df_out = df.copy(False)
    df_out['flying_distance_' + dist_unit] = list(map(lambda a, b, c, d: flying_distance_AB(a, b, c, d, ang_unit, r),
                                                             df_out['pickup_latitude'],
                                                             df_out['pickup_longitude'], 
                                                             df_out['dropoff_latitude'],
                                                             df_out['dropoff_longitude']))                                                
                                                    
    return df_out
   
def L1_distance_AB(latitude_A, longitude_A, latitude_B, longitude_B, ang_unit, r, plane_rot_angle):
    """
    Given the specificity of streets in Manhattan, calculates an approximated L1-distance between points A and B
    in the plane where the (N,E) spatial system (resp. the map) is rotated by plane_rot_angle radians
    trigonometric wise (resp. clockwise).
    """
    if ang_unit.lower() in ('rad', 'radian'):
        conv_k = 1
    elif ang_unit.lower() in ('deg', 'degree', '°'):
        conv_k = pi/180
    else:
        print('Unknown unit for angles:' + ang_unit)
        pass
    
    if latitude_A==latitude_B:
        if longitude_A==longitude_B:
            X = [0, 0]
        else:
            X = [r*(acos(cos(latitude_A*conv_k)*cos(latitude_B*conv_k)*cos(fabs(longitude_A-longitude_B)*conv_k)
                         +sin(latitude_A*conv_k)*sin(latitude_B*conv_k))),
                 0
                ]
    else:
        cos_lat_squared_avg = (sin(2*latitude_B*conv_k)/4
                               + latitude_B*conv_k/2
                               - sin(2*latitude_A*conv_k)/4
                               - latitude_A*conv_k/2
                              )/((latitude_B - latitude_A)*conv_k)
        sin_lat_squared_avg = (- sin(2*latitude_B*conv_k)/4
                               + latitude_B*conv_k/2
                               + sin(2*latitude_A*conv_k)/4
                               - latitude_A*conv_k/2
                                )/((latitude_B - latitude_A)*conv_k)
        cos_norm_X1 = min(max(-1.0, cos_lat_squared_avg*cos(longitude_A*conv_k - longitude_B*conv_k) + sin_lat_squared_avg), 1.0)
            #Because of the approximation of mean(arccos(...)) by arccos(mean(...)), it is possible that the approximate value whose
            #arcos is to be calculated be slightly out of range. We put it back in the [-1;1] interval.
        X = [r*acos(cos_norm_X1),
             r*fabs(latitude_A*conv_k - latitude_B*conv_k)      
            ]

    a=cos(plane_rot_angle)
    b=sin(plane_rot_angle)
    #the transfer matrix from the old axis system to the new axis system is P=np.matrix([[a, -b], [b, a]])
    #the inverse matrix is:
    P_inv = np.matrix([[a, b], [-b, a]]) #because P belongs to SO_2(R)
    X_prime = P_inv.dot(X)
   
    return fabs(X_prime[0, 0]) + fabs(X_prime[0, 1])

def add_L1_distance(df, ang_unit='rad', r=6371, unit='km', plane_rot_angle=0):
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
    df_out = df.copy(False)
    df_out['L1_distance_' + unit] = list(map(lambda a, b, c, d: L1_distance_AB(a, b, c, d, ang_unit, r, plane_rot_angle),
                                             df_out['pickup_latitude'],
                                             df_out['pickup_longitude'], 
                                             df_out['dropoff_latitude'],
                                             df_out['dropoff_longitude']))
    return df_out  