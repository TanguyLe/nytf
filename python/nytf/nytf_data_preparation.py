#!/usr/bin/env python3
# -*- coding: utf-8-unix -*-

# Data preparation for the NYC taxi fares challenge.

import numpy as np
import os
import pandas as pd
import pickle
import subprocess
   
def prepare_data(df_to_prepare, df_type, var_types, b_parse_dates, b_save_file=False, save_dir='', file_name=''):
    """
    Prepares data by changing the types to more relevant ones and removing records with negative fares.
    Can parse dates and/or save dataframes in pickle files whenever asked.
    
    Parameters:
    df_train_to_prepare - dataframe -- raw training or test dataframe that needs to be prepared
    df_type - string -- 'train' or 'test' (not case-sensitive)
    var_types - dictionnary -- specified types for chosen features
    b_parse_dates - boolean (optional) -- True if 'pickup_datetime' is  not already parsed, False if it is
    save_dir - string/path (optiona) -- path to directory where to save the pickle files
    file_name - string -- name of the file
    
    Return:
    df_train_prep - dataframe -- clean (prepared) training dataframe
    df_test_prep - dataframe -- clean (prepared) test dataframe
    
    Note:
    'pickup_datetime' is left unchanged in the "choose more relevant types" sections.
    It is dealt with in the parse date section (if not already dealt with in the read_csv step).
    """
    
    df_prep = df_to_prepare.copy()
    print(df_type)
    
    # 1- Change types to more relevant ones (part 1)
    if df_type.lower()=='train':
        new_types = {key: var_types[key] for key in var_types.keys() if key!='passenger_count'}
        #in the specific case of the training dataset,
        #'passenger_count' will be dealt with later as it contains NaN values but can be converted to uint8
        #once these NaNs are removed.
    elif df_type.lower()=='test':
        new_types = {key: var_types[key] for key in var_types.keys() if key!='fare_amount'}
        #no 'fare_amount' in test set: this is what we want to predict
    try:
        df_prep = df_prep.astype(new_types)
        print('Step 1/7 complete.')
    except (RuntimeError, TypeError, NameError):
        print('Step 1/7 failed. Wrong df type or wrong variable types.')
        
    # 2- Remove incomplete rows (from training dataset, there aren't any in the test dataset)
    if df_type.lower()=='train':
        df_prep = df_prep.replace([np.inf, -np.inf], np.nan).dropna(how='any', axis='rows')
        print('Step 2/7 complete. Incomplete rows have been removed.')
    else:
        print('Step 2/7 skipped. Not a training set.')
        
    # 3- Change types to more relevant ones (part 2)
    if df_type.lower()=='train':
        try:
            df_prep = df_prep.astype({'passenger_count': var_types['passenger_count']})
            print('Step 3/7 complete. Types changed to more relevant ones.')        
        except (RuntimeError, TypeError, NameError):
            print('Step 3/7 failed. Wrong variable type for passenger_count.')
    else:
        print('Step 3/7 skipped. Not a training set.')
        
    # 4- Parse dates
    if b_parse_dates:
        df_prep['pickup_datetime'] = df_prep['pickup_datetime'].str.slice(0, 16)
        #characteristic time for changes in traffic situation is supposed to exceed 1 minute
        #(i.e. seconds shouldn't bring relevant information: we ditch them)
        df_prep['pickup_datetime'] = pd.to_datetime(df_prep['pickup_datetime'], format='%Y-%m-%d %H:%M', utc=True)   
        print('Step 4/7 complete. Dates have been parsed.')
    else:
        print('Step 4/7 skipped. Dates already parsed.')
        
    # 5- Remove negative values and >=$100 values for fares 
    if df_type.lower()=='train':
        df_prep = df_prep[(df_prep['fare_amount']>0) & (df_prep['fare_amount']<1000)]
        print('Step 5/7 complete. Records with negative or >=$1000 fares have been removed.')
    else:
        print('Step 5/7 skipped. Not a training set.')
        
    # 6- Reset indexes
    df_prep = df_prep.reset_index(drop=True)
    print('Step 6/7 complete. Indexes reset.')
    
    # 7- Save clean dataframe
    if b_save_file:
        with open(os.path.join(save_dir, file_name), 'wb') as f:
            pickle.dump(df_prep, f)       
        print('Step 7/7 complete. Prepared dataframe has been saved in a pickle file.')
    else:
        print("""Step 7/7 skipped.
        Dataframe is prepared. If you want to save it in a pickle file, choose saveDir and fileName, change dfPrep to its real name and type
               with open(os.path.join(saveDir, fileName), 'wb') as f:
                  pickle.dump(dfPrep, f)""")
    
    return df_prep 
    
def row_count(file_path, method='unixwc'):
    """
    Counts number of rows in given file.
    
    Parameters:
    filePath - path (string) -- directory to input file
    method - string (optional) -- method used to calculate the number of rows. Try to choose the optimal one!
    
    Return:
    n_rows - int -- number of rows of input file
    """
    try:
        if method=='readlines':
            with open(file_path) as file:
                n_rows = len(file.readlines())
        if method=='unixwc':
            p = subprocess.Popen(['wc', '-l', file_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            result, err = p.communicate()
            if p.returncode != 0:
                raise IOError(err)
            n_rows = int(result.strip().split()[0]) + 1 #we still add 1 because of the last line with no newline character
        return n_rows
    except:
        print('Method and/or file not found.')
        pass
