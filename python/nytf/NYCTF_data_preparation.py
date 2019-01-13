#!/usr/bin/env python3
# -*- coding: utf-8-unix -*-

# Data preparation for the NYC taxi fares challenge.

import numpy as np
import os
import pandas as pd
import pickle
import subprocess
   
def prepare_data(dfToPrepare, dfType, varTypes, b_parseDates, b_saveFile=False, saveDir='', fileName=''):
    """
    Prepares data by changing the types to more relevant ones and removing records with negative fares.
    Can parse dates and/or save dataframes in pickle files whenever asked.
    
    Parameters:
    dfTrainToPrepare - dataframe -- raw training or test dataframe that needs to be prepared
    dfType - string -- 'train' or 'test' (not case-sensitive)
    varTypes - dictionnary -- specified types for chosen features
    b_parseDates - boolean (optional) -- True if 'pickup_datetime' is  not already parsed, False if it is
    saveDir - string/path (optiona) -- path to directory where to save the pickle files
    fileName - string -- name of the file
    
    Return:
    dfTrainPrep - dataframe -- clean (prepared) training dataframe
    dfTestPrep - dataframe -- clean (prepared) test dataframe
    
    Note:
    'pickup_datetime' is left unchanged in the "choose more relevant types" sections.
    It is dealt with in the parse date section (if not already dealt with in the read_csv step).
    """
    
    dfPrep = dfToPrepare.copy()
    print(dfType)
    
    # 1- Change types to more relevant ones (part 1)
    if dfType.lower()=='train':
        newTypes = {key: varTypes[key] for key in varTypes.keys() if key!='passenger_count'}
        #in the specific case of the training dataset,
        #'passenger_count' will be dealt with later as it contains NaN values but can be converted to uint8
        #once these NaNs are removed.
    elif dfType.lower()=='test':
        newTypes = {key: varTypes[key] for key in varTypes.keys() if key!='fare_amount'}
        #no 'fare_amount' in test set: this is what we want to predict
    try:
        dfPrep = dfPrep.astype(newTypes)
        print('Step 1/6 complete.')
    except (RuntimeError, TypeError, NameError):
        print('Step 1/6 failed. Wrong df type or wrong variable types.')
        
    # 2- Remove incomplete rows (from training dataset, there aren't any in the test dataset)
    if dfType.lower()=='train':
        dfPrep = dfPrep.replace([np.inf, -np.inf], np.nan).dropna(how='any', axis='rows')
        print('Step 2/6 complete. Incomplete rows have been removed.')
    else:
        print('Step 2/6 skipped. Not a training set.')
        
    # 3- Change types to more relevant ones (part 2)
    if dfType.lower()=='train':
        try:
            dfPrep = dfPrep.astype({'passenger_count': varTypes['passenger_count']})
            print('Step 3/6 complete. Types changed to more relevant ones.')        
        except (RuntimeError, TypeError, NameError):
            print('Step 3/6 failed. Wrong variable type for passenger_count.')
    else:
        print('Step 3/6 skipped. Not a training set.')
        
    # 4- Parse dates
    if b_parseDates:
        dfPrep['pickup_datetime'] = dfPrep['pickup_datetime'].str.slice(0, 16)
        #characteristic time for changes in traffic situation is supposed to exceed 1 minute
        #(i.e. seconds shouldn't bring relevant information: we ditch them)
        dfPrep['pickup_datetime'] = pd.to_datetime(dfPrep['pickup_datetime'], format='%Y-%m-%d %H:%M', utc=True)   
        print('Step 4/6 complete. Dates have been parsed.')
    else:
        print('Step 4/6 skipped. Dates already parsed.')
        
    # 5- Remove negative values and >=$100 values for fares 
    if dfType.lower()=='train':
        dfPrep = dfPrep[(dfPrep['fare_amount']>0) & (dfPrep['fare_amount']<1000)]
        print('Step 5/6 complete. Records with negative or >=$1000 fares have been removed.')
    else:
        print('Step 5/6 skipped. Not a training set.')
    
    # N- Save clean dataframe
    if b_saveFile:
        with open(os.path.join(saveDir, fileName), 'wb') as f:
            pickle.dump(dfPrep, f)       
        print('Step 6/6 completed. Prepared dataframe has been saved in a pickle file.')
    else:
        print("""Step 6/6 skipped.
        Dataframe is prepared. If you want to save it in a pickle file, choose saveDir and fileName, change dfPrep to its real name and type
               with open(os.path.join(saveDir, fileName), 'wb') as f:
                  pickle.dump(dfPrep, f)""")
    
    return dfPrep 
    
def row_count(filePath, method='unixwc'):
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
            with open(filePath) as file:
                n_rows = len(file.readlines())
        if method=='unixwc':
            p = subprocess.Popen(['wc', '-l', filePath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            result, err = p.communicate()
            if p.returncode != 0:
                raise IOError(err)
            n_rows = int(result.strip().split()[0]) + 1 #we still add 1 because of the last line with no newline character
        return n_rows
    except:
        print('Method and/or file not found.')
        pass
