import os
import pickle

import pandas as pd

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
        dataframe = pd.read_csv(csv_path, usecols=(lambda colname: colname != 'key') if drop_key else None)
        dataframe.pickup_datetime = pd.to_datetime(dataframe.pickup_datetime, format='%Y-%m-%d %H:%M:%S UTC', utc=True)
        if save_pkl:
            with open(pickle_path, 'wb') as file:
                pickle.dump(dataframe, file)
        return dataframe

    raise ValueError('Not supported extension.')
