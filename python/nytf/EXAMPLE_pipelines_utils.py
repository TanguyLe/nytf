import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from utils import get_timestamp, get_day_of_week, get_day_of_year, get_hour_of_day, get_month_of_year, flatten_list

DATE_COLUMNS = {
    "timestamp": {
        "name": "Timestamp",
        "type_col": "float",
        "fct": get_timestamp
    },
    "day_of_year": {
        "name": "DayOfYear",
        "type_col": "int",
        "fct": get_day_of_year
    },
    "month_of_year": {
        "name": "MonthOfYear",
        "type_col": "int",
        "fct": get_month_of_year
    },
    "day_of_week": {
        "name": "DayOfWeek",
        "type_col": "int",
        "fct": get_day_of_week
    },
    "hour_of_day": {
        "name": "HourOfDay",
        "type_col": "int",
        "fct": get_hour_of_day
    }
}


class DatePipeline(BaseEstimator, TransformerMixin):
    def __init__(self, date_format, col_options=None):
        self._orig_col_options = col_options

        if col_options is None:
            col_options = DATE_COLUMNS
        else:
            col_options = {e: DATE_COLUMNS[e] for e in col_options}

        self._col_options = col_options
        self._format = date_format
        self._columns = None
        self._dtypes = None
        self._date_cols = None

    def fit(self, X, y=None):
        self._date_cols = X.columns
        self._columns = self._get_cols_names()
        self._dtypes = self._get_cols_dtypes()

        return self

    def get_params(self, deep=True):
        return {"format": self._format, "col_options": self._orig_col_options}

    def get_cols_names(self):
        return self._columns

    def get_cols_dtypes(self):
        return self._dtypes

    def _get_cols_dtypes(self):
        return flatten_list(
            [[col_opt["type_col"] for _, col_opt in self._col_options.items()] for _ in self._date_cols])

    def _get_cols_names(self):
        return flatten_list(
            [[c + '_' + col_opt["name"] for _, col_opt in self._col_options.items()] for c in self._date_cols])

    def transform(self, X):
        dates_col = pd.to_datetime(X.stack(), format='%Y-%m-%d %H:%M:%S').unstack()
        new_cols = []

        for c in self._date_cols:
            target_col = dates_col.loc[:, c]
            new_cols += [np.array(col_opt["fct"](target_col)) for _, col_opt in self._col_options.items()]

        return np.array(new_cols).T


class NumericalPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=None):
        pipeline_steps = [('Imputer', SimpleImputer(strategy='median')),
                          ('scaler', scaler if scaler is not None else StandardScaler())]

        self._pipeline = Pipeline(pipeline_steps)

        self._orig_scaler = scaler
        self._columns = None
        self._dtypes = None

    def fit(self, X, y=None):
        self._pipeline.fit(X, y)
        self._columns = self._get_cols_names(X)
        self._dtypes = self._get_cols_dtypes()

        return self

    def get_params(self, deep=True):
        return {"scaler": self._orig_scaler}

    def get_cols_names(self):
        return self._columns

    def get_cols_dtypes(self):
        return self._dtypes

    def _get_cols_dtypes(self):
        return ["float" for _ in self._columns]

    @staticmethod
    def _get_cols_names(X):
        return [c + '_scaled' for c in X.columns]

    def transform(self, X):
        return self._pipeline.transform(X)


class CategoricalPipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        pipeline_steps = [('Imputer', SimpleImputer(strategy='most_frequent')),
                          ('OneHotEncoder', OneHotEncoder(sparse=False))]

        self._pipeline = Pipeline(pipeline_steps)

        self._columns = None
        self._dtypes = None

    def fit(self, X, y=None):
        self._pipeline.fit(X, y)
        self._columns = self._get_cols_names()
        self._dtypes = self._get_cols_dtypes()

        return self

    def get_params(self, deep=True):
        return dict()

    def get_cols_names(self):
        return self._columns

    def get_cols_dtypes(self):
        return self._dtypes

    def _get_cols_dtypes(self):
        return ["int" for _ in self._columns]

    def _get_cols_names(self):
        return self._pipeline.named_steps['OneHotEncoder'].get_feature_names()

    def transform(self, X):
        return self._pipeline.transform(X)


class CompleteDataprepPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, cat_columns=None, num_columns=None, date_columns=None, date_config=None,
                 id_columns=None, custom_configs=None):

        self._orig_cat_columns = cat_columns
        self._orig_num_columns = num_columns
        self._orig_date_columns = date_columns
        self._orig_date_config = date_config
        self._orig_id_columns = id_columns
        self._orig_custom_configs = custom_configs

        self._transformers = []

        if cat_columns is not None:
            self._transformers.append(('categorical_pipeline', CategoricalPipeline(), cat_columns))

        if num_columns is not None:
            self._transformers.append(('numerical_pipeline', NumericalPipeline(), num_columns))

        if date_columns is not None:
            if "col_options" not in date_config.keys():
                date_config["col_options"] = None

            self._transformers.append(('date_pipeline', DatePipeline(date_format=date_config["format"],
                                                                     col_options=date_config["col_options"]),
                                       date_columns))

        if custom_configs is not None:
            self._transformers += [(c['name'], c['pipeline'], c['cols']) for c in custom_configs]

        if id_columns is not None:
            self._transformers.insert(0, ('id_columns', 'passthrough', id_columns))

        self._main = ColumnTransformer(transformers=self._transformers)
        self._columns = None
        self._dtypes = None
        self._original_dtypes = None

    def fit(self, X, y=None):
        self._main.fit(X=X, y=y)
        self._columns = self._get_cols_names()
        self._original_dtypes = X.dtypes
        self._dtypes = self._get_cols_dtypes()

        return self

    def get_params(self, deep=True):
        return {"cat_columns": self._orig_cat_columns, "num_columns": self._orig_num_columns,
                "date_columns": self._orig_date_columns, "date_config": self._orig_date_config,
                "id_columns": self._orig_id_columns, "custom_configs": self._orig_custom_configs}

    def _get_element_col_name(self, element_name, element_val):
        if isinstance(element_val, str):
            if element_val == "passthrough":
                return [t for t in self._transformers if element_name == t[0]][0][2]

            return []

        return element_val.get_cols_names()

    def _get_element_col_type(self, element_name, element_val):
        if isinstance(element_val, str):
            if element_val == "passthrough":
                return self._original_dtypes[[t for t in self._transformers if element_name == t[0]][0][2]].tolist()

            return []

        return element_val.get_cols_dtypes()

    def _get_cols_names(self):
        full_list = [self._get_element_col_name(k, v) for k, v in self._main.named_transformers_.items()]
        return flatten_list(full_list)

    def _get_cols_dtypes(self):
        full_list = [self._get_element_col_type(k, v) for k, v in self._main.named_transformers_.items()]
        return flatten_list(full_list)

    def transform(self, X):
        return pd.DataFrame({c: pd.Series(v, dtype=t) for v, c, t in zip(self._main.transform(X=X).T,
                                                                         self._columns,
                                                                         self._dtypes)})


training_data = pd.read_csv("data/train.csv")
testing_data = pd.read_csv("data/test.csv")
sample_submission = pd.read_csv("data/sampleSubmission.csv")

pip = CompleteDataprepPipeline(cat_columns=["PdDistrict"],
                               date_columns=["Dates"],
                               date_config=dict(format="%Y-%m-%d %H:%M:%S",
                                                col_options=["timestamp", "day_of_year", "month_of_year",
                                                             "hour_of_day"]),
                               num_columns=["X", "Y"])

pip.fit(training_data)
res_test = pip.transform(testing_data)
res_train = pip.transform(training_data)

id_test = testing_data["Id"]

a = 2
b = 3
