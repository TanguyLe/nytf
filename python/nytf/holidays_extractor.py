import pandas as pd
from collections import Counter
import datetime
import holidays
from sklearn.base import BaseEstimator, TransformerMixin


class HolidayExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_col, country='US', state=None):
        self._date_col = date_col
        self._country = country
        self._state = state

        self._holidays = holidays.CountryHoliday(self._country, state=self._state)
        self._relative_holiday_importance = None

    def get_params(self, deep=True):
        return {"date_col": self._date_col, "country": self._country, "state": self._state}

    def _fit(self, X, y=None):
        max_date = X[self._date_col].max()
        min_date = X[self._date_col].min()

        delta = pd.Timedelta(max_date - min_date)
        nb_of_days = delta.days

        # Let's compute all the possible holidays over the period between the first and last date of X
        holidays = [self._holidays.get(min_date + datetime.timedelta(days=x)) for x in range(0, nb_of_days)]
        holidays = [h for h in holidays if h is not None]

        # Count how many there is of each
        nb_of_each_holiday = dict(Counter(holidays))
        # And in total
        nb_total_holidays = sum([v for _, v in nb_of_each_holiday.items()])

        # Compute all the the holidays in data
        holiday_serie = X[self._date_col].copy(False).apply(lambda d: self._holidays.get(d))
        # Of course most of the days are "normal" days
        holiday_serie.loc[pd.isna(holiday_serie)] = "normal"

        trips_count_by_holiday = holiday_serie.value_counts()

        # Then we divide each count of trips (by holiday) by the number of holidays
        # over the period (by holiday)
        for holiday, nb in nb_of_each_holiday.items():
            try:
                trips_count_by_holiday[holiday] = trips_count_by_holiday[holiday] / nb
            except KeyError:
                pass

        # For the 'normal' days, the average number of trips
        trips_count_by_holiday["normal"] = trips_count_by_holiday["normal"] / (nb_of_days - nb_total_holidays)

        # Then we finally compute the relative importance by dividing everything by the 'normal' average
        self._relative_holiday_importance = trips_count_by_holiday / trips_count_by_holiday["normal"]

        return holiday_serie

    def fit(self, X, y=None):
        self._fit(X=X, y=y)

        return self

    def _transform(self, X, holiday_serie):
        X["holidays_score"] = holiday_serie.replace(self._relative_holiday_importance)
        X["holidays_score"] = X["holidays_score"].fillna(1)

        return X

    def transform(self, X):
        holiday_serie = X[self._date_col].copy(False).apply(lambda d: self._holidays.get(d))
        return self._transform(X=X, holiday_serie=holiday_serie)

    def fit_transform(self, X, y=None):
        # Implemented just not to compute twice holiday_serie, in general it's unnecessary
        holiday_serie = self._fit(X=X, y=y)

        return self._transform(X=X, holiday_serie=holiday_serie)
