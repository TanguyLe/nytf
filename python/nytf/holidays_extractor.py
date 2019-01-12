import pandas as pd
import holidays
from sklearn.base import BaseEstimator, TransformerMixin

HOLIDAY_EXTRACTED_FEATURES = ["holiday_score",
                              "next_holiday_dist", "next_holiday_score",
                              "prev_holiday_dist", "prev_holiday_score"]


def get_timestamp(col):
    return col.copy(False).astype("int64") // 10 ** 9


class HolidayFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_col, interest_col, country='US', state=None):
        self.date_col = date_col
        self.interest_col = interest_col
        self.country = country
        self.state = state

        self._holidays = holidays.CountryHoliday(self.country, state=self.state)
        self._holiday_scores = None

    @staticmethod
    def _get_holiday_info(date_col, days_ref):
        # And then merge it with the data

        holiday_info = pd.DataFrame(date_col)
        holiday_info["norm_date"] = holiday_info[date_col.name].dt.normalize()
        days_ref["date"] = days_ref["date"].dt.normalize()
        holiday_info = holiday_info.merge(how="left", right=days_ref,
                                          left_on="norm_date", right_on="date")
        holiday_info = holiday_info.drop(["norm_date", "date"], axis=1)
        holiday_info["next_holiday_date"] = holiday_info["next_holiday_date"] + pd.Timedelta(hours=12)
        holiday_info["prev_holiday_date"] = holiday_info["prev_holiday_date"] + pd.Timedelta(hours=12)

        return holiday_info

    def _get_days_ref(self, min_date, max_date):
        days_ref = pd.DataFrame(columns=["date", "holiday",
                                         "prev_holiday_date", "prev_holiday",
                                         "next_holiday_date", "next_holiday"])
        days_ref["date"] = pd.Series(pd.date_range(start=min_date, end=max_date, normalize=True))
        days_ref["holiday"] = days_ref["date"].copy(False).apply(lambda d: self._holidays.get(d))
        days_ref["holiday"] = days_ref["holiday"].fillna('normal')
        days_ref["next_holiday_date"] = pd.to_datetime(days_ref["next_holiday_date"])
        days_ref["prev_holiday_date"] = pd.to_datetime(days_ref["prev_holiday_date"])

        # And then for each day, the closest holiday in the past & future
        past_holiday = None
        past_holiday_date = None
        no_holiday_last_index = 0
        for idx, row in days_ref.iterrows():
            if row["holiday"] == "normal":
                if past_holiday is None:
                    continue
                days_ref.loc[idx, ["prev_holiday_date", "prev_holiday"]] = past_holiday_date, past_holiday
            else:
                # First holiday of the set
                if past_holiday is None:
                    past_holiday = row["holiday"]
                    past_holiday_date = row["date"]

                # We set the previous holiday of the holiday
                hol_info = (past_holiday_date, past_holiday)
                days_ref.loc[idx, ["prev_holiday_date", "prev_holiday"]] = hol_info

                # Prepare the holiday to be the past holiday of the next days
                past_holiday = row["holiday"]
                past_holiday_date = row["date"]

                # Then set it as the next holiday of previous days
                hol_info = (past_holiday_date, past_holiday)
                days_ref.loc[no_holiday_last_index:idx, ["next_holiday_date", "next_holiday"]] = hol_info
                no_holiday_last_index = idx

        # Last holiday of the set
        hol_info = (past_holiday_date, past_holiday)
        days_ref.loc[no_holiday_last_index:, ["next_holiday_date", "next_holiday"]] = hol_info

        return days_ref

    def _fit(self, X, y=None):
        # Compute all the the holidays in the data
        max_date = X[self.date_col].max()
        min_date = X[self.date_col].min()

        self._days_ref_df = self._get_days_ref(min_date=min_date, max_date=max_date)

        # And then merge it with the data
        holiday_info = self._get_holiday_info(days_ref=self._days_ref_df, date_col=X[self.date_col].copy(False))

        # We can use it to compute the holiday score
        df_to_group = pd.DataFrame(columns=["holiday", "holiday_score"])
        df_to_group.loc[:, "holiday"] = holiday_info["holiday"].copy(False)
        df_to_group.loc[:, "holiday_score"] = X[self.interest_col].copy(False)

        self._holiday_scores = df_to_group.groupby("holiday").mean()
        self._holiday_scores = self._holiday_scores / self._holiday_scores.loc["normal"]

        # But it also may be of some use for the transform
        return holiday_info

    def fit(self, X, y=None):
        self._fit(X=X, y=y)

        return self

    def _transform(self, X, holiday_info):
        holiday_features = holiday_info.merge(how="left",
                                              left_on=["holiday"],
                                              right_index=True,
                                              right=self._holiday_scores)
        holidays_temp = self._holiday_scores.copy()
        holidays_temp.columns = ["prev_" + str(e) for e in holidays_temp.columns]
        holiday_features = holiday_features.merge(how="left",
                                                  left_on=["prev_holiday"],
                                                  right_index=True,
                                                  right=holidays_temp)
        holidays_temp = self._holiday_scores.copy()
        holidays_temp.columns = ["next_" + str(e) for e in holidays_temp.columns]
        holiday_features = holiday_features.merge(how="left",
                                                  left_on=["next_holiday"],
                                                  right_index=True,
                                                  right=holidays_temp)

        date_col_data = X[self.date_col].copy(False)
        prev_hol_data = holiday_features["prev_holiday_date"].copy(False)
        next_hol_data = holiday_features["next_holiday_date"].copy(False)
        holiday_features["prev_holiday_dist"] = date_col_data - prev_hol_data
        holiday_features["next_holiday_dist"] = next_hol_data - date_col_data

        holiday_features["prev_holiday_dist"] = get_timestamp(holiday_features["prev_holiday_dist"])
        holiday_features["next_holiday_dist"] = get_timestamp(holiday_features["next_holiday_dist"])

        return X.assign(**holiday_features[HOLIDAY_EXTRACTED_FEATURES].copy(False))

    def transform(self, X):
        holiday_info = self._get_holiday_info(days_ref=self._days_ref_df, date_col=X[self.date_col].copy(False))
        return self._transform(X=X, holiday_info=holiday_info)

    def fit_transform(self, X, y=None, **_):
        # Implemented just not to compute twice holiday_serie, in general it's unnecessary
        holiday_info = self._fit(X=X, y=y)

        return self._transform(X=X, holiday_info=holiday_info)
