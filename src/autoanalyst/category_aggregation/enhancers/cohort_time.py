import string
from typing import Callable

import pandas as pd

from autoanalyst.core import string_maps
from autoanalyst.core.base_classes import BaseAggregatorEnhancer


class CohortTimeEnhancer(BaseAggregatorEnhancer):
    """
    Enhancer that calculates the time since a cohort started for each category.
    It uses a start identifier function to determine the start date for each category
    and then calculates the time since that start date, normalized by the duration.
    """

    def __init__(
        self,
        start_identifier: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
        duration: pd.Timedelta,
        id_col: str = string_maps.ID_COL,
        date_col: str = string_maps.DATE_COL,
        name: str = string_maps.COHORT_TIME_COL,
    ):
        """
        Initialize the CohortTimeEnhancer.

        Parameters:
            start_identifier (Callable): Function that takes the features DataFrame
                and categories DataFrame, returning a Series with start dates for each
                category.
            duration (pd.Timedelta): Duration of the cohort period.
        """
        super().__init__(id_col, date_col)

        self.name = name
        self.start_identifier = start_identifier
        self.duration = duration

    def enhance(self, X: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
        # Use the identifier to get the start dates for each category
        start_dates = self.start_identifier(X, categories)
        end_dates = start_dates + self.duration

        date_col = pd.Series(X.index.get_level_values(self.date_col), index=X.index)
        date_col = (date_col - start_dates) / (end_dates - start_dates)

        out = date_col.rename(self.name).clip(lower=0).to_frame()
        return out
