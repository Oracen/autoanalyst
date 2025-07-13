from typing import Callable

import pandas as pd

from autoanalyst.core import string_maps
from autoanalyst.core.base_classes import BaseAggregatorEnhancer


class ActivityCountEnhancer(BaseAggregatorEnhancer):
    """
    Enhancer that calculates the time since a cohort started for each category.
    It uses a start identifier function to determine the start date for each category
    and then calculates the time since that start date, normalized by the duration.
    """

    def __init__(
        self,
        event_identifier: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
        id_col: str = string_maps.ID_COL,
        date_col: str = string_maps.DATE_COL,
        name: str = string_maps.ACTIVITY_COUNT_COL,
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
        self.event_identifier = event_identifier

    def enhance(self, X: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
        # Use the identifier to get the start dates for each category
        event_timings = self.event_identifier(X, categories).astype(int)

        return event_timings.groupby(self.id_col).cumsum().rename(self.name).to_frame()
