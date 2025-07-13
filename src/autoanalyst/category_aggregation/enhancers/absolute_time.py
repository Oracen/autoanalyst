import string

import pandas as pd

from autoanalyst.core import string_maps
from autoanalyst.core.base_classes import BaseAggregatorEnhancer


class AbsoluteTimeEnhancer(BaseAggregatorEnhancer):

    def __init__(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        id_col: str = string_maps.ID_COL,
        date_col: str = string_maps.DATE_COL,
        name: str = string_maps.ABSOLUTE_TIME_COL,
    ):
        super().__init__(id_col, date_col)

        self.name = name
        self.start_date = start_date
        self.end_date = end_date

    def enhance(self, X: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
        date_col = pd.Series(X.index.get_level_values(self.date_col), index=X.index)
        date_col = (date_col - self.start_date) / (self.end_date - self.start_date)

        return date_col.rename(self.name).to_frame()
