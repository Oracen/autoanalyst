from typing import Any

import pandas as pd

from autoanalyst.core.base_classes import BaseCategoryAggregator


class NoOpAggregator(BaseCategoryAggregator):
    def __init__(self, id_col: str = "id", date_col: str = "date"):
        super().__init__(id_col, date_col, preprocessors=[])

    def aggregate(self, X: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
        """
        No-op aggregation that returns the input DataFrame unchanged.
        """
        return categories

    def aggregate_raw(self, X: pd.DataFrame, categories: pd.DataFrame) -> None:
        """
        No-op raw aggregation that returns the input DataFrame unchanged.
        """
        return None
